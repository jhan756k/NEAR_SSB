import pickle
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from data_prep.data_prep import prepare
from ablation.model_white import SpectralSBUNet, NoiseSchedule

class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).squeeze(-1).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze(-1).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_one_epoch(model, loader, optimizer, schedule, device, eps, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:04d}/{total_epochs:04d} [train]", leave=False)
    for x1, x0 in progress:
        x0 = x0.to(device)
        x1 = x1.to(device)
        t = torch.empty(x0.shape[0], device=device).uniform_(eps, 1.0 - eps)
        xt, _ , _ = model.sample_xt(x0, x1, t, schedule)
        x0_pred = model(xt, x1, t)
        loss = F.mse_loss(x0_pred, x0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, schedule, device, eps, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:04d}/{total_epochs:04d} [val]", leave=False)
    for step, (x1, x0) in enumerate(progress, start=1):
        x0 = x0.to(device)
        x1 = x1.to(device)
        t = torch.empty(x0.shape[0], device=device).uniform_(eps, 1.0 - eps)
        xt, _ , _ = model.sample_xt(x0, x1, t, schedule)
        x0_pred = model(xt, x1, t)
        total_loss += F.mse_loss(x0_pred, x0).item()

        progress.set_postfix(loss=f"{(total_loss / step):.4f}")
    return total_loss / len(loader)

def train(
    output_dir="checkpoints/ablation_noise_artifact",
    
    seg_len=512,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    t_emb_dim=128,
    num_groups=8,
    dropout=0.1,

    sqrt_h_path="ablation/ablation_spectral_h/white_spectral_h.npy",

    sigma_max=0.1,
    g_min=1e-6,
    g_max=1.3e-4,
    n_steps=1000,
    eps=1e-4,

    lr=1e-3,
    batch_size=128,
    epochs=400,
    eta_min=1e-6,
    save_every=50,

    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    x_train_full, y_train_full, x_test, y_test = prepare(noise_version=1, qtdb_pkl="ablation/pkl/ma_qtdb.pkl", nstdb_pkl="ablation/pkl/ma_mitnoise.pkl", reference_rnd_test="ablation/pkl/ma_rnd_test.npy")

    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3, random_state=42)

    train_loader = DataLoader(ECGDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ECGDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    model = SpectralSBUNet(
        seg_len=seg_len,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        t_emb_dim=t_emb_dim,
        num_groups=num_groups,
        dropout=dropout,
        sqrt_h_path=sqrt_h_path,
    ).to(device)

    schedule = NoiseSchedule(
        sigma_max=sigma_max,
        g_min=g_min,
        g_max=g_max,
        num_steps=n_steps,
        device=str(device),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )

    # --- Add these variables for early stopping ---
    best_val_loss = float('inf')
    patience = 50
    epochs_no_improve = 0

    epoch_progress = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, optimizer, schedule, device, eps, epoch, epochs)
        val_loss = evaluate(model, val_loader, schedule, device, eps, epoch, epochs)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", lr=f"{current_lr:.2e}")
        tqdm.write(f"epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

        # --- Consolidated Early Stopping & Saving Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save the absolute best model immediately
            best_ckpt_path = os.path.join(output_dir, "ma_white_ckpt_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "schedule_config": {
                    "sigma_max": sigma_max,
                    "g_min": g_min,
                    "g_max": g_max,
                    "num_steps": n_steps,
                },
            }, best_ckpt_path)
            tqdm.write(f"New best val_loss! Saved {best_ckpt_path}")
        else:
            epochs_no_improve += 1

        # Original interval saving (optional, good for backups)
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(output_dir, f"ckpt_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "schedule_config": {
                    "sigma_max": sigma_max,
                    "g_min": g_min,
                    "g_max": g_max,
                    "num_steps": n_steps,
                },
            }, ckpt_path)
            tqdm.write(f"saved {ckpt_path}")

        # Trigger Early Stop
        if epochs_no_improve >= patience:
            tqdm.write(f"\nValidation loss hasn't improved for {patience} epochs. Early stopping!")
            break  # Exit loop. You will use ckpt_best.pt for inference!

    best_ckpt_path = os.path.join(output_dir, "ma_white_ckpt_best.pt")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device) # Actually load the file
        model.load_state_dict(best_ckpt["model_state"])
        
    return model, schedule

if __name__ == "__main__":
    train()