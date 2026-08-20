import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import SpectralSBUNet, NoiseSchedule
from data_prep.data_prep import prepare

class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).squeeze(-1).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze(-1).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_one_epoch(model, loader, optimizer, schedule, device, eps):
    model.train()
    total_loss = 0.0
    for x1, x0 in loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        t = torch.empty(x0.shape[0], device=device).uniform_(eps, 1.0 - eps)
        xt, sigma_t, _ = model.sample_xt(x0, x1, t, schedule)
        score_pred = model(xt, t)
        target = (xt - x0) / sigma_t
        loss = F.mse_loss(score_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, schedule, device, eps):
    model.eval()
    total_loss = 0.0
    for x1, x0 in loader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        t = torch.empty(x0.shape[0], device=device).uniform_(eps, 1.0 - eps)
        xt, sigma_t, _ = model.sample_xt(x0, x1, t, schedule)
        score_pred = model(xt, t)
        target = (xt - x0) / sigma_t
        total_loss += F.mse_loss(score_pred, target).item()
    return total_loss / len(loader)

def train(
    data_dir="data_prep",
    output_dir="checkpoints",
    seg_len=512,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    t_emb_dim=128,
    num_groups=8,
    dropout=0.1,
    sqrt_h_path="data_prep/spectral_h.npy",
    sigma_max=1.0,
    g_min=1e-6,
    g_max=1.3e-4,
    n_steps=1000,
    eps=1e-4,
    lr=5e-4,
    batch_size=16,
    epochs=400,
    lr_step=90,
    lr_gamma=0.1,
    n_inference_steps=50,
    save_every=50,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    x_train, y_train, x_test, y_test = prepare()

    train_loader = DataLoader(ECGDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ECGDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, schedule, device, eps)
        val_loss = evaluate(model, test_loader, schedule, device, eps)
        scheduler.step()

        print(f"epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(output_dir, f"ckpt_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"saved {ckpt_path}")

    return model, schedule

if __name__ == "__main__":
    train()