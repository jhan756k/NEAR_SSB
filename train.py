import os
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from config import Config
from model import SpectralSBUNet, NoiseSchedule

class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).squeeze(-1).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze(-1).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def compute_loss(model, xt, x1, x0, t, sigma_t, loss_type):
    out = model(xt, x1, t)
    if loss_type == "score":
        target = (xt - x0) / sigma_t
        return F.mse_loss(out, target)
    elif loss_type == "eps":
        target = (xt - x0) / sigma_t
        return F.mse_loss(out, target)
    elif loss_type == "l1":
        return F.l1_loss(out, x0)
    else:
        return F.mse_loss(out, x0)

def train_one_epoch(model, loader, optimizer, schedule, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:04d}/{cfg.epochs:04d} [train]", leave=False)
    for x1, x0 in progress:
        x0, x1 = x0.to(device), x1.to(device)
        t = torch.empty(x0.shape[0], device=device).uniform_(cfg.eps, 1.0 - cfg.eps)
        xt, sigma_t, _ = model.sample_xt(x0, x1, t, schedule)
        loss = compute_loss(model, xt, x1, x0, t, sigma_t, cfg.loss_type)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, schedule, device, cfg, epoch):
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch:04d}/{cfg.epochs:04d} [val]", leave=False)
    for step, (x1, x0) in enumerate(progress, start=1):
        x0, x1 = x0.to(device), x1.to(device)
        t = torch.empty(x0.shape[0], device=device).uniform_(cfg.eps, 1.0 - cfg.eps)
        xt, sigma_t, _ = model.sample_xt(x0, x1, t, schedule)
        loss = compute_loss(model, xt, x1, x0, t, sigma_t, cfg.loss_type)
        total_loss += loss.item()
        progress.set_postfix(loss=f"{(total_loss / step):.4f}")
    return total_loss / len(loader)

def train(cfg=None):
    c = cfg or Config()
    os.makedirs(c.output_dir, exist_ok=True)
    device = torch.device(c.device if torch.cuda.is_available() and c.device == "cuda" else "cpu")

    with open(os.path.join(c.data_dir, "dataset.pkl"), "rb") as f:
        dataset = pickle.load(f)

    x_train_full, y_train_full, x_test, y_test = dataset

    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3, random_state=42)
    train_loader = DataLoader(ECGDataset(x_train, y_train), batch_size=c.batch_size, shuffle=True)
    val_loader = DataLoader(ECGDataset(x_val, y_val), batch_size=c.batch_size, shuffle=False)

    model = SpectralSBUNet(c).to(device)
    schedule = NoiseSchedule(
        sigma_max=c.sigma_max,
        g_min=c.g_min,
        g_max=c.g_max,
        num_steps=c.n_schedule_steps,
        device=str(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs, eta_min=c.eta_min)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, c.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, schedule, device, c, epoch)
        val_loss = evaluate(model, val_loader, schedule, device, c, epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        tqdm.write(f"epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

        ckpt_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": vars(c),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt_payload, os.path.join(c.output_dir, "ckpt_best.pt"))
        else:
            epochs_no_improve += 1

        if epoch % c.save_every == 0 or epoch == c.epochs:
            torch.save(ckpt_payload, os.path.join(c.output_dir, f"ckpt_epoch{epoch:04d}.pt"))

        if epochs_no_improve >= c.patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch}")
            break

    best_path = os.path.join(c.output_dir, "ckpt_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    return model, schedule

if __name__ == "__main__":
    train()
