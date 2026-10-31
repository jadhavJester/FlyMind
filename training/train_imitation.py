"""
Imitation learning training script for BioFlight (Phase 6).
Trains an MLP policy to predict FlyBody flight control actions from kinematic state observations.
Tracks training/validation loss, learning rate, and checkpoints best model.
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_flight_dataloaders
from src.models.mlp_policy import MLPPolicy


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(X_batch)
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        running_loss += loss.item() * len(X_batch)
    return running_loss / len(dataloader.dataset)


def main():
    cfg_path = "configs/imitation.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing imitation training on device: {device}")

    # Output directories
    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    results_dir = cfg["experiment"]["results_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dataloaders
    train_loader, val_loader, test_loader, meta = get_flight_dataloaders(
        npz_path=cfg["data"]["npz_path"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )
    print(f"Loaded {len(train_loader.dataset):,} training samples, {len(val_loader.dataset):,} validation samples.")

    # 2. Model
    model = MLPPolicy(
        input_dim=cfg["model"]["input_dim"],
        action_dim=cfg["model"]["action_dim"],
        hidden_dims=cfg["model"]["hidden_dims"],
        activation=cfg["model"]["activation"],
        action_scale=cfg["model"]["action_scale"],
        dropout=cfg["model"].get("dropout", 0.0)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLP Policy Initialized with {total_params:,} trainable parameters.")

    # 3. Optimization
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    epochs = cfg["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cfg["training"].get("min_lr", 1e-5)
    )

    # 4. Training Loop
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    start_time = time.time()

    print("\nStarting Training...")
    print(f"{'Epoch':<6} | {'Train Loss (MSE)':<18} | {'Val Loss (MSE)':<16} | {'LR':<10} | Time")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)

        scheduler.step()
        ep_time = time.time() - ep_start

        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
                "mean": meta["mean"].tolist(),
                "std": meta["std"].tolist()
            }, os.path.join(ckpt_dir, "best_model.pt"))

        status = f"{epoch:<6} | {train_loss:<18.6f} | {val_loss:<16.6f} | {cur_lr:<10.6f} | {ep_time:.1f}s"
        if is_best:
            status += " [BEST]"
        print(status)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.6f}")

    # 5. Evaluate on Test Set
    best_checkpoint = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss (MSE): {test_loss:.6f}")

    # 6. Save Metrics & Curves
    metrics = {
        "experiment_name": cfg["experiment"]["name"],
        "epochs": epochs,
        "best_epoch": best_checkpoint["epoch"],
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "training_time_s": float(total_time),
        "device": str(device)
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot Loss Curve
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="#2B6CB0", lw=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="#DD6B20", lw=2)
    plt.axvline(best_checkpoint["epoch"], color="green", linestyle="--", label=f"Best Epoch ({best_checkpoint['epoch']})")
    plt.title("BioFlight — MLP Imitation Learning Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Mean Squared Error (MSE)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(results_dir, "training_loss_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved convergence plot to {curve_path}")


if __name__ == "__main__":
    main()
