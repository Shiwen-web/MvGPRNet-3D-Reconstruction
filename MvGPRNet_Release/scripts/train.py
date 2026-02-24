"""MvGPRNet training entry point."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import csv
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs import MvGPRConfig, get_run_dir
from datasets import MvGPRDataset
from models import MvGPRNet
from utils import setup_logging


def train(
    config: MvGPRConfig,
    resume: bool = False,
    seed: int = 42,
) -> str:
    """Train MvGPRNet with given config."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_dir = get_run_dir(config)
    setup_logging(run_dir)
    logger = logging.getLogger(__name__)
    logger.info("Starting training in %s (seed=%d)", run_dir, seed)
    logger.info("Configuration: %s", config)
    config.save(os.path.join(run_dir, "config.json"))

    if not os.path.exists(config.projection_folder):
        raise FileNotFoundError(
            f"Projection folder not found: {config.projection_folder}\n"
            f"Ensure n_views={config.n_views} data exists."
        )

    full_dataset = MvGPRDataset(
        config.label_folder,
        config.projection_folder,
        config.n_views,
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    logger.info("Train size: %d, Val size: %d", train_size, val_size)

    model = MvGPRNet(n_views=config.n_views).to(config.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=0.9
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume:
        checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=config.device
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            logger.info("Resumed from epoch %d", start_epoch)
        else:
            logger.warning("No checkpoint found, starting from scratch")

    csv_path = os.path.join(run_dir, "losses.csv")
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(start_epoch, config.n_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (imaging_projs, label_projs) in enumerate(train_loader):
            try:
                imaging_projs = imaging_projs.to(config.device)
                label_projs = label_projs.to(config.device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        "GPU OOM at epoch %d batch %d. Try smaller batch_size.",
                        epoch + 1,
                        batch_idx,
                    )
                    torch.cuda.empty_cache()
                    raise
                raise

            if batch_idx == 0 and epoch == 0:
                logger.info(
                    "imaging_projs range: [%.4f, %.4f]",
                    imaging_projs.min().item(),
                    imaging_projs.max().item(),
                )
                logger.info(
                    "label_projs range: [%.4f, %.4f]",
                    label_projs.min().item(),
                    label_projs.max().item(),
                )

            optimizer.zero_grad()
            recon = model(imaging_projs)
            recon_loss = F.mse_loss(recon, label_projs, reduction="mean")

            if torch.isnan(recon).any() or torch.isinf(recon).any():
                logger.error(
                    "Batch %d: recon contains NaN/inf",
                    batch_idx,
                )
                logger.error(
                    "imaging_projs: %.4f, %.4f",
                    imaging_projs.min().item(),
                    imaging_projs.max().item(),
                )
                break
            if torch.isnan(recon_loss).any():
                logger.error("Batch %d: recon_loss is NaN", batch_idx)
                logger.error(
                    "recon: %.4f, %.4f",
                    recon.min().item(),
                    recon.max().item(),
                )
                logger.error(
                    "label_projs: %.4f, %.4f",
                    label_projs.min().item(),
                    label_projs.max().item(),
                )
                break

            loss = recon_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.detach().item()

            if batch_idx % config.log_interval == 0:
                logger.info(
                    "Epoch %d/%d Batch %d/%d Loss: %.6f Recon: %.6f",
                    epoch + 1,
                    config.n_epochs,
                    batch_idx,
                    len(train_loader),
                    loss.detach().item(),
                    recon_loss.detach().item(),
                )

        avg_train_loss = total_train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d Avg Train Loss (train): %.6f LR: %.6f",
            epoch + 1,
            config.n_epochs,
            avg_train_loss,
            current_lr,
        )

        model.eval()
        total_train_loss_eval = 0.0
        with torch.no_grad():
            for imaging_projs, label_projs in train_loader:
                imaging_projs = imaging_projs.to(config.device)
                label_projs = label_projs.to(config.device)
                recon = model(imaging_projs)
                recon_loss = F.mse_loss(recon, label_projs, reduction="mean")
                total_train_loss_eval += recon_loss.detach().item()
        avg_train_loss_eval = total_train_loss_eval / len(train_loader)
        logger.info(
            "Epoch %d/%d Avg Train Loss (eval): %.6f",
            epoch + 1,
            config.n_epochs,
            avg_train_loss_eval,
        )

        total_val_loss = 0.0
        with torch.no_grad():
            for imaging_projs, label_projs in val_loader:
                imaging_projs = imaging_projs.to(config.device)
                label_projs = label_projs.to(config.device)
                recon = model(imaging_projs)
                val_loss = F.mse_loss(recon, label_projs, reduction="mean")
                total_val_loss += val_loss.detach().item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(
            "Epoch %d/%d Avg Val Loss: %.6f",
            epoch + 1,
            config.n_epochs,
            avg_val_loss,
        )
        scheduler.step(avg_val_loss)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss_eval,
                    "val_loss": avg_val_loss,
                    "lr": current_lr,
                }
            )

        model.train()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(
                "Best model saved at %s (val_loss=%.6f)",
                best_path,
                best_val_loss,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(run_dir, "checkpoint.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
            logger.info("Checkpoint saved at %s", ckpt_path)

    final_path = os.path.join(run_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    logger.info("Training completed. Final model saved at %s", final_path)
    return run_dir


def parse_args() -> tuple:
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train MvGPRNet for multi-view GPR 3D reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_views", type=int, default=16)
    parser.add_argument(
        "--label_folder", "--label_dir",
        type=str, dest="label_dir", required=True,
    )
    parser.add_argument(
        "--projection_folder", "--projection_dir",
        type=str, dest="projection_dir", required=True,
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    config = MvGPRConfig(
        n_views=args.n_views,
        device_id=args.device_id,
        label_dir=args.label_dir,
        projection_dir=args.projection_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.n_epochs,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        patience=args.patience,
    )
    return config, args.seed, args.resume


if __name__ == "__main__":
    config, seed, resume = parse_args()
    train(config, resume=resume, seed=seed)
