import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from scipy.optimize import linear_sum_assignment
import os
import csv
import argparse
import yaml
from omegaconf import OmegaConf

from dataloader import get_dataloaders
from models.model import PaletteModel
from utils.checkpoint_utils import load_model_checkpoint, get_last_epoch_from_csv

save_every = 1  # epochs


def compute_mse_loss(predicted, target, mask=None):
    """
    Compute masked MSE reconstruction loss.

    Args:
        predicted: [B, S, 3] predicted palette in Lab space
        target:    [B, S, 3] ground-truth palette
        mask:      [B, S] binary mask, 1 = valid color, 0 = padded

    Returns:
        loss: scalar
    """
    diff = predicted - target
    mse = (diff**2).sum(dim=-1)  # [B, S]

    if mask is not None:
        mask = mask.float()
        loss = (mse * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = mse.mean()

    return loss


def compute_hungarian_loss(predicted, target, mask=None):
    """
    Compute Hungarian matching loss.

    Args:
        predicted: [B, S, 3] predicted palette in Lab space
        target:    [B, S, 3] ground-truth palette
        mask:      [B, S] binary mask, 1 = valid color, 0 = padded

    Returns:
        loss: scalar
    """
    B, _, _ = predicted.shape
    hungarian_losses = []
    for b in range(B):
        if mask is not None:
            valid_idx = mask[b].bool()
            pred_b = predicted[b, valid_idx]
            tgt_b = target[b, valid_idx]
        else:
            pred_b, tgt_b = predicted[b], target[b]

        # cost matrix [P, T]
        diff = pred_b[:, None, :] - tgt_b[None, :, :]
        cost = (diff**2).sum(-1).cpu().detach().numpy()

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_cost = cost[row_ind, col_ind].mean()
        hungarian_losses.append(matched_cost)

    loss = torch.tensor(hungarian_losses, device=predicted.device).mean()

    return loss


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    cfg,
    checkpoint_dir,
    log_csv_path,
):
    num_epochs = cfg.train.num_epochs
    log_last_epoch = get_last_epoch_from_csv(log_csv_path)
    if log_last_epoch is not None and num_epochs >= log_last_epoch:
        print(f"Training already completed at epoch {log_last_epoch}; skip training")
        return
    clip_norm = cfg.train.gradient_clip_norm

    lambda_mse, lambda_hungarian = (
        cfg.loss.lambda_mse,
        cfg.loss.lambda_hungarian,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = load_model_checkpoint(checkpoint_dir=checkpoint_dir, epoch=None)

    if checkpoint:
        ckpt_epoch = checkpoint["epoch"]
        if log_last_epoch is not None and log_last_epoch != ckpt_epoch:
            raise RuntimeError(
                f"Epoch mismatch: checkpoint epoch = {ckpt_epoch} csv last epoch = {log_last_epoch}"
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint; resume training from epoch {start_epoch}")
    else:
        if log_last_epoch is not None:
            raise RuntimeError(
                f"Log exists (last epoch = {log_last_epoch}) but no checkpoint found"
            )
        start_epoch = 0
        print(f"No checkpoint found; train from epoch 0")

    for epoch in range(start_epoch, num_epochs):
        loss_metric = {
            "train": {"total": 0.0, "mse": 0.0, "hungarian": 0.0},
            "val": {"total": 0.0, "mse": 0.0, "hungarian": 0.0},
        }
        fieldnames = (
            ["epoch"]
            + [f"train_{k}" for k in loss_metric["train"].keys()]
            + [f"val_{k}" for k in loss_metric["val"].keys()]
        )
        scaler = GradScaler()
        model.train()

        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in train_bar:

            input_ids = batch["input_ids"].to(device)  # [B, T]
            attention_mask = batch["attention_mask"].to(device)  # [B, T]
            palette = batch["palette"].to(device)  # [B, n_colors, 3]
            palette_mask = batch["palette_mask"].to(device)  # [B, n_colors]

            optimizer.zero_grad()

            with autocast(device_type=str(device)):

                out = model(input_ids, attention_mask, palette, palette_mask)

                mse_loss = compute_mse_loss(
                    predicted=out, target=palette, mask=palette_mask
                )
                hungarian_loss = compute_hungarian_loss(
                    predicted=out, target=palette, mask=palette_mask
                )

                loss = lambda_mse * mse_loss + lambda_hungarian * hungarian_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_metric["train"]["total"] += loss.item()
            loss_metric["train"]["mse"] += mse_loss.item()
            loss_metric["train"]["hungarian"] += hungarian_loss.item()

        model.eval()

        val_bar = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch}/{num_epochs} [Val]",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in val_bar:

                input_ids = batch["input_ids"].to(device)  # [B, T]
                attention_mask = batch["attention_mask"].to(device)  # [B, T]
                palette = batch["palette"].to(device)  # [B, n_colors, 3]
                palette_mask = batch["palette_mask"].to(device)  # [B, n_colors]

                with autocast(device_type=str(device)):
                    out = model(input_ids, attention_mask, palette, palette_mask)

                    mse_loss = compute_mse_loss(
                        predicted=out, target=palette, mask=palette_mask
                    )
                    hungarian_loss = compute_hungarian_loss(
                        predicted=out, target=palette, mask=palette_mask
                    )

                    loss = lambda_mse * mse_loss + lambda_hungarian * hungarian_loss

                loss_metric["val"]["total"] += loss.item()
                loss_metric["val"]["mse"] += mse_loss.item()
                loss_metric["val"]["hungarian"] += hungarian_loss.item()

        avg_train_metric = {
            k: v / len(train_dataloader) for k, v in loss_metric["train"].items()
        }
        avg_val_metric = {
            k: v / len(val_dataloader) for k, v in loss_metric["val"].items()
        }

        row = {
            "epoch": epoch,
            **{f"train_{k}": round(v, 6) for k, v in avg_train_metric.items()},
            **{f"val_{k}": round(v, 6) for k, v in avg_val_metric.items()},
        }

        file_exists = os.path.exists(log_csv_path)
        with open(log_csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        if (epoch + 1) % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"{checkpoint_dir}/epoch_{epoch}.pth",
            )

    print(f"Training completed at epoch {num_epochs}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="Directory to load and save model checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="Directory to load and save training logs",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset .jsonl file",
    )

    return parser.parse_args()


def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    LOG_CSV_PATH = os.path.join(args.log_dir, "logs.csv")

    with open(args.config, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    model = PaletteModel(cfg.model)

    train_dataloader, val_dataloader = get_dataloaders(
        dataset_path=args.dataset_path,
        tokenizer=model.tokenizer,
        tokenizer_input_length=model.tokenizer_input_length,
        test_split=cfg.train.test_split,
        batch_size=cfg.train.batch_size,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )

    warmup_iters = cfg.scheduler.warmup_linear.warmup_iters

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=cfg.scheduler.warmup_linear.start_factor,
        end_factor=cfg.scheduler.warmup_linear.end_factor,
        total_iters=warmup_iters,
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.scheduler.cosine.T_0,
        T_mult=cfg.scheduler.cosine.T_mult,
        eta_min=cfg.scheduler.cosine.eta_min,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters],
    )

    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        cfg,
        checkpoint_dir=args.checkpoint_dir,
        log_csv_path=LOG_CSV_PATH,
    )


if __name__ == "__main__":
    main(parse_args())
