#!/usr/bin/env python3
"""
mammo-18-v3.py
==============
Multi-architecture binary classifier for VinDr-Mammo, reading from TAR shards.

Supports ResNet-18, EfficientNet-B4, and ConvNeXt-Tiny (--arch flag).
Training uses a two-phase schedule: head-only warmup (phase 1) followed by
full fine-tuning with cosine LR decay (phase 2). WeightedRandomSampler
handles class imbalance, with pos_weight derived from the effective batch
ratio after oversampling to avoid double-counting.

Usage
-----
    python mammo-18-v3.py \\
        --data-dir /path/to/vindr_tar_shards \\
        --output-dir ./output \\
        --arch convnext_tiny \\
        --wandb-project mammo-thesis \\
        --batch-size 48 \\
        --epochs 50
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms as T
from torchvision.models import (
    ResNet18_Weights,
    EfficientNet_B4_Weights,
    ConvNeXt_Tiny_Weights,
)
from tqdm import tqdm


# ── Worker-local tar cache ────────────────────────────────────────────────────
# Module-level dict is process-local: each DataLoader worker keeps its own
# open handles, so we never re-open the same shard on every __getitem__.

_WORKER_TAR_CACHE: dict[str, tarfile.TarFile] = {}


def _get_worker_tar(path: str) -> tarfile.TarFile:
    if path not in _WORKER_TAR_CACHE:
        _WORKER_TAR_CACHE[path] = tarfile.open(path, "r:*")
    return _WORKER_TAR_CACHE[path]


# ── Dataset ───────────────────────────────────────────────────────────────────

class TarShardDataset(Dataset):
    """
    Reads float32 (H×W) images and labels from TAR shards.

    Index is built from shard_manifest.csv at init time — no tar files are
    opened until the first __getitem__ in each worker process. Labels,
    manufacturer, and density come from the manifest and are kept in self.df
    for downstream stratified analysis.

    Expected directory layout (output of dicom_to_tar_shards.py):
        data_dir/
          train/*.tar   val/*.tar   test/*.tar
          metadata/shard_manifest.csv
    """

    def __init__(self, data_dir: Path, split: str, augment: bool = False) -> None:
        self.split   = split
        self.augment = augment
        self._shard_dir = data_dir / split

        manifest_path = data_dir / "metadata" / "shard_manifest.csv"
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No rows found for split='{split}' in {manifest_path}. "
                f"Available splits: {df['split'].unique().tolist()}"
            )

        # (shard_full_path, image_id, label) — resolved once, reused in workers
        self._index: list[tuple[str, str, int]] = [
            (
                str(self._shard_dir / str(row["shard"])),
                str(row["image_id"]),
                int(row["label"]),
            )
            for _, row in self.df.iterrows()
        ]
        self._labels: list[int] = [t[2] for t in self._index]

        # Augmentations: mammo-net style
        self._photometric = T.RandomApply(
            [T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
        )
        self._geometric = T.RandomApply(
            [T.RandomAffine(degrees=10, scale=(0.9, 1.1))], p=0.5
        )

    def __len__(self) -> int:
        return len(self._index)

    def get_labels(self) -> list[int]:
        """Used by build_sampler to compute per-sample weights."""
        return self._labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        shard_path, image_id, label = self._index[idx]

        tar = _get_worker_tar(shard_path)
        arr = np.load(io.BytesIO(tar.extractfile(f"{image_id}.npy").read()))

        # (1, H, W) float32 → (3, H, W) for pretrained ResNet weights
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        tensor = tensor.repeat(3, 1, 1)

        if self.augment:
            tensor = self._photometric(tensor)
            tensor = self._geometric(tensor)

        return tensor, torch.tensor(float(label), dtype=torch.float32), idx


# ── Model ─────────────────────────────────────────────────────────────────────

ARCH_HEAD_NAME = {
    "resnet18":        "fc",
    "efficientnet_b4": "classifier",
    "convnext_tiny":   "classifier",
}


def build_model(arch: str, device: torch.device, dropout: float = 0.0) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        head_in = model.fc.in_features
        model.fc = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 1))
            if dropout > 0.0 else nn.Linear(head_in, 1)
        )
    elif arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        head_in = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout if dropout > 0.0 else 0.4),
            nn.Linear(head_in, 1),
        )
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        head_in = model.classifier[-1].in_features
        new_layers = list(model.classifier.children())[:-1]
        if dropout > 0.0:
            new_layers.append(nn.Dropout(dropout))
        new_layers.append(nn.Linear(head_in, 1))
        model.classifier = nn.Sequential(*new_layers)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose resnet18, efficientnet_b4, convnext_tiny.")
    return model.to(device)


# ── Loss ──────────────────────────────────────────────────────────────────────

def build_criterion(
    train_labels: list[int], device: torch.device, oversample_rate: float = 0.0
) -> nn.Module:
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if oversample_rate > 0.0:
        # pos_weight reflects effective batch ratio after oversampling, not raw dataset ratio,
        # to avoid double-counting the positive weight.
        pos_weight_val = (1.0 - oversample_rate) / oversample_rate
    else:
        pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    print(f"  pos_weight = {pos_weight.item():.2f}  (neg={n_neg}, pos={n_pos}, oversample_rate={oversample_rate:.2f})")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_sampler(labels: list[int], target_pos_rate: float) -> WeightedRandomSampler:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = target_pos_rate / max(n_pos, 1)
    w_neg = (1.0 - target_pos_rate) / max(n_neg, 1)
    weights = [w_pos if l == 1 else w_neg for l in labels]
    print(f"  sampler: target_pos_rate={target_pos_rate:.2f}  "
          f"(each pos seen ~{target_pos_rate / max((n_pos / len(labels)), 1e-9):.1f}x per epoch)")
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def apply_label_smoothing(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    # 0 → smoothing/2,  1 → 1 - smoothing/2
    return labels * (1.0 - smoothing) + smoothing * 0.5


# ── Optimizers ────────────────────────────────────────────────────────────────

def build_phase1_optimizer(model: nn.Module, arch: str, lr_head: float) -> torch.optim.Optimizer:
    head_name = ARCH_HEAD_NAME[arch]
    head_params = [p for n, p in model.named_parameters() if head_name in n]
    return torch.optim.AdamW(head_params, lr=lr_head, weight_decay=0.0)


def build_phase2_optimizer(
    model: nn.Module, arch: str, lr_backbone: float, lr_head: float, weight_decay: float
) -> torch.optim.Optimizer:
    head_name       = ARCH_HEAD_NAME[arch]
    backbone_params = [p for n, p in model.named_parameters() if head_name not in n]
    head_params     = [p for n, p in model.named_parameters() if head_name in n]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params,     "lr": lr_head,     "weight_decay": 0.0},
    ])


# ── LR scheduler ─────────────────────────────────────────────────────────────

def build_phase2_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    lr_backbone: float,
    lr_head: float,
    lr_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    def make_lambda(base_lr: float):
        min_factor = lr_min / base_lr
        def fn(ep: int) -> float:
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            progress = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine
        return fn

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(lr_backbone), make_lambda(lr_head)],
    )


# ── BatchNorm helpers ─────────────────────────────────────────────────────────

def freeze_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (
            (probs >= lo) & (probs <= hi) if i == n_bins - 1
            else (probs >= lo) & (probs < hi)
        )
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


# ── Train / validate ──────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch_idx: int,
    label_smoothing: float = 0.0,
) -> float:
    model.train()
    if epoch_idx == 0:
        freeze_bn(model)

    total_loss = 0.0
    n = 0
    for images, labels, _ in tqdm(loader, desc=f"  train e{epoch_idx + 1}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(images).view(-1)
            targets = apply_label_smoothing(labels, label_smoothing) if label_smoothing > 0.0 else labels
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * len(images)
        n += len(images)

    return total_loss / n


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_labels: list[float] = []
    all_probs: list[float] = []
    n = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="  val  ", leave=False):
            images = images.to(device, non_blocking=True)
            labels_gpu = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images).view(-1)
                loss = criterion(logits, labels_gpu)

            probs = torch.sigmoid(logits).float().cpu().numpy()
            total_loss += loss.item() * len(images)
            n += len(images)
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    val_auc = roc_auc_score(y_true, y_prob) if len(set(all_labels)) > 1 else float("nan")
    return total_loss / n, val_auc, compute_ece(y_prob, y_true)


# ── Test evaluation ───────────────────────────────────────────────────────────

def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []
    all_probs_tta: list[float] = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="  test "):
            images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images).view(-1)
            probs = torch.sigmoid(logits).float().cpu().numpy()

            if use_tta:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits_flip = model(torch.flip(images, dims=[3])).view(-1)
                probs_tta = (probs + torch.sigmoid(logits_flip).float().cpu().numpy()) / 2
            else:
                probs_tta = probs

            all_probs.extend(probs.tolist())
            all_probs_tta.extend(probs_tta.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_prob_tta = np.array(all_probs_tta)

    def _metrics(y_t: np.ndarray, y_p: np.ndarray, tag: str = "") -> dict:
        auc = roc_auc_score(y_t, y_p)
        threshold = find_youden_threshold(y_t, y_p)
        y_pred = (y_p >= threshold).astype(int)
        tn = int(((y_pred == 0) & (y_t == 0)).sum())
        fp = int(((y_pred == 1) & (y_t == 0)).sum())
        sens = float(recall_score(y_t, y_pred, zero_division=0))
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return {
            f"auc{tag}":         auc,
            f"sensitivity{tag}": sens,
            f"specificity{tag}": spec,
            f"f1{tag}":          float(f1_score(y_t, y_pred, zero_division=0)),
            f"accuracy{tag}":    float(accuracy_score(y_t, y_pred)),
            f"ece{tag}":         compute_ece(y_p, y_t),
            f"brier{tag}":       compute_brier(y_p, y_t),
            f"threshold{tag}":   threshold,
        }

    metrics = _metrics(y_true, y_prob)
    if use_tta:
        metrics.update(_metrics(y_true, y_prob_tta, tag="_tta"))
    return metrics, y_true, y_prob, y_prob_tta


# ── Checkpoint management ─────────────────────────────────────────────────────

class TopKCheckpoints:
    def __init__(self, output_dir: Path, k: int = 3) -> None:
        self.output_dir = output_dir
        self.k = k
        self._ckpts: list[tuple[float, int, Path]] = []

    def maybe_save(self, model: nn.Module, val_auc: float, epoch: int) -> bool:
        if math.isnan(val_auc):
            return False
        if len(self._ckpts) >= self.k and val_auc <= self._ckpts[-1][0]:
            return False
        path = self.output_dir / f"ckpt_epoch{epoch:03d}_auc{val_auc:.4f}.pth"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_auc": val_auc}, path)
        self._ckpts.append((val_auc, epoch, path))
        self._ckpts.sort(key=lambda x: x[0], reverse=True)
        if len(self._ckpts) > self.k:
            _, _, old = self._ckpts.pop()
            if old.exists():
                old.unlink()
        return True

    @property
    def best_path(self) -> Path | None:
        return self._ckpts[0][2] if self._ckpts else None

    @property
    def best_auc(self) -> float:
        return self._ckpts[0][0] if self._ckpts else -1.0


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_training_curves(log_rows: list[dict], freeze_epochs: int, output_dir: Path) -> None:
    if not log_rows:
        return
    epochs     = [r["epoch"]      for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_loss   = [r["val_loss"]   for r in log_rows]
    val_auc    = [r["val_auc"]    for r in log_rows]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    ax1.plot(epochs, train_loss, color="steelblue", label="Train loss")
    ax1.plot(epochs, val_loss,   color="steelblue", linestyle="--", label="Val loss")
    ax2.plot(epochs, val_auc,    color="firebrick", label="Val AUC")
    if max(epochs) >= freeze_epochs:
        ax1.axvline(x=freeze_epochs, color="gray", linestyle=":", linewidth=1.5,
                    label=f"Phase 1→2 (epoch {freeze_epochs})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="steelblue")
    ax2.set_ylabel("AUC", color="firebrick")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right")
    ax1.set_title("Training curves")
    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    wandb.log({"charts/training_curves": wandb.Image(str(path))})


def save_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, auc: float,
    y_prob_tta: np.ndarray, auc_tta: float, save_path: Path,
) -> None:
    fpr, tpr, _         = roc_curve(y_true, y_prob)
    fpr_tta, tpr_tta, _ = roc_curve(y_true, y_prob_tta)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr,     tpr,     lw=2, label=f"No TTA  AUC={auc:.4f}")
    ax.plot(fpr_tta, tpr_tta, lw=2, linestyle="--", label=f"TTA     AUC={auc_tta:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (Test)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    wandb.log({"charts/roc_curve": wandb.Image(str(save_path))})


def save_reliability_diagram(
    y_true: np.ndarray, y_prob: np.ndarray, ece: float, save_path: Path, n_bins: int = 10,
) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf: list[float] = []; frac_pos: list[float] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (
            (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1
            else (y_prob >= lo) & (y_prob < hi)
        )
        if mask.sum() == 0:
            continue
        mean_conf.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.bar(mean_conf, frac_pos, width=0.9 / n_bins, align="center",
           alpha=0.7, label=f"ECE={ece:.4f}")
    ax.set_xlabel("Mean confidence"); ax.set_ylabel("Fraction positive")
    ax.set_title("Reliability Diagram (Test)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    wandb.log({"charts/reliability_diagram": wandb.Image(str(save_path))})


# ── Stratified analysis ───────────────────────────────────────────────────────

def stratified_analysis(
    pred_df: pd.DataFrame, group_col: str, y_true: np.ndarray, y_prob: np.ndarray,
) -> list[str]:
    if group_col not in pred_df.columns:
        return [f"  ('{group_col}' not in manifest — skipped)"]
    lines: list[str] = []
    for group, sub in pred_df.groupby(group_col):
        idx   = sub.index.tolist()
        y_t   = y_true[idx]; y_p = y_prob[idx]
        n_pos = int(y_t.sum())
        auc_str = (
            f"{roc_auc_score(y_t, y_p):.4f}" if len(np.unique(y_t)) > 1
            else "N/A (1 class)"
        )
        flag = "  *** UNDERPOWERED (<30 pos) ***" if n_pos < 30 else ""
        lines.append(
            f"  {str(group):<25}  AUC={auc_str}  n={len(sub):5d}  n_pos={n_pos:4d}{flag}"
        )
    return lines


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mammo-18-v2: ResNet-18 on VinDr-Mammo TAR shards with W&B logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",      type=Path, required=True,
                   help="Root shard dir: must contain train/, val/, test/, metadata/")
    p.add_argument("--output-dir",    type=Path, default=Path("./output/mammo18v2"))
    p.add_argument("--batch-size",    type=int,  default=256)
    p.add_argument("--epochs",        type=int,  default=50)
    p.add_argument("--freeze-epochs", type=int,  default=5,
                   help="Phase 1 duration: head-only training before backbone unfreeze")
    p.add_argument("--lr-backbone",     type=float, default=3e-4)
    p.add_argument("--lr-head",         type=float, default=1e-3)
    p.add_argument("--weight-decay",    type=float, default=1e-3,
                   help="AdamW weight decay (raised from 1e-4 to 1e-3 to combat overfitting)")
    p.add_argument("--arch",            type=str,   default="resnet18",
                   choices=["resnet18", "efficientnet_b4", "convnext_tiny"],
                   help="Model architecture")
    p.add_argument("--dropout",         type=float, default=0.3,
                   help="Dropout rate before FC head (0.0 = disabled)")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing for BCE loss (0.0 = disabled)")
    p.add_argument("--oversample-rate", type=float, default=0.20,
                   help="Target positive rate for WeightedRandomSampler (0.0 = no sampler, use shuffle)")
    p.add_argument("--warmup-epochs", type=int,   default=3)
    p.add_argument("--patience",      type=int,   default=15,
                   help="Early-stopping patience on val AUC")
    p.add_argument("--rolling-window", type=int,  default=3,
                   help="Window size for rolling-average early stopping on val AUC (1 = off)")
    p.add_argument("--num-workers",   type=int,   default=8)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--no-tta",        action="store_true",
                   help="Disable test-time augmentation (horizontal flip)")
    p.add_argument("--no-compile",    action="store_true",
                   help="Disable torch.compile")
    # W&B
    p.add_argument("--wandb-project", type=str, default="mammo-thesis")
    p.add_argument("--wandb-entity",  type=str, default=None,
                   help="W&B entity (team or username); omit to use default")
    p.add_argument("--wandb-name",    type=str, default=None,
                   help="W&B run name; omit for auto-generated name")
    p.add_argument("--wandb-offline", action="store_true",
                   help="Run W&B in offline mode (no network required; sync later with wandb sync)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=vars(args),
        mode="offline" if args.wandb_offline else "online",
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    ds_train = TarShardDataset(args.data_dir, split="train", augment=True)
    ds_val   = TarShardDataset(args.data_dir, split="val",   augment=False)
    ds_test  = TarShardDataset(args.data_dir, split="test",  augment=False)
    print(f"Split sizes — train: {len(ds_train)}, val: {len(ds_val)}, test: {len(ds_test)}")

    seed = args.seed
    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)

    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )
    if args.oversample_rate > 0.0:
        train_sampler = build_sampler(ds_train.get_labels(), args.oversample_rate)
        loader_train = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            **loader_kw,
        )
    else:
        loader_train = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kw,
        )
    loader_val  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False, **loader_kw)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # ── Model + loss ─────────────────────────────────────────────────────────
    model     = build_model(args.arch, device, dropout=args.dropout)
    criterion = build_criterion(ds_train.get_labels(), device, oversample_rate=args.oversample_rate)

    if not args.no_compile and int(torch.__version__.split(".")[0]) >= 2:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    scaler = torch.cuda.amp.GradScaler()

    # ── Phase 1: freeze backbone, train head only ─────────────────────────────
    head_name = ARCH_HEAD_NAME[args.arch]
    for name, param in model.named_parameters():
        if head_name not in name:
            param.requires_grad_(False)

    opt_phase1 = build_phase1_optimizer(model, args.arch, args.lr_head)
    print(f"\nPhase 1: head-only warmup for {args.freeze_epochs} epochs "
          f"(lr_head={args.lr_head})")

    # ── Training state ────────────────────────────────────────────────────────
    topk_ckpts        = TopKCheckpoints(args.output_dir, k=3)
    patience_cnt      = 0
    log_rows:    list[dict] = []
    opt_phase2        = None
    scheduler         = None
    val_auc_window:   list[float] = []
    best_smoothed_auc = -1.0

    for epoch in range(args.epochs):
        is_phase2 = epoch >= args.freeze_epochs

        if epoch == args.freeze_epochs:
            for param in model.parameters():
                param.requires_grad_(True)
            opt_phase2 = build_phase2_optimizer(
                model, args.arch, args.lr_backbone, args.lr_head, args.weight_decay
            )
            scheduler = build_phase2_scheduler(
                opt_phase2,
                args.epochs - args.freeze_epochs,
                args.warmup_epochs,
                args.lr_backbone,
                args.lr_head,
            )
            print(f"\nPhase 2: full fine-tuning begins (epoch {epoch + 1}/{args.epochs}, "
                  f"lr_backbone={args.lr_backbone}, lr_head={args.lr_head}, "
                  f"warmup={args.warmup_epochs} epochs)")

        optimizer = opt_phase2 if is_phase2 else opt_phase1

        train_loss              = train_epoch(model, loader_train, criterion, optimizer, scaler, device, epoch, args.label_smoothing)
        val_loss, val_auc, val_ece = validate_epoch(model, loader_val, criterion, device)

        if is_phase2 and scheduler is not None:
            scheduler.step()

        lr_head     = optimizer.param_groups[-1]["lr"]
        lr_backbone = optimizer.param_groups[0]["lr"] if is_phase2 else 0.0

        auc_str = f"{val_auc:.4f}" if not math.isnan(val_auc) else "nan"
        if not math.isnan(val_auc):
            val_auc_window.append(val_auc)
        smoothed_auc = (
            float(np.mean(val_auc_window[-args.rolling_window:]))
            if val_auc_window else float("nan")
        )
        smooth_str = f"{smoothed_auc:.4f}" if not math.isnan(smoothed_auc) else "nan"
        tqdm.write(
            f"Epoch {epoch + 1:3d}/{args.epochs}  phase={'2' if is_phase2 else '1'}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_auc={auc_str}  val_auc_smooth={smooth_str}  val_ece={val_ece:.4f}  "
            f"lr_bb={lr_backbone:.2e}  lr_hd={lr_head:.2e}"
        )

        row = {
            "epoch":            epoch + 1,
            "phase":            2 if is_phase2 else 1,
            "train_loss":       train_loss,
            "val_loss":         val_loss,
            "val_auc":          val_auc,
            "val_auc_smoothed": smoothed_auc,
            "val_ece":          val_ece,
            "lr_backbone":      lr_backbone,
            "lr_head":          lr_head,
        }
        log_rows.append(row)
        wandb.log({
            "train/loss":       train_loss,
            "val/loss":         val_loss,
            "val/auc":          val_auc,
            "val/auc_smoothed": smoothed_auc,
            "val/ece":          val_ece,
            "lr/backbone":      lr_backbone,
            "lr/head":          lr_head,
        }, step=epoch + 1)

        saved = topk_ckpts.maybe_save(model, val_auc, epoch + 1)
        if saved:
            tqdm.write(f"  ✓ checkpoint saved (best: {topk_ckpts.best_auc:.4f})")

        if not math.isnan(smoothed_auc) and smoothed_auc >= best_smoothed_auc:
            best_smoothed_auc = smoothed_auc
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                tqdm.write(f"\nEarly stopping: smoothed val AUC ({smoothed_auc:.4f}) "
                           f"not improving for {args.patience} epochs.")
                break

        if (epoch + 1) % 5 == 0:
            save_training_curves(log_rows, args.freeze_epochs, args.output_dir)

    save_training_curves(log_rows, args.freeze_epochs, args.output_dir)
    pd.DataFrame(log_rows).to_csv(args.output_dir / "training_log.csv", index=False)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint: {topk_ckpts.best_path}")
    ckpt   = torch.load(topk_ckpts.best_path, map_location=device)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(ckpt["model_state_dict"])

    use_tta = not args.no_tta
    metrics, y_true, y_prob, y_prob_tta = evaluate_test(model, loader_test, device, use_tta)
    wandb.log({f"test/{k}": v for k, v in metrics.items()})

    # Merge predictions back onto the manifest slice for stratified analysis.
    # Row order matches loader_test (shuffle=False, sequential over ds_test._index).
    pred_df = ds_test.df.copy().reset_index(drop=True)
    pred_df["true_label"]    = y_true
    pred_df["pred_prob"]     = y_prob
    pred_df["pred_prob_tta"] = y_prob_tta
    pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False)

    save_roc_curve(
        y_true, y_prob, metrics["auc"],
        y_prob_tta, metrics.get("auc_tta", metrics["auc"]),
        args.output_dir / "roc_curve.png",
    )
    ece_key = "ece_tta" if use_tta else "ece"
    save_reliability_diagram(
        y_true, y_prob_tta, metrics.get(ece_key, metrics["ece"]),
        args.output_dir / "reliability_diagram.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  TEST SET RESULTS",
        "=" * 60,
        f"  AUC-ROC         : {metrics['auc']:.4f}",
    ]
    if use_tta:
        lines.append(f"  AUC-ROC (TTA)   : {metrics['auc_tta']:.4f}")
    lines += [
        f"  Sensitivity     : {metrics['sensitivity']:.4f}  (threshold={metrics['threshold']:.3f})",
        f"  Specificity     : {metrics['specificity']:.4f}",
        f"  F1 Score        : {metrics['f1']:.4f}",
        f"  Accuracy        : {metrics['accuracy']:.4f}",
        f"  ECE (10 bins)   : {metrics['ece']:.4f}",
        f"  Brier score     : {metrics['brier']:.4f}",
        "",
        "  STRATIFIED — by Manufacturer",
    ]
    lines += stratified_analysis(pred_df, "manufacturer", y_true, y_prob_tta)
    lines += ["", "  STRATIFIED — by Density"]
    lines += stratified_analysis(pred_df, "density", y_true, y_prob_tta)
    lines += [
        "=" * 60,
        f"  Best val AUC    : {topk_ckpts.best_auc:.4f}  ({topk_ckpts.best_path})",
    ]

    summary = "\n".join(lines)
    print("\n" + summary)
    (args.output_dir / "metrics_summary.txt").write_text(summary + "\n", encoding="utf-8")

    wandb.finish()


if __name__ == "__main__":
    main()
