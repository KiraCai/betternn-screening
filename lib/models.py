"""
Three surrogate model architectures for Boltz-2 binding affinity prediction.

DeepEnsConcat  - feed-forward MLP, LayerNorm + GELU + dropout
BoltzNNv1Wide  - residual MLP with dual heads (regression + top-k classification)
BetterNN       - residual MLP that auto-configures by training set size

All three are trained as deep ensembles (5 members by default).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_deterministic() -> None:
    """Enable deterministic operations for reproducibility on GPU."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------------------------------------------------------
# 1. DeepEns_concat - baseline deep ensemble
# ---------------------------------------------------------------------------
class DeepEnsConcat(nn.Module):
    """Feed-forward MLP: Linear -> LayerNorm -> GELU -> Dropout per layer."""

    def __init__(
        self,
        input_size: int,
        layer_sizes: tuple[int, ...] = (1024, 512, 256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in layer_sizes:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. BoltzNN v1-wide - residual network with dual heads
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BoltzNNv1Wide(nn.Module):
    """Residual MLP with two output heads: regression + top-k probability."""

    def __init__(
        self,
        input_size: int,
        width: int = 1280,
        depth: int = 5,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(input_size, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(width, dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1),
        )
        self.top_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.inp(x))
        return self.head(h).squeeze(-1), self.top_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 3. BetterNN - auto-configuring residual network
# ---------------------------------------------------------------------------
def _auto_config(n_train: int) -> tuple[list[int], float, float]:
    """Returns (layer_sizes, dropout, ranking_loss_weight) for a given training set size."""
    if n_train < 15_000:
        return [512, 256], 0.40, 0.0
    if n_train < 40_000:
        return [768, 512, 256], 0.30, 0.0
    if n_train < 80_000:
        return [1024, 512, 256], 0.20, 0.0
    return [1024, 512, 256], 0.10, 0.30


class _BetterNNResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shortcut(x)
        h = self.dropout(F.gelu(self.fc1(x)))
        return F.gelu(self.fc2(h) + s)


class BetterNN(nn.Module):
    """Residual MLP that auto-configures depth and regularization by training set size."""

    def __init__(self, input_dim: int, layer_sizes: list[int], dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        prev = input_dim
        for sz in layer_sizes:
            blocks.append(_BetterNNResBlock(prev, sz, dropout))
            prev = sz
        self.blocks = nn.Sequential(*blocks)
        self.head_pre = nn.Linear(prev, prev // 2)
        self.head_out = nn.Linear(prev // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        return self.head_out(F.gelu(self.head_pre(h))).squeeze(-1)

    @classmethod
    def from_train_size(cls, input_dim: int, n_train: int) -> BetterNN:
        layer_sizes, dropout, _ = _auto_config(n_train)
        return cls(input_dim, layer_sizes, dropout)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def pairwise_softplus_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    max_pairs: int = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Softplus pairwise ranking loss (used by BoltzNNv1Wide)."""
    n = len(y)
    if n < 2:
        return pred.new_tensor(0.0)
    n_pairs = min(max_pairs, n * (n - 1))
    i = torch.randint(0, n, (n_pairs,), device=pred.device, generator=generator)
    j = torch.randint(0, n, (n_pairs,), device=pred.device, generator=generator)
    mask = y[i] != y[j]
    if not mask.any():
        return pred.new_tensor(0.0)
    i, j = i[mask], j[mask]
    sign = torch.sign(y[i] - y[j])
    return F.softplus(-sign * (pred[i] - pred[j])).mean()


def hinge_ranking_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    n_pairs: int = 512,
    margin: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Hinge pairwise ranking loss (used by BetterNN)."""
    n = len(y)
    if n < 2:
        return pred.new_tensor(0.0)
    i = torch.randint(0, n, (n_pairs,), device=pred.device, generator=generator)
    j = torch.randint(0, n, (n_pairs,), device=pred.device, generator=generator)
    valid = (y[i] - y[j] > margin).float()
    return (valid * F.relu(1.0 - (pred[i] - pred[j]))).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyperparameters for training a single ensemble member."""
    model_type: Literal["deepens", "boltznn", "betternn"] = "deepens"
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    cosine: bool = True
    val_frac: float = 0.2
    # BoltzNN-specific
    rank_weight: float = 0.10
    top_weight: float = 0.20
    top_frac: float = 0.05
    max_rank_pairs: int = 4096
    # BetterNN-specific
    warmup_epochs: int = 5


DEFAULT_CONFIGS = {
    "deepens": TrainConfig(
        model_type="deepens", epochs=200, batch_size=256, lr=1e-3,
        weight_decay=1e-4, patience=20, cosine=True, val_frac=0.2,
    ),
    "boltznn": TrainConfig(
        model_type="boltznn", epochs=200, batch_size=512, lr=5e-4,
        weight_decay=1e-4, patience=22, cosine=True, val_frac=0.2,
        rank_weight=0.10, top_weight=0.20, top_frac=0.05, max_rank_pairs=4096,
    ),
    "betternn": TrainConfig(
        model_type="betternn", epochs=80, batch_size=512, lr=3e-4,
        weight_decay=1e-5, patience=15, cosine=True, val_frac=0.10,
        warmup_epochs=5,
    ),
}


def _make_lr_schedule(peak_lr: float, warmup_epochs: int, total_epochs: int):
    min_lr = peak_lr / 100.0

    def lr_at(epoch: int) -> float:
        if epoch < warmup_epochs:
            return min_lr + (peak_lr - min_lr) * (epoch / max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))

    return lr_at


def train_single(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TrainConfig,
    seed: int = 0,
) -> nn.Module:
    """Train one ensemble member. Returns the best-validation model."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    n = len(y_train)
    perm = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr = torch.from_numpy(X_train[tr_idx]).to(DEVICE)
    ytr = torch.from_numpy(y_scaled[tr_idx]).to(DEVICE)
    Xva = torch.from_numpy(X_train[val_idx]).to(DEVICE)
    yva = torch.from_numpy(y_scaled[val_idx]).to(DEVICE)

    is_boltznn = cfg.model_type == "boltznn"
    is_betternn = cfg.model_type == "betternn"

    if is_boltznn:
        top_cut = np.quantile(y_train, 1.0 - cfg.top_frac)
        top_label = (y_train >= top_cut).astype(np.float32)
        ttr = torch.from_numpy(top_label[tr_idx]).to(DEVICE)
        tva = torch.from_numpy(top_label[val_idx]).to(DEVICE)
        ds = TensorDataset(Xtr, ytr, ttr)
    else:
        ds = TensorDataset(Xtr, ytr)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if is_betternn:
        schedule_fn = _make_lr_schedule(cfg.lr, cfg.warmup_epochs, cfg.epochs)
        sched = None
    elif cfg.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        schedule_fn = None
    else:
        sched = None
        schedule_fn = None

    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed + 500)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=dl_gen)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed + 1000)

    if is_betternn:
        _, _, alpha = _auto_config(n)
    else:
        alpha = 0.0

    best_state = None
    best_val = float("inf")
    bad = 0

    for ep in range(cfg.epochs):
        if schedule_fn is not None:
            for g in opt.param_groups:
                g["lr"] = schedule_fn(ep)

        model.train()
        for batch in dl:
            opt.zero_grad()
            if is_boltznn:
                xb, yb, tb = batch
                pred, top_logit = model(xb)
                mse = F.mse_loss(pred, yb)
                rank = pairwise_softplus_loss(pred, yb, cfg.max_rank_pairs, gen)
                top = F.binary_cross_entropy_with_logits(top_logit, tb)
                loss = mse + cfg.rank_weight * rank + cfg.top_weight * top
            elif is_betternn and alpha > 0.0:
                xb, yb = batch
                pred = model(xb)
                mse = F.mse_loss(pred, yb)
                rank = hinge_ranking_loss(pred, yb, n_pairs=512, margin=0.3, gen=gen)
                loss = (1.0 - alpha) * mse + alpha * rank
            else:
                xb, yb = batch
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            loss.backward()
            if is_boltznn:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            if is_boltznn:
                pv, tv = model(Xva)
                vloss = (F.mse_loss(pv, yva) + cfg.top_weight * F.binary_cross_entropy_with_logits(tv, tva)).item()
            else:
                vpred = model(Xva) if not is_boltznn else model(Xva)
                vloss = F.mse_loss(vpred, yva).item()

        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model._y_mean = y_mean  # type: ignore[attr-defined]
    model._y_std = y_std  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_single(
    model: nn.Module,
    X: np.ndarray,
    y_mean: float,
    y_std: float,
    batch_size: int = 8192,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Predict with a single model. Returns array for single-head, tuple for dual-head."""
    model.eval()
    dual = isinstance(model, BoltzNNv1Wide)
    Xt = torch.from_numpy(X).to(DEVICE)
    pred = np.empty(len(X), dtype=np.float32)
    if dual:
        top_prob = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch_size):
        chunk = Xt[i : i + batch_size]
        if dual:
            y, logit = model(chunk)
            pred[i : i + batch_size] = y.cpu().numpy() * y_std + y_mean
            top_prob[i : i + batch_size] = torch.sigmoid(logit).cpu().numpy()
        else:
            pred[i : i + batch_size] = model(chunk).cpu().numpy() * y_std + y_mean
    if dual:
        return pred, top_prob
    return pred


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------
@dataclass
class EnsembleResult:
    mean: np.ndarray
    std: np.ndarray
    top_prob: np.ndarray | None = None


def build_model(
    model_type: Literal["deepens", "boltznn", "betternn"],
    input_size: int,
    n_train: int = 0,
) -> nn.Module:
    """Instantiate a fresh model of the given type."""
    if model_type == "deepens":
        return DeepEnsConcat(input_size)
    elif model_type == "boltznn":
        return BoltzNNv1Wide(input_size)
    elif model_type == "betternn":
        return BetterNN.from_train_size(input_size, n_train)
    raise ValueError(f"Unknown model type: {model_type}")


def train_ensemble(
    model_type: Literal["deepens", "boltznn", "betternn"],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    seed: int = 42,
    n_models: int = 5,
    cfg: TrainConfig | None = None,
) -> EnsembleResult:
    """Train a deep ensemble and predict on X_pool."""
    ensure_deterministic()
    if cfg is None:
        cfg = DEFAULT_CONFIGS[model_type]

    preds = np.zeros((n_models, len(X_pool)), dtype=np.float32)
    has_top = model_type == "boltznn"
    top_probs = np.zeros((n_models, len(X_pool)), dtype=np.float32) if has_top else None

    for m in range(n_models):
        if model_type == "boltznn":
            member_seed = seed + 17 * m
        else:
            member_seed = seed * 1000 + m

        torch.manual_seed(member_seed)
        net = build_model(model_type, X_train.shape[1], len(X_train))
        net = train_single(net, X_train, y_train, cfg, seed=member_seed)
        y_mean = net._y_mean  # type: ignore[attr-defined]
        y_std = net._y_std  # type: ignore[attr-defined]

        result = predict_single(net, X_pool, y_mean, y_std)
        if has_top:
            preds[m], top_probs[m] = result  # type: ignore[misc]
        else:
            preds[m] = result  # type: ignore[assignment]

        del net
        torch.cuda.empty_cache()
        gc.collect()

    return EnsembleResult(
        mean=preds.mean(axis=0),
        std=preds.std(axis=0),
        top_prob=top_probs.mean(axis=0) if top_probs is not None else None,
    )
