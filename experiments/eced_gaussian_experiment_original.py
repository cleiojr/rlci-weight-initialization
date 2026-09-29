#!/usr/bin/env python3
"""
ECED prototype: Entropy-Calibrated Early Descent initialization.

Working research idea:
1. Generate anisotropic Gaussian-mixture classification datasets.
2. Search a small scale factor alpha over a He/Xavier-style initialization.
3. Prefer candidates whose epoch-0 validation loss is near ln(number_of_classes)
   and whose validation loss decreases during the first few epochs.
4. Compare the selected ECED initialization against He, Xavier and Orthogonal
   across several random seeds.
5. Save detailed CSV files for later statistical analysis.

This is an experimental prototype, not an established published initializer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Regime:
    separation: float
    covariance_scale: float
    anisotropy: float
    label_noise: float


REGIMES: dict[str, Regime] = {
    "easy": Regime(
        separation=5.0,
        covariance_scale=0.70,
        anisotropy=2.0,
        label_noise=0.00,
    ),
    "medium": Regime(
        separation=3.0,
        covariance_scale=1.00,
        anisotropy=8.0,
        label_noise=0.03,
    ),
    "hard": Regime(
        separation=1.8,
        covariance_scale=1.30,
        anisotropy=20.0,
        label_noise=0.08,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an entropy-calibrated early-descent initializer on synthetic Gaussian mixtures."
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of independent seeds.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=sorted(REGIMES),
        default=["easy", "medium", "hard"],
    )
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu", "tanh"],
        default="relu",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--search-epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        help="Candidate global weight scales used by ECED.",
    )
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eced_results"))
    parser.add_argument(
        "--calibration-weight",
        type=float,
        default=1.0,
        help="Weight of |L0-ln(C)|/ln(C) in the ECED score.",
    )
    parser.add_argument(
        "--rise-weight",
        type=float,
        default=2.0,
        help="Penalty when early loss rises.",
    )
    parser.add_argument(
        "--drop-weight",
        type=float,
        default=0.50,
        help="Reward when early loss falls.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is prioritized over maximum speed.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if name == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not available:
            raise RuntimeError("MPS was requested, but it is not available.")
    return device


def _random_covariance(
    rng: np.random.Generator,
    n_features: int,
    covariance_scale: float,
    anisotropy: float,
) -> np.ndarray:
    """Create a positive-definite covariance matrix with controlled anisotropy."""
    raw = rng.normal(size=(n_features, n_features))
    q, _ = np.linalg.qr(raw)

    # Eigenvalues have a prescribed max/min ratio but random ordering.
    half_log = 0.5 * math.log(max(anisotropy, 1.0))
    eigenvalues = np.exp(np.linspace(-half_log, half_log, n_features))
    rng.shuffle(eigenvalues)
    eigenvalues *= covariance_scale**2

    return q @ np.diag(eigenvalues) @ q.T


def generate_gaussian_mixture(
    *,
    n_samples: int,
    n_features: int,
    n_classes: int,
    regime: Regime,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced, heteroscedastic, anisotropic Gaussian mixture.

    Every class has its own mean and covariance matrix. Optional label noise
    makes the medium and hard regimes less idealized.
    """
    if n_samples < n_classes * 10:
        raise ValueError("n_samples is too small for the requested number of classes.")
    if n_features < 2:
        raise ValueError("n_features must be at least 2.")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")

    rng = np.random.default_rng(seed)

    # Random class directions, normalized so separation has an interpretable scale.
    means = rng.normal(size=(n_classes, n_features))
    means /= np.linalg.norm(means, axis=1, keepdims=True).clip(min=1e-12)
    means *= regime.separation

    counts = np.full(n_classes, n_samples // n_classes, dtype=int)
    counts[: n_samples % n_classes] += 1

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for class_id, count in enumerate(counts):
        covariance = _random_covariance(
            rng,
            n_features=n_features,
            covariance_scale=regime.covariance_scale,
            anisotropy=regime.anisotropy,
        )
        samples = rng.multivariate_normal(
            mean=means[class_id],
            cov=covariance,
            size=int(count),
            check_valid="raise",
        )
        xs.append(samples.astype(np.float32))
        ys.append(np.full(int(count), class_id, dtype=np.int64))

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # Shuffle before splitting.
    permutation = rng.permutation(len(y))
    x, y = x[permutation], y[permutation]

    # Symmetric label noise.
    n_noisy = int(round(regime.label_noise * len(y)))
    if n_noisy > 0:
        noisy_indices = rng.choice(len(y), size=n_noisy, replace=False)
        offsets = rng.integers(1, n_classes, size=n_noisy)
        y[noisy_indices] = (y[noisy_indices] + offsets) % n_classes

    return x, y


def split_and_standardize(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, torch.Tensor]:
    """
    70% train, 15% validation, 15% test.
    StandardScaler is fitted only on the training split.
    """
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=seed + 1,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    return {
        "x_train": torch.from_numpy(x_train),
        "y_train": torch.from_numpy(y_train),
        "x_val": torch.from_numpy(x_val),
        "y_val": torch.from_numpy(y_val),
        "x_test": torch.from_numpy(x_test),
        "y_test": torch.from_numpy(y_test),
    }


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int],
        n_classes: int,
        activation: str,
    ) -> None:
        super().__init__()
        dims = [n_features, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_module(activation))
        layers.append(nn.Linear(dims[-1], n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@torch.no_grad()
def initialize_model(
    model: nn.Module,
    *,
    initializer: str,
    scale: float,
    activation: str,
) -> None:
    """
    Initialize hidden layers using the requested baseline.
    The classifier head uses Xavier with gain=1 to avoid activation-specific gain
    at the logits. The same global scale is applied to every weight matrix.
    """
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    hidden_gain = nn.init.calculate_gain("relu" if activation == "gelu" else activation)

    for index, layer in enumerate(linear_layers):
        is_output = index == len(linear_layers) - 1

        if is_output:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
        elif initializer == "he":
            # Kaiming/He is exact for ReLU and a practical approximation for GELU.
            nn.init.kaiming_normal_(
                layer.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
        elif initializer == "xavier":
            nn.init.xavier_normal_(layer.weight, gain=hidden_gain)
        elif initializer == "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=hidden_gain)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")

        layer.weight.mul_(scale)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def build_model(
    *,
    n_features: int,
    hidden_dims: Sequence[int],
    n_classes: int,
    activation: str,
    initializer: str,
    scale: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    # Reusing the same seed makes alpha candidates share the same underlying
    # random directions; only their scale changes.
    set_global_seed(seed)
    model = MLP(n_features, hidden_dims, n_classes, activation)
    initialize_model(
        model,
        initializer=initializer,
        scale=scale,
        activation=activation,
    )
    return model.to(device)


def build_optimizer(
    model: nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = model(x_batch)
        total_loss += float(loss_fn(logits, y_batch).item())
        total_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
        total_examples += int(y_batch.numel())

    if total_examples == 0:
        raise RuntimeError("Evaluation loader was empty.")
    return total_loss / total_examples, total_correct / total_examples


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        if not torch.isfinite(loss):
            return float("inf")

        loss.backward()
        optimizer.step()

        batch_size = int(y_batch.numel())
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def eced_score(
    *,
    initial_loss: float,
    early_loss: float,
    target_loss: float,
    calibration_weight: float,
    rise_weight: float,
    drop_weight: float,
) -> tuple[float, float, float, float]:
    """
    Lower is better.

    calibration_error = |L0 - ln(C)| / ln(C)
    normalized_change = (L_early - L0) / ln(C)

    A rise is strongly penalized. A decrease receives a capped reward so that a
    candidate cannot compensate for terrible initial calibration merely through
    a large later drop.
    """
    if not (math.isfinite(initial_loss) and math.isfinite(early_loss)):
        return float("inf"), float("inf"), float("inf"), float("inf")

    calibration_error = abs(initial_loss - target_loss) / target_loss
    normalized_change = (early_loss - initial_loss) / target_loss
    rise_penalty = max(normalized_change, 0.0)
    drop_reward = min(max(-normalized_change, 0.0), 0.50)

    score = (
        calibration_weight * calibration_error
        + rise_weight * rise_penalty
        - drop_weight * drop_reward
    )
    return score, calibration_error, rise_penalty, drop_reward


def search_eced_scale(
    *,
    data: dict[str, torch.Tensor],
    n_features: int,
    hidden_dims: Sequence[int],
    n_classes: int,
    activation: str,
    alphas: Iterable[float],
    search_epochs: int,
    batch_size: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    seed: int,
    device: torch.device,
    calibration_weight: float,
    rise_weight: float,
    drop_weight: float,
) -> tuple[float, list[dict[str, float]]]:
    pin_memory = device.type == "cuda"
    val_loader = make_loader(
        data["x_val"],
        data["y_val"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )
    target_loss = math.log(n_classes)
    rows: list[dict[str, float]] = []

    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("All alphas must be positive.")

        model = build_model(
            n_features=n_features,
            hidden_dims=hidden_dims,
            n_classes=n_classes,
            activation=activation,
            initializer="he",
            scale=float(alpha),
            seed=seed,
            device=device,
        )
        optimizer = build_optimizer(
            model,
            name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        train_loader = make_loader(
            data["x_train"],
            data["y_train"],
            batch_size=batch_size,
            shuffle=True,
            seed=seed + 10_000,
            pin_memory=pin_memory,
        )

        initial_loss, initial_accuracy = evaluate(model, val_loader, device)
        early_loss = initial_loss
        early_accuracy = initial_accuracy

        for _ in range(search_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            if not math.isfinite(train_loss):
                early_loss = float("inf")
                early_accuracy = 0.0
                break
            early_loss, early_accuracy = evaluate(model, val_loader, device)

        score, calibration_error, rise_penalty, drop_reward = eced_score(
            initial_loss=initial_loss,
            early_loss=early_loss,
            target_loss=target_loss,
            calibration_weight=calibration_weight,
            rise_weight=rise_weight,
            drop_weight=drop_weight,
        )
        rows.append(
            {
                "alpha": float(alpha),
                "target_loss": target_loss,
                "initial_val_loss": initial_loss,
                "initial_val_accuracy": initial_accuracy,
                "early_val_loss": early_loss,
                "early_val_accuracy": early_accuracy,
                "early_loss_change": early_loss - initial_loss,
                "calibration_error": calibration_error,
                "rise_penalty": rise_penalty,
                "drop_reward": drop_reward,
                "score": score,
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    calibration_tolerance = 0.10  # diferença máxima de 10% para ln(C)

    calibrated_candidates = [
        row
        for row in rows
        if row["calibration_error"] <= calibration_tolerance
    ]

    if calibrated_candidates:
        # Entre os candidatos calibrados, seleciona quem aprende mais rápido.
        best = min(
            calibrated_candidates,
            key=lambda row: (
                row["early_val_loss"],
                row["calibration_error"],
            ),
        )
    else:
        # Caso nenhum candidato cumpra a tolerância, usa o mais próximo de ln(C).
        best = min(
            rows,
            key=lambda row: (
                row["calibration_error"],
                row["early_val_loss"],
            ),
        )
    return float(best["alpha"]), rows


def run_final_training(
    *,
    method: str,
    initializer: str,
    scale: float,
    data: dict[str, torch.Tensor],
    n_features: int,
    hidden_dims: Sequence[int],
    n_classes: int,
    activation: str,
    epochs: int,
    batch_size: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        data["x_train"],
        data["y_train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 10_000,
        pin_memory=pin_memory,
    )
    train_eval_loader = make_loader(
        data["x_train"],
        data["y_train"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        data["x_val"],
        data["y_val"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        data["x_test"],
        data["y_test"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )

    model = build_model(
        n_features=n_features,
        hidden_dims=hidden_dims,
        n_classes=n_classes,
        activation=activation,
        initializer=initializer,
        scale=scale,
        seed=seed,
        device=device,
    )
    optimizer = build_optimizer(
        model,
        name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    initial_train_loss, initial_train_accuracy = evaluate(model, train_eval_loader, device)
    initial_val_loss, initial_val_accuracy = evaluate(model, val_loader, device)
    initial_test_loss, initial_test_accuracy = evaluate(model, test_loader, device)

    curve: list[dict[str, float | int | str]] = [
        {
            "method": method,
            "epoch": 0,
            "train_loss": initial_train_loss,
            "train_accuracy": initial_train_accuracy,
            "val_loss": initial_val_loss,
            "val_accuracy": initial_val_accuracy,
            "test_loss": initial_test_loss,
            "test_accuracy": initial_test_accuracy,
        }
    ]

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device)
        train_loss, train_accuracy = evaluate(model, train_eval_loader, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)

        curve.append(
            {
                "method": method,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        if not math.isfinite(val_loss):
            break

    epoch3_index = min(3, len(curve) - 1)
    last = curve[-1]
    result: dict[str, float | int | str] = {
        "method": method,
        "initializer": initializer,
        "scale": scale,
        "target_loss": math.log(n_classes),
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "initial_test_loss": initial_test_loss,
        "initial_val_accuracy": initial_val_accuracy,
        "epoch3_val_loss": float(curve[epoch3_index]["val_loss"]),
        "epoch3_val_accuracy": float(curve[epoch3_index]["val_accuracy"]),
        "early_val_drop": initial_val_loss - float(curve[epoch3_index]["val_loss"]),
        "early_descent_success": int(float(curve[epoch3_index]["val_loss"]) < initial_val_loss),
        "final_epoch": int(last["epoch"]),
        "final_train_loss": float(last["train_loss"]),
        "final_train_accuracy": float(last["train_accuracy"]),
        "final_val_loss": float(last["val_loss"]),
        "final_val_accuracy": float(last["val_accuracy"]),
        "final_test_loss": float(last["test_loss"]),
        "final_test_accuracy": float(last["test_accuracy"]),
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, curve


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby(["regime", "method"], as_index=False)
    summary = grouped.agg(
        n=("seed", "count"),
        scale_mean=("scale", "mean"),
        scale_std=("scale", "std"),
        initial_val_loss_mean=("initial_val_loss", "mean"),
        initial_val_loss_std=("initial_val_loss", "std"),
        epoch3_val_loss_mean=("epoch3_val_loss", "mean"),
        epoch3_val_loss_std=("epoch3_val_loss", "std"),
        early_val_drop_mean=("early_val_drop", "mean"),
        early_val_drop_std=("early_val_drop", "std"),
        early_descent_rate=("early_descent_success", "mean"),
        final_test_loss_mean=("final_test_loss", "mean"),
        final_test_loss_std=("final_test_loss", "std"),
        final_test_accuracy_mean=("final_test_accuracy", "mean"),
        final_test_accuracy_std=("final_test_accuracy", "std"),
    )
    return summary.sort_values(["regime", "final_test_accuracy_mean"], ascending=[True, False])


def main() -> None:
    args = parse_args()

    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1.")
    if args.search_epochs < 1:
        raise ValueError("--search-epochs must be at least 1.")
    if args.final_epochs < 3:
        raise ValueError("--final-epochs must be at least 3 to evaluate epoch 3.")
    if args.n_classes < 2:
        raise ValueError("--n-classes must be at least 2.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Target epoch-0 loss: ln({args.n_classes}) = {math.log(args.n_classes):.6f}")

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["device_resolved"] = str(device)
    config["regime_parameters"] = {name: asdict(REGIMES[name]) for name in args.regimes}
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_rows: list[dict[str, float | int | str]] = []
    search_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int | str]] = []

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    total_cases = len(args.regimes) * len(seeds)
    case_index = 0
    started = time.time()

    for regime_index, regime_name in enumerate(args.regimes):
        regime = REGIMES[regime_name]

        for seed in seeds:
            case_index += 1
            data_seed = seed + 100_000 * (regime_index + 1)
            model_seed = seed + 1_000
            print(
                f"\n[{case_index}/{total_cases}] regime={regime_name}, seed={seed}, "
                f"data_seed={data_seed}"
            )

            x, y = generate_gaussian_mixture(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_classes=args.n_classes,
                regime=regime,
                seed=data_seed,
            )
            data = split_and_standardize(x, y, seed=data_seed)

            best_alpha, candidate_rows = search_eced_scale(
                data=data,
                n_features=args.n_features,
                hidden_dims=args.hidden_dims,
                n_classes=args.n_classes,
                activation=args.activation,
                alphas=args.alphas,
                search_epochs=args.search_epochs,
                batch_size=args.batch_size,
                optimizer_name=args.optimizer,
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                seed=model_seed,
                device=device,
                calibration_weight=args.calibration_weight,
                rise_weight=args.rise_weight,
                drop_weight=args.drop_weight,
            )
            print(f"Selected ECED alpha: {best_alpha:g}")

            for row in candidate_rows:
                search_rows.append(
                    {
                        "regime": regime_name,
                        "seed": seed,
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        **row,
                    }
                )

            methods = [
                ("ECED", "he", best_alpha),
                ("He", "he", 1.0),
                ("Xavier", "xavier", 1.0),
                ("Orthogonal", "orthogonal", 1.0),
            ]

            for method, initializer, scale in methods:
                print(f"  Training {method} (scale={scale:g})")
                result, curve = run_final_training(
                    method=method,
                    initializer=initializer,
                    scale=scale,
                    data=data,
                    n_features=args.n_features,
                    hidden_dims=args.hidden_dims,
                    n_classes=args.n_classes,
                    activation=args.activation,
                    epochs=args.final_epochs,
                    batch_size=args.batch_size,
                    optimizer_name=args.optimizer,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    seed=model_seed,
                    device=device,
                )

                common = {
                    "regime": regime_name,
                    "seed": seed,
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "n_samples": args.n_samples,
                    "n_features": args.n_features,
                    "n_classes": args.n_classes,
                    "activation": args.activation,
                    "optimizer": args.optimizer,
                    "learning_rate": args.lr,
                }
                run_rows.append({**common, **result})
                curve_rows.extend({**common, **row} for row in curve)

            # Save incrementally, so an interrupted long run still preserves results.
            pd.DataFrame(search_rows).to_csv(
                args.output_dir / "alpha_search.csv", index=False
            )
            runs_df = pd.DataFrame(run_rows)
            runs_df.to_csv(args.output_dir / "runs.csv", index=False)
            pd.DataFrame(curve_rows).to_csv(
                args.output_dir / "learning_curves.csv", index=False
            )
            build_summary(runs_df).to_csv(
                args.output_dir / "summary.csv", index=False
            )

    elapsed = time.time() - started
    runs_df = pd.DataFrame(run_rows)
    summary = build_summary(runs_df)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print("\n=== SUMMARY ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False))
    print(f"\nFinished in {elapsed / 60.0:.2f} minutes.")
    print(f"Files saved in: {args.output_dir.resolve()}")
    print("Main file to send back for analysis: summary.csv")
    print("Also useful: runs.csv and alpha_search.csv")


if __name__ == "__main__":
    main()
