#!/usr/bin/env python
"""Comparable downstream evaluation with leakage-free head normalization.

Both representations use exactly the same selected samples and train/val/test
rows. Each head derives its target mean and standard deviation from that head's
training rows only. Country-head summaries are equal-country macro averages;
pooled raw-space metrics are retained separately as diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


TARGET_NORM_PROTOCOL = "train_only_per_seed_head_v1"
TARGET_NORM_DESCRIPTION = "per-head raw_z using training rows only for each seed"
from sklearn.model_selection import train_test_split


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(in_dim), int(hidden)), nn.ReLU(), nn.Linear(int(hidden), 1))

    def forward(self, x):
        return self.net(x)


def eval_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    mse = float(np.mean((pred_np - target_np) ** 2))
    return {
        "MAE": float(np.mean(np.abs(pred_np - target_np))),
        "RMSE": float(math.sqrt(mse)),
        "R2": float(r2_score(target_np, pred_np)),
        "MSE": mse,
    }


def training_target_stats(raw: np.ndarray, idx_tr: np.ndarray) -> tuple[float, float]:
    """Return leakage-free target statistics for one head and split."""
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    idx_tr = np.asarray(idx_tr, dtype=np.int64)
    if len(idx_tr) == 0:
        raise ValueError("Cannot normalize a target without training rows.")
    train = raw[idx_tr].astype(np.float64, copy=False)
    mean = float(train.mean())
    std = float(train.std())
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Non-finite training target statistics.")
    return mean, max(std, 1e-6)


def normalize_target(raw: np.ndarray, mean: float, std: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    return ((raw - np.float32(mean)) / np.float32(std)).reshape(-1, 1).astype(np.float32)


def metrics_in_raw_space(
    metrics_norm: dict,
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mean: float,
    std: float,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Promote raw-space metrics while retaining normalized diagnostics."""
    pred_raw = pred_norm * float(std) + float(mean)
    target_raw = target_norm * float(std) + float(mean)
    metrics = dict(metrics_norm)
    for name in ["MAE", "RMSE", "R2", "MSE"]:
        metrics[f"{name}_norm"] = float(metrics_norm[name])
    metrics.update(eval_metrics(pred_raw, target_raw))
    metrics["target_mean_train"] = float(mean)
    metrics["target_std_train"] = float(std)
    return metrics, pred_raw, target_raw


def load_group_labels(
    labels_path: Path,
    n: int,
    group_col: str,
    group_name_col: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    import pyarrow.parquet as pq

    schema_cols = set(pq.ParquetFile(labels_path).schema_arrow.names)
    if group_col not in schema_cols:
        raise KeyError(f"{group_col!r} not found in {labels_path}")
    cols = [group_col]
    if "sample_idx" in schema_cols:
        cols.append("sample_idx")
    if group_name_col and group_name_col != group_col and group_name_col in schema_cols:
        cols.append(group_name_col)

    labels = pd.read_parquet(labels_path, columns=cols)
    group_vals = labels[group_col].astype("string").fillna("").astype(str).to_numpy(dtype=object)
    name_vals = None
    if group_name_col == group_col:
        name_vals = group_vals.copy()
    elif group_name_col and group_name_col in labels.columns:
        name_vals = labels[group_name_col].astype("string").fillna("").astype(str).to_numpy(dtype=object)

    if "sample_idx" not in labels.columns:
        if len(labels) != n:
            raise ValueError(f"labels length {len(labels):,} does not match cache length {n:,}")
        return group_vals, name_vals

    sample_idx = labels["sample_idx"].to_numpy(dtype=np.int64, copy=False)
    if len(sample_idx) != len(group_vals):
        raise ValueError("sample_idx and group column length mismatch")
    if sample_idx.min(initial=0) < 0 or sample_idx.max(initial=-1) >= n:
        raise ValueError("sample_idx is outside cache row range")
    if len(labels) == n and np.array_equal(sample_idx, np.arange(n, dtype=np.int64)):
        return group_vals, name_vals

    aligned_groups = np.empty(n, dtype=object)
    aligned_groups[:] = ""
    aligned_groups[sample_idx] = group_vals
    aligned_names = None
    if name_vals is not None:
        aligned_names = np.empty(n, dtype=object)
        aligned_names[:] = ""
        aligned_names[sample_idx] = name_vals
    return aligned_groups, aligned_names


def group_table(
    groups: np.ndarray,
    names: np.ndarray | None,
    include_missing: bool,
    min_samples: int,
) -> pd.DataFrame:
    df = pd.DataFrame({"group": groups.astype(str)})
    if names is not None:
        df["group_name"] = names.astype(str)
    else:
        df["group_name"] = df["group"]
    if not include_missing:
        missing = df["group"].isin(["", "nan", "None", "UNKNOWN", "NA", "N/A"])
        df = df.loc[~missing]
    counts = df.groupby(["group", "group_name"], dropna=False, sort=True).size().reset_index(name="n")
    counts = counts[counts["n"] >= int(min_samples)].copy()
    return counts.sort_values(["group"]).reset_index(drop=True)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Linear(int(in_dim), 1)

    def forward(self, x):
        return self.net(x)


def build_head(in_dim: int, *, head_type: str, hidden: int) -> nn.Module:
    if head_type == "linear":
        return LinearHead(int(in_dim))
    if head_type == "mlp":
        return MLPHead(int(in_dim), hidden=int(hidden))
    raise ValueError(f"Unsupported head_type={head_type!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.int64)
    if n <= 0:
        raise ValueError("Cannot split an empty group.")
    if n < 5:
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(idx).astype(np.int64)
        if n == 1:
            return idx, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if n == 2:
            return idx[:1], np.empty(0, dtype=np.int64), idx[1:]
        return idx[: n - 2], idx[n - 2:n - 1], idx[n - 1:]
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=seed)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.1, random_state=seed)
    return idx_tr.astype(np.int64), idx_va.astype(np.int64), idx_te.astype(np.int64)


def make_block_split(
    block_ids: np.ndarray,
    seed: int,
    *,
    test_frac: float = 0.2,
    val_frac_of_train: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_ids = np.asarray(block_ids)
    n = int(len(block_ids))
    if n <= 0:
        raise ValueError("Cannot split an empty block array.")
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 2:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    blocks = rng.permutation(unique_blocks)
    n_test = max(1, int(round(len(blocks) * float(test_frac))))
    n_test = min(n_test, len(blocks) - 1)
    test_blocks = blocks[:n_test]
    remaining = blocks[n_test:]
    if len(remaining) >= 2:
        n_val = max(1, int(round(len(remaining) * float(val_frac_of_train))))
        n_val = min(n_val, len(remaining) - 1)
    else:
        n_val = 0
    val_blocks = remaining[:n_val]
    train_blocks = remaining[n_val:]

    idx = np.arange(n, dtype=np.int64)
    idx_tr = idx[np.isin(block_ids, train_blocks)]
    idx_va = idx[np.isin(block_ids, val_blocks)] if len(val_blocks) else np.empty(0, dtype=np.int64)
    idx_te = idx[np.isin(block_ids, test_blocks)]
    return idx_tr.astype(np.int64), idx_va.astype(np.int64), idx_te.astype(np.int64)


def spatial_block_ids(lon: np.ndarray, lat: np.ndarray, block_km: float) -> np.ndarray:
    if float(block_km) <= 0:
        raise ValueError(f"spatial block size must be positive, got {block_km}")
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("spatial_block split requires pyproj to project lon/lat to meters.") from exc

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x, y = transformer.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    block_m = float(block_km) * 1000.0
    bx = np.floor(np.asarray(x) / block_m).astype(np.int64)
    by = np.floor(np.asarray(y) / block_m).astype(np.int64)
    bx0 = bx - int(bx.min())
    by0 = by - int(by.min())
    width = int(by0.max()) + 1
    return (bx0 * np.int64(width) + by0).astype(np.int64)


def fit_predict(
    X: np.ndarray,
    Y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    hidden: int,
    head_type: str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    X_tr = torch.from_numpy(np.ascontiguousarray(X[idx_tr], dtype=np.float32))
    Y_tr = torch.from_numpy(np.ascontiguousarray(Y[idx_tr], dtype=np.float32))
    X_va = torch.from_numpy(np.ascontiguousarray(X[idx_va], dtype=np.float32))
    Y_va = torch.from_numpy(np.ascontiguousarray(Y[idx_va], dtype=np.float32))
    X_te = torch.from_numpy(np.ascontiguousarray(X[idx_te], dtype=np.float32))
    Y_te = torch.from_numpy(np.ascontiguousarray(Y[idx_te], dtype=np.float32))

    model = build_head(int(X.shape[1]), head_type=str(head_type), hidden=int(hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    actual_epochs = 0
    wait = 0
    n_tr = int(X_tr.shape[0])
    n_va = int(X_va.shape[0])
    n_te = int(X_te.shape[0])
    train_steps = math.ceil(n_tr / batch_size)

    for ep in range(1, int(epochs) + 1):
        actual_epochs = int(ep)
        model.train()
        perm = torch.randperm(n_tr, generator=generator)
        train_loss = 0.0
        train_seen = 0
        for start in range(0, n_tr, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = X_tr[batch_idx].to(device, non_blocking=True)
            yb = Y_tr[batch_idx].to(device, non_blocking=True)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = int(batch_idx.shape[0])
            train_loss += float(loss.item()) * bs
            train_seen += bs

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for start in range(0, n_va, batch_size):
                xb = X_va[start:start + batch_size].to(device, non_blocking=True)
                yb = Y_va[start:start + batch_size].to(device, non_blocking=True)
                loss = loss_fn(model(xb), yb)
                bs = int(xb.shape[0])
                val_loss += float(loss.item()) * bs
                val_seen += bs
        val = val_loss / max(1, val_seen)
        if val + 1e-8 < best_val:
            best_val = val
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = []
    tgts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_te, batch_size):
            xb = X_te[start:start + batch_size].to(device, non_blocking=True)
            preds.append(model(xb).cpu())
            tgts.append(Y_te[start:start + batch_size])
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    metrics = eval_metrics(pred, tgt)
    metrics.update(
        {
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "actual_epochs": int(actual_epochs),
            "n_train": n_tr,
            "n_val": n_va,
            "n_test": n_te,
            "train_steps": int(train_steps),
        }
    )
    return metrics, pred, tgt


def fit_predict_indexed(
    X_source: np.ndarray,
    Y: np.ndarray,
    source_idx: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    hidden: int,
    head_type: str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Train/evaluate without materializing very large feature matrices.

    `idx_*` are local positions into `source_idx` and `Y`; `source_idx` maps
    those local positions back to rows in the memory-mapped feature array.
    This keeps large country heads, especially 128-d AETHER population heads,
    below RAM limits.
    """

    set_seed(seed)
    source_idx = np.asarray(source_idx, dtype=np.int64)
    Y = np.asarray(Y, dtype=np.float32)
    model = build_head(int(X_source.shape[1]), head_type=str(head_type), hidden=int(hidden)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    actual_epochs = 0
    wait = 0
    n_tr = int(len(idx_tr))
    n_va = int(len(idx_va))
    n_te = int(len(idx_te))
    train_steps = math.ceil(n_tr / batch_size)
    use_stream_batches = n_tr >= 10_000_000
    if use_stream_batches:
        print(
            f"[STREAM] indexed large-group batches n_train={n_tr:,} batch_size={batch_size} steps={train_steps:,}",
            flush=True,
        )

    def batch_xy(local_positions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        local_positions = np.asarray(local_positions, dtype=np.int64)
        src = source_idx[local_positions]
        order = np.argsort(src, kind="mergesort")
        src_sorted = src[order]
        pos_sorted = local_positions[order]
        xb_np = np.ascontiguousarray(X_source[src_sorted], dtype=np.float32)
        yb_np = np.ascontiguousarray(Y[pos_sorted], dtype=np.float32)
        return torch.from_numpy(xb_np), torch.from_numpy(yb_np)

    for ep in range(1, int(epochs) + 1):
        actual_epochs = int(ep)
        model.train()
        if use_stream_batches:
            batch_order = torch.randperm(train_steps, generator=generator).numpy()
        else:
            perm = torch.randperm(n_tr, generator=generator).numpy()
        train_loss = 0.0
        train_seen = 0
        for step in range(train_steps):
            if use_stream_batches:
                batch_id = int(batch_order[step])
                start = batch_id * batch_size
                batch_local = idx_tr[start:start + batch_size]
            else:
                start = step * batch_size
                batch_local = idx_tr[perm[start:start + batch_size]]
            xb, yb = batch_xy(batch_local)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = int(len(batch_local))
            train_loss += float(loss.item()) * bs
            train_seen += bs

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for start in range(0, n_va, batch_size):
                batch_local = idx_va[start:start + batch_size]
                xb, yb = batch_xy(batch_local)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                loss = loss_fn(model(xb), yb)
                bs = int(len(batch_local))
                val_loss += float(loss.item()) * bs
                val_seen += bs
        val = val_loss / max(1, val_seen)
        if val + 1e-8 < best_val:
            best_val = val
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = []
    tgts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_te, batch_size):
            batch_local = idx_te[start:start + batch_size]
            xb, yb = batch_xy(batch_local)
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).cpu())
            tgts.append(yb)
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    metrics = eval_metrics(pred, tgt)
    metrics.update(
        {
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "actual_epochs": int(actual_epochs),
            "n_train": n_tr,
            "n_val": n_va,
            "n_test": n_te,
            "train_steps": int(train_steps),
        }
    )
    return metrics, pred, tgt


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def sample_array(samples, key: str, cache_idx: np.ndarray, default=np.nan) -> np.ndarray:
    if key in samples.files:
        return np.asarray(samples[key][cache_idx])
    return np.full(len(cache_idx), default)


def label_values(labels_df: pd.DataFrame, cache_idx: np.ndarray, col: str, default: str = "") -> np.ndarray:
    if col in labels_df.columns:
        return labels_df.iloc[cache_idx][col].to_numpy()
    return np.full(len(cache_idx), default, dtype=object)


def build_prediction_frame(
    *,
    samples,
    labels_df: pd.DataFrame,
    union_idx: np.ndarray,
    union_positions: np.ndarray,
    pred: torch.Tensor,
    tgt: torch.Tensor,
    y_mean: float,
    y_std: float,
    task: str,
    region: str,
    method: str,
    head_scope: str,
    head_group: str,
    head_group_name: str,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    pos = np.asarray(union_positions, dtype=np.int64)
    cache_idx = np.asarray(union_idx[pos], dtype=np.int64)
    pred_norm = pred.detach().cpu().numpy().reshape(-1).astype(np.float32)
    y_norm = tgt.detach().cpu().numpy().reshape(-1).astype(np.float32)
    y_raw = y_norm * np.float32(y_std) + np.float32(y_mean)
    pred_raw = pred_norm * np.float32(y_std) + np.float32(y_mean)

    if "sample_idx" in labels_df.columns:
        sample_idx = labels_df.iloc[cache_idx]["sample_idx"].to_numpy()
    else:
        sample_idx = cache_idx

    return pd.DataFrame(
        {
            "task": task,
            "region": region,
            "method": method,
            "head_scope": head_scope,
            "head_group": head_group,
            "head_group_name": head_group_name,
            "model": model_name,
            "seed": int(seed),
            "split": "test",
            "sample_idx": sample_idx,
            "cache_idx": cache_idx,
            "union_pos": pos,
            "lon": sample_array(samples, "lon", cache_idx),
            "lat": sample_array(samples, "lat", cache_idx),
            "row": sample_array(samples, "row", cache_idx),
            "col": sample_array(samples, "col", cache_idx),
            "country_code": label_values(labels_df, cache_idx, "country_code"),
            "country_name": label_values(labels_df, cache_idx, "country_name"),
            "y_norm": y_norm,
            "pred_norm": pred_norm,
            "residual_norm": pred_norm - y_norm,
            "abs_error_norm": np.abs(pred_norm - y_norm),
            "y_raw": y_raw,
            "pred_raw": pred_raw,
            "residual_raw": pred_raw - y_raw,
            "abs_error_raw": np.abs(pred_raw - y_raw),
        }
    )


def write_predictions(path: Path, frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def one_cache_file(cache_dir: Path, pattern: str) -> Path:
    files = sorted(cache_dir.glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern} in {cache_dir}, found {len(files)}")
    return files[0]


def build_group_splits(
    labels: np.ndarray,
    names: np.ndarray | None,
    *,
    sample_frac: float,
    sample_seed: int,
    min_effective_samples: int,
) -> tuple[list[dict], list[dict]]:
    groups_df = group_table(labels.astype(str), names, include_missing=False, min_samples=1)
    rng = np.random.default_rng(int(sample_seed))
    records = []
    dropped = []
    offset = 0
    for row in groups_df.itertuples(index=False):
        group = str(row.group)
        group_name = str(row.group_name)
        source_idx = np.flatnonzero(labels.astype(str) == group).astype(np.int64)
        n_source = int(len(source_idx))
        take = max(1, int(math.ceil(n_source * float(sample_frac))))
        selected = rng.choice(source_idx, size=take, replace=False).astype(np.int64)
        selected.sort()
        if len(selected) < int(min_effective_samples):
            dropped.append(
                {
                    "group": group,
                    "group_name": group_name,
                    "n_source": n_source,
                    "n": int(len(selected)),
                    "reason": f"n < min_effective_samples ({int(min_effective_samples)})",
                }
            )
            continue
        records.append(
            {
                "group": group,
                "group_name": group_name,
                "n_source": n_source,
                "n": int(len(selected)),
                "cache_idx": selected,
                "offset": offset,
            }
        )
        offset += int(len(selected))
    return records, dropped


def summarize_metrics(metrics_csv: Path, out_dir: Path, aether_model: str) -> None:
    df = pd.read_csv(metrics_csv)
    metric_cols = ["MAE", "RMSE", "R2", "MSE", "best_val"]
    agg = (
        df.groupby(["method", "model"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([x for x in col if x]) for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "summary_mean.csv", index=False)

    pivot = agg.pivot_table(index="method", columns="model", values="R2_mean", aggfunc="first").reset_index()
    if "AlphaEarth64" in pivot.columns and aether_model in pivot.columns:
        pivot["delta_AETHER_minus_AE"] = pivot[aether_model] - pivot["AlphaEarth64"]
    pivot.to_csv(out_dir / "r2_pivot.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "metrics_csv": str(metrics_csv),
                "summary_mean_csv": str(out_dir / "summary_mean.csv"),
                "r2_pivot_csv": str(out_dir / "r2_pivot.csv"),
                "n_rows": int(len(df)),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--labels-parquet", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--group-name-col", default="")
    ap.add_argument(
        "--merge-group-codes",
        default="",
        help="Comma-separated group codes to merge before splitting and normalization; first code is canonical.",
    )
    ap.add_argument(
        "--merge-group-name",
        default="",
        help="Display name for a merged group.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", default="")
    ap.add_argument("--region", default="")
    ap.add_argument("--head-scope-label", default="")
    ap.add_argument("--seeds", default="42,24,7,0,100")
    ap.add_argument("--models", default="AlphaEarth64,AETHER128")
    ap.add_argument("--aether-model", default="AETHER128")
    ap.add_argument("--aether-cache-glob", default="*.aether128_exact.npy")
    ap.add_argument(
        "--skip-global-models",
        default="",
        help="Comma-separated model names to skip for the global-head method.",
    )
    ap.add_argument(
        "--skip-group-models",
        default="",
        help="Comma-separated model names to skip for the group-head method.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size; task schedulers must apply the registered task-specific value.",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--head-type", choices=["mlp", "linear"], default="mlp")
    ap.add_argument("--sample-frac-per-group", type=float, default=0.1)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--min-effective-samples", type=int, default=1)
    ap.add_argument("--split-mode", choices=["random", "spatial_block"], default="spatial_block")
    ap.add_argument("--spatial-block-km", type=float, default=1.0)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--predictions-dir", default="")
    ap.add_argument(
        "--materialize-group-features",
        action="store_true",
        help="Load the selected feature rows into RAM before country-head training.",
    )
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    region_csv = out_dir / "region_metrics.csv"

    print(f"[LOAD] cache_dir={cache_dir}", flush=True)
    npz_path = one_cache_file(cache_dir, "*.npz")
    ae_path = one_cache_file(cache_dir, "*.ae64.npy")
    aether_path = one_cache_file(cache_dir, args.aether_cache_glob)
    print(f"[LOAD] npz={npz_path.name} ae={ae_path.name} aether={aether_path.name}", flush=True)
    samples = np.load(npz_path)
    if "raw" not in samples:
        raise KeyError("Cache npz must contain raw target values.")
    X_ae = np.load(ae_path, mmap_mode="r")
    X_aether = np.load(aether_path, mmap_mode="r")
    n = int(X_ae.shape[0])
    labels, names = load_group_labels(
        Path(args.labels_parquet),
        n,
        args.group_col,
        args.group_name_col.strip() or None,
    )
    print(f"[LOAD] group labels loaded n={n:,}", flush=True)
    labels_df = pd.read_parquet(args.labels_parquet)
    if len(labels_df) != n:
        raise ValueError(f"labels parquet rows ({len(labels_df)}) != cache rows ({n})")
    print("[LOAD] prediction labels dataframe loaded", flush=True)
    labels = labels.astype(str)
    merge_codes = [x.strip() for x in args.merge_group_codes.split(",") if x.strip()]
    if not merge_codes and args.region == "china" and args.head_scope_label == "country":
        merge_codes = ["CHN", "TWN", "HKG", "MAC"]
        if not args.merge_group_name.strip():
            args.merge_group_name = "China"
    if merge_codes:
        if len(merge_codes) < 2:
            raise ValueError("--merge-group-codes requires at least two codes")
        canonical = merge_codes[0]
        merge_mask = np.isin(labels, np.asarray(merge_codes, dtype=object))
        labels[merge_mask] = canonical
        if names is not None:
            merged_name = args.merge_group_name.strip() or canonical
            names[merge_mask] = merged_name
        print(
            f"[MERGE] groups={','.join(merge_codes)} -> {canonical} "
            f"n={int(merge_mask.sum()):,} before split/normalization",
            flush=True,
        )
    raw_all = np.asarray(samples["raw"], dtype=np.float32)
    valid_target_mask = np.isfinite(raw_all)
    labels_for_split = labels.copy()
    labels_for_split[~valid_target_mask] = ""
    groups, dropped_groups = build_group_splits(
        labels_for_split,
        names,
        sample_frac=float(args.sample_frac_per_group),
        sample_seed=int(args.sample_seed),
        min_effective_samples=int(args.min_effective_samples),
    )
    if not groups:
        raise ValueError("No groups selected.")
    union_idx = np.concatenate([g["cache_idx"] for g in groups]).astype(np.int64)
    print(
        f"[FILTER] valid_target={int(valid_target_mask.sum()):,}/{n:,} selected={len(union_idx):,} groups={len(groups)}",
        flush=True,
    )
    raw_union = np.asarray(raw_all[union_idx], dtype=np.float32)
    if not np.isfinite(raw_union).all():
        raise ValueError("Non-finite target values remain after target-valid filtering.")
    union_block_ids = None
    if args.split_mode == "spatial_block":
        if "lon" not in samples or "lat" not in samples:
            raise KeyError("spatial_block split requires lon and lat arrays in cache npz.")
        union_block_ids = spatial_block_ids(
            np.asarray(samples["lon"][union_idx], dtype=np.float64),
            np.asarray(samples["lat"][union_idx], dtype=np.float64),
            float(args.spatial_block_km),
        )
        print(f"[SPLIT] spatial blocks={len(np.unique(union_block_ids)):,}", flush=True)

    group_meta_rows = [
        {
            "group": g["group"],
            "group_name": g["group_name"],
            "n_source": g["n_source"],
            "n": g["n"],
            "offset": g["offset"],
        }
        for g in groups
    ]
    pd.DataFrame(group_meta_rows).to_csv(out_dir / "groups.csv", index=False)
    dropped_cols = ["group", "group_name", "n_source", "n", "reason"]
    pd.DataFrame(dropped_groups, columns=dropped_cols).to_csv(out_dir / "dropped_groups.csv", index=False)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    skip_global_models = {m.strip() for m in args.skip_global_models.split(",") if m.strip()}
    skip_group_models = {m.strip() for m in args.skip_group_models.split(",") if m.strip()}
    valid_models = {"AlphaEarth64": X_ae, str(args.aether_model): X_aether}
    unknown = [m for m in requested_models if m not in valid_models]
    if unknown:
        raise ValueError(f"Unknown models requested: {unknown}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pred_dir = Path(args.predictions_dir) if args.predictions_dir else (out_dir / "predictions")
    materialize_group_features = bool(args.materialize_group_features) and args.task != "pop"
    if args.materialize_group_features and args.task == "pop":
        print("[MATERIALIZE] disabled for POP; using streaming country-head batches", flush=True)

    done = set()
    if args.resume and metrics_csv.exists():
        meta_path = out_dir / "meta.json"
        if not meta_path.is_file():
            raise RuntimeError(
                f"Cannot resume {out_dir}: metrics.csv exists without meta.json."
            )
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        existing_protocol = existing_meta.get("target_norm_protocol")
        if existing_protocol != TARGET_NORM_PROTOCOL:
            raise RuntimeError(
                "Cannot mix downstream normalization protocols while resuming: "
                f"found protocol={existing_protocol!r}, "
                f"target_norm={existing_meta.get('target_norm')!r}; "
                f"required protocol={TARGET_NORM_PROTOCOL!r}."
            )
        expected_protocol = {
            "task": str(args.task),
            "region": str(args.region),
            "head_scope_label": str(args.head_scope_label),
            "group_col": str(args.group_col),
            "seeds": seeds,
            "models": requested_models,
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "hidden": int(args.hidden),
            "head_type": str(args.head_type),
            "split_mode": str(args.split_mode),
            "spatial_block_km": float(args.spatial_block_km) if args.split_mode == "spatial_block" else None,
            "sample_frac_per_group": float(args.sample_frac_per_group),
            "sample_seed": int(args.sample_seed),
            "min_effective_samples": int(args.min_effective_samples),
        }
        mismatches = {
            key: {"existing": existing_meta.get(key), "requested": value}
            for key, value in expected_protocol.items()
            if existing_meta.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Cannot resume with different downstream protocol settings: {mismatches}"
            )
        old = pd.read_csv(metrics_csv)
        done = set((str(r.method), str(r.model), int(r.seed)) for r in old.itertuples(index=False))

    meta = {
        "cache_dir": str(cache_dir),
        "npz": str(npz_path),
        "ae64": str(ae_path),
        "aether": str(aether_path),
        "aether_model": str(args.aether_model),
        "labels_parquet": str(args.labels_parquet),
        "task": str(args.task),
        "region": str(args.region),
        "head_scope_label": str(args.head_scope_label),
        "group_col": args.group_col,
        "group_name_col": args.group_name_col,
        "merge_group_codes": merge_codes,
        "merge_group_name": args.merge_group_name.strip() or (merge_codes[0] if merge_codes else ""),
        "n_cache": n,
        "n_valid_target": int(valid_target_mask.sum()),
        "n_excluded_invalid_target": int((~valid_target_mask).sum()),
        "n_selected": int(len(union_idx)),
        "n_groups": int(len(groups)),
        "n_dropped_groups": int(len(dropped_groups)),
        "dropped_groups": dropped_groups,
        "sample_frac_per_group": float(args.sample_frac_per_group),
        "sample_seed": int(args.sample_seed),
        "min_effective_samples": int(args.min_effective_samples),
        "split_mode": str(args.split_mode),
        "spatial_block_km": float(args.spatial_block_km) if args.split_mode == "spatial_block" else None,
        "n_spatial_blocks": int(len(np.unique(union_block_ids))) if union_block_ids is not None else None,
        "target_norm": TARGET_NORM_DESCRIPTION,
        "target_norm_protocol": TARGET_NORM_PROTOCOL,
        "metric_space": "raw target units; normalized metrics carry a _norm suffix",
        "group_summary": "equal-group macro average; pooled raw metrics are diagnostic only",
        "seeds": seeds,
        "models": requested_models,
        "skip_global_models": sorted(skip_global_models),
        "skip_group_models": sorted(skip_group_models),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden": int(args.hidden),
        "head_type": str(args.head_type),
        "save_predictions": bool(args.save_predictions),
        "predictions_dir": str(pred_dir),
        "materialize_group_features": materialize_group_features,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for model_name in requested_models:
        need_global = (
            model_name not in skip_global_models
            and any(("global_head", model_name, int(seed)) not in done for seed in seeds)
        )
        if need_global:
            print(f"\n[MODEL] {model_name} materialize union n={len(union_idx):,}", flush=True)
            X = np.ascontiguousarray(valid_models[model_name][union_idx], dtype=np.float32)
        else:
            print(f"\n[MODEL] {model_name} stream group heads n={len(union_idx):,}", flush=True)
            X = None
        if materialize_group_features and any(
            (model_name, int(seed)) not in done for seed in seeds
        ):
            print(f"[MATERIALIZE] country-head features n={len(union_idx):,}", flush=True)
            X = np.ascontiguousarray(valid_models[model_name][union_idx], dtype=np.float32)
        for seed in seeds:
            group_split_rows = []
            global_tr = []
            global_va = []
            global_te = []
            if args.split_mode == "spatial_block":
                idx_tr_all, idx_va_all, idx_te_all = make_block_split(union_block_ids, int(seed))
                split_name = np.full(len(union_idx), "train", dtype=object)
                split_name[idx_va_all] = "val"
                split_name[idx_te_all] = "test"
                for g in groups:
                    off = int(g["offset"])
                    sl = slice(off, off + int(g["n"]))
                    local_split = split_name[sl]
                    group_split_rows.append({
                        **{k: g[k] for k in ["group", "group_name", "n_source", "n", "offset"]},
                        "seed": seed,
                        "n_train": int((local_split == "train").sum()),
                        "n_val": int((local_split == "val").sum()),
                        "n_test": int((local_split == "test").sum()),
                    })
            else:
                for g in groups:
                    idx_tr, idx_va, idx_te = make_split(int(g["n"]), int(seed))
                    off = int(g["offset"])
                    global_tr.append(idx_tr + off)
                    global_va.append(idx_va + off)
                    global_te.append(idx_te + off)
                    group_split_rows.append({**{k: g[k] for k in ["group", "group_name", "n_source", "n", "offset"]}, "seed": seed, "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te)})
                idx_tr_all = np.concatenate(global_tr).astype(np.int64)
                idx_va_all = np.concatenate(global_va).astype(np.int64)
                idx_te_all = np.concatenate(global_te).astype(np.int64)
            if len(idx_tr_all) == 0 or len(idx_te_all) == 0:
                raise ValueError(
                    f"Split for seed={seed} has train={len(idx_tr_all)} test={len(idx_te_all)}; "
                    "need at least one train and test sample."
                )

            key = ("global_head", model_name, int(seed))
            if model_name not in skip_global_models and key not in done:
                if X is None:
                    X = np.ascontiguousarray(valid_models[model_name][union_idx], dtype=np.float32)
                print(f"[TRAIN] method=global_head model={model_name} seed={seed} train={len(idx_tr_all):,}", flush=True)
                y_mean, y_std = training_target_stats(raw_union, idx_tr_all)
                Y_head = normalize_target(raw_union, y_mean, y_std)
                metrics_norm, pred_all, tgt_all = fit_predict(
                    X,
                    Y_head,
                    idx_tr_all,
                    idx_va_all,
                    idx_te_all,
                    seed=int(seed),
                    device=device,
                    epochs=int(args.epochs),
                    patience=int(args.patience),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    hidden=int(args.hidden),
                    head_type=str(args.head_type),
                )
                metrics, _, _ = metrics_in_raw_space(
                    metrics_norm, pred_all, tgt_all, y_mean, y_std
                )
                if args.save_predictions:
                    frame = build_prediction_frame(
                        samples=samples,
                        labels_df=labels_df,
                        union_idx=union_idx,
                        union_positions=idx_te_all,
                        pred=pred_all,
                        tgt=tgt_all,
                        y_mean=y_mean,
                        y_std=y_std,
                        task=str(args.task),
                        region=str(args.region),
                        method="global_head",
                        head_scope=str(args.head_scope_label or "macroregion"),
                        head_group=str(args.region or "selected_samples"),
                        head_group_name=str(args.region or "selected_samples"),
                        model_name=model_name,
                        seed=int(seed),
                    )
                    write_predictions(
                        pred_dir / "macroregion_heads" / f"{args.region}_{args.task}__global_head__{model_name}__seed{seed}.parquet",
                        [frame],
                    )
                append_csv(metrics_csv, [{"method": "global_head", "model": model_name, "seed": seed, **metrics}])
                done.add(key)

            key = ("group_heads_country_macro", model_name, int(seed))
            if model_name in skip_group_models or key in done:
                continue
            local_preds = []
            local_tgts = []
            region_rows = []
            skipped_runtime_rows = []
            prediction_frames = []
            print(f"[TRAIN] method=group_heads_country_macro model={model_name} seed={seed} groups={len(groups)}", flush=True)
            for g in groups:
                print(
                    f"[GROUP] model={model_name} seed={seed} group={g['group']} n={int(g['n']):,}",
                    flush=True,
                )
                if args.split_mode == "spatial_block":
                    g_blocks = union_block_ids[int(g["offset"]): int(g["offset"]) + int(g["n"])]
                    idx_tr, idx_va, idx_te = make_block_split(g_blocks, int(seed))
                else:
                    idx_tr, idx_va, idx_te = make_split(int(g["n"]), int(seed))
                if len(idx_tr) == 0 or len(idx_te) == 0:
                    skipped_runtime_rows.append(
                        {
                            "method": "group_head",
                            "model": model_name,
                            "seed": seed,
                            "group": g["group"],
                            "group_name": g["group_name"],
                            "n_source": g["n_source"],
                            "n": g["n"],
                            "n_train": int(len(idx_tr)),
                            "n_val": int(len(idx_va)),
                            "n_test": int(len(idx_te)),
                            "reason": "empty train or test split",
                        }
                    )
                    continue
                off = int(g["offset"])
                sl = slice(off, off + int(g["n"]))
                raw_group = raw_union[sl]
                y_mean, y_std = training_target_stats(raw_group, idx_tr)
                Y_group = normalize_target(raw_group, y_mean, y_std)
                if X is None:
                    metrics_norm_g, pred_g, tgt_g = fit_predict_indexed(
                        valid_models[model_name],
                        Y_group,
                        union_idx[sl],
                        idx_tr,
                        idx_va,
                        idx_te,
                        seed=int(seed),
                        device=device,
                        epochs=int(args.epochs),
                        patience=int(args.patience),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        hidden=int(args.hidden),
                        head_type=str(args.head_type),
                    )
                else:
                    metrics_norm_g, pred_g, tgt_g = fit_predict(
                        X[sl],
                        Y_group,
                        idx_tr,
                        idx_va,
                        idx_te,
                        seed=int(seed),
                        device=device,
                        epochs=int(args.epochs),
                        patience=int(args.patience),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        hidden=int(args.hidden),
                        head_type=str(args.head_type),
                    )
                metrics_g, pred_raw_g, tgt_raw_g = metrics_in_raw_space(
                    metrics_norm_g, pred_g, tgt_g, y_mean, y_std
                )
                local_preds.append(pred_raw_g)
                local_tgts.append(tgt_raw_g)
                if args.save_predictions:
                    prediction_frames.append(
                        build_prediction_frame(
                            samples=samples,
                            labels_df=labels_df,
                            union_idx=union_idx,
                            union_positions=off + idx_te,
                            pred=pred_g,
                            tgt=tgt_g,
                            y_mean=y_mean,
                            y_std=y_std,
                            task=str(args.task),
                            region=str(args.region),
                            method="group_head",
                            head_scope="country",
                            head_group=str(g["group"]),
                            head_group_name=str(g["group_name"]),
                            model_name=model_name,
                            seed=int(seed),
                        )
                    )
                region_rows.append(
                    {
                        "method": "group_head",
                        "model": model_name,
                        "seed": seed,
                        "group": g["group"],
                        "group_name": g["group_name"],
                        "n_source": g["n_source"],
                        "n": g["n"],
                        **metrics_g,
                    }
                )
            append_csv(out_dir / "runtime_skipped_groups.csv", skipped_runtime_rows)
            if not local_preds:
                print(f"[WARN] no valid group-head predictions for model={model_name} seed={seed}", flush=True)
                continue
            if args.save_predictions:
                write_predictions(
                    pred_dir / "country_heads" / f"{args.region}_{args.task}__group_heads__{model_name}__seed{seed}.parquet",
                    prediction_frames,
                )
            macro_metrics = {
                name: float(np.nanmean([r[name] for r in region_rows]))
                for name in ["MAE", "RMSE", "R2", "MSE", "MAE_norm", "RMSE_norm", "R2_norm", "MSE_norm"]
            }
            macro_metrics.update(
                {
                    "best_val": float(np.mean([r["best_val"] for r in region_rows])),
                    "best_epoch": float(np.mean([r["best_epoch"] for r in region_rows])),
                    "actual_epochs": float(np.mean([r["actual_epochs"] for r in region_rows])),
                    "n_train": int(sum(r["n_train"] for r in region_rows)),
                    "n_val": int(sum(r["n_val"] for r in region_rows)),
                    "n_test": int(sum(r["n_test"] for r in region_rows)),
                    "train_steps": int(sum(r["train_steps"] for r in region_rows)),
                    "n_groups": int(len(region_rows)),
                }
            )
            pooled_metrics = eval_metrics(
                torch.cat(local_preds, dim=0), torch.cat(local_tgts, dim=0)
            )
            pooled_metrics.update(
                {
                    "method": "group_heads_pooled_raw_diagnostic",
                    "model": model_name,
                    "seed": seed,
                    "n_groups": int(len(region_rows)),
                    "n_test": int(sum(r["n_test"] for r in region_rows)),
                }
            )
            append_csv(region_csv, region_rows)
            append_csv(out_dir / "pooled_metrics_diagnostic.csv", [pooled_metrics])
            append_csv(
                metrics_csv,
                [{"method": "group_heads_country_macro", "model": model_name, "seed": seed, **macro_metrics}],
            )
            done.add(key)
        del X
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summarize_metrics(metrics_csv, out_dir, str(args.aether_model))
    print(f"[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
