#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.features import geometry_mask
from rasterio.transform import xy as tf_xy
from rasterio.windows import Window, from_bounds, transform as window_transform
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import AEProj, AEProjSH  # noqa: E402

DEFAULT_BBOX = (115.4262, 39.4479, 117.5046, 41.0589)
_AETHER_HEAD_CACHE = {}

class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))
    def __len__(self): return int(self.x.shape[0])
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

def l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)).astype(np.float32)

def decode_ae_band(arr: np.ndarray, nodata: float | int | None) -> tuple[np.ndarray, np.ndarray]:
    if arr.dtype == np.int8:
        nodata_value = -128 if nodata is None else int(nodata)
        valid = arr != nodata_value
        vals = arr.astype(np.float32)
        out = np.zeros(vals.shape, dtype=np.float32)
        vv = vals[valid]
        out[valid] = np.sign(vv) * np.square(vv / 127.5)
        return out, valid

    vals = arr.astype(np.float32)
    valid = np.isfinite(vals)
    if nodata is not None:
        valid &= vals != float(nodata)
    return np.where(valid, vals, 0.0).astype(np.float32), valid

def decode_ae_stack(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    if arr.dtype == np.int8:
        nodata_value = -128 if nodata is None else int(nodata)
        valid = arr != nodata_value
        vals = arr.astype(np.float32)
        out = np.zeros(vals.shape, dtype=np.float32)
        vv = vals[valid]
        out[valid] = np.sign(vv) * np.square(vv / 127.5)
        return out

    vals = arr.astype(np.float32)
    valid = np.isfinite(vals)
    if nodata is not None:
        valid &= vals != float(nodata)
    return np.where(valid, vals, 0.0).astype(np.float32)

def parse_bbox(s: str | None) -> Tuple[float, float, float, float]:
    if not s: return DEFAULT_BBOX
    if s.lower() in {"raster", "tile-index", "tile_index", "ae"}:
        raise ValueError(f"--bbox={s} must be resolved after opening raster/tile-index")
    vals = [float(v.strip()) for v in s.split(',')]
    if len(vals) != 4: raise ValueError('--bbox must be min_lon,min_lat,max_lon,max_lat')
    return vals[0], vals[1], vals[2], vals[3]

def resolve_bbox_arg(bbox_arg: str | None, raster_path: Path, tile_index_path: Path) -> Tuple[float, float, float, float]:
    if not bbox_arg:
        return DEFAULT_BBOX
    key = bbox_arg.lower()
    if key == "raster":
        with rasterio.open(raster_path) as src:
            b = src.bounds
            return float(b.left), float(b.bottom), float(b.right), float(b.top)
    if key in {"tile-index", "tile_index", "ae"}:
        tiles = pd.read_parquet(tile_index_path)
        return (
            float(tiles.left.min()),
            float(tiles.bottom.min()),
            float(tiles.right.max()),
            float(tiles.top.max()),
        )
    return parse_bbox(bbox_arg)

def _intersect_windows(a: Window, b: Window) -> Window | None:
    c0 = max(int(math.floor(a.col_off)), int(math.floor(b.col_off)))
    r0 = max(int(math.floor(a.row_off)), int(math.floor(b.row_off)))
    c1 = min(int(math.ceil(a.col_off + a.width)), int(math.ceil(b.col_off + b.width)))
    r1 = min(int(math.ceil(a.row_off + a.height)), int(math.ceil(b.row_off + b.height)))
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)

def _sample_raster_blockwise(src, band: int, window: Window, max_samples: int, seed: int, valid_min: float | None):
    rng = np.random.default_rng(seed)
    keep_keys = np.empty((0,), dtype=np.float64)
    keep_rows = np.empty((0,), dtype=np.int64)
    keep_cols = np.empty((0,), dtype=np.int64)
    keep_vals = np.empty((0,), dtype=np.float32)
    nodata = src.nodata
    block_windows = src.block_windows(int(band))
    for _, block_window in block_windows:
        clipped = _intersect_windows(block_window, window)
        if clipped is None:
            continue
        arr = src.read(int(band), window=clipped, boundless=False)
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= arr != nodata
        if valid_min is not None:
            mask &= arr >= float(valid_min)
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        vals = arr[rows, cols].astype(np.float32)
        keys = rng.random(len(vals), dtype=np.float64)
        rows = rows.astype(np.int64) + int(clipped.row_off)
        cols = cols.astype(np.int64) + int(clipped.col_off)

        keep_keys = np.concatenate([keep_keys, keys])
        keep_rows = np.concatenate([keep_rows, rows])
        keep_cols = np.concatenate([keep_cols, cols])
        keep_vals = np.concatenate([keep_vals, vals])
        if len(keep_keys) > max_samples * 2:
            idx = np.argpartition(keep_keys, max_samples - 1)[:max_samples]
            keep_keys, keep_rows, keep_cols, keep_vals = keep_keys[idx], keep_rows[idx], keep_cols[idx], keep_vals[idx]

    if len(keep_keys) > max_samples:
        idx = np.argpartition(keep_keys, max_samples - 1)[:max_samples]
        keep_rows, keep_cols, keep_vals = keep_rows[idx], keep_cols[idx], keep_vals[idx]
    return keep_rows, keep_cols, keep_vals

def transform_target_values(vals: np.ndarray, target_transform: str) -> np.ndarray:
    vals = vals.astype(np.float32, copy=False)
    if target_transform == 'log1p':
        if np.nanmin(vals) < -1.0:
            raise ValueError('target_transform=log1p requires values >= -1; use --target-transform identity for signed targets.')
        return np.log1p(vals)
    if target_transform == 'identity':
        return vals
    raise ValueError(f'Unknown target_transform: {target_transform}')


def sample_raster(raster_path: Path, bbox, max_samples: int, seed: int, out_path: Path, band: int, force: bool, valid_min: float | None = 0.0, target_transform: str = 'identity'):
    if out_path.exists() and not force:
        data = np.load(out_path)
        return {k: data[k] for k in data.files}
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bbox
    with rasterio.open(raster_path) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
        full_window = Window(0, 0, src.width, src.height)
        window = _intersect_windows(window, full_window)
        if window is None:
            raise ValueError(f"bbox {bbox} does not overlap raster bounds {src.bounds}")
        est_cells = int(window.width * window.height)
        if est_cells <= 100_000_000:
            arr = src.read(int(band), window=window, boundless=False)
            nodata = src.nodata
            mask = np.isfinite(arr)
            if nodata is not None: mask &= arr != nodata
            if valid_min is not None:
                mask &= arr >= float(valid_min)
            rows, cols = np.where(mask)
            vals = arr[rows, cols].astype(np.float32)
            if len(vals) > max_samples:
                idx = rng.choice(len(vals), size=max_samples, replace=False)
                rows, cols, vals = rows[idx], cols[idx], vals[idx]
            rows = rows.astype(np.int64) + int(window.row_off)
            cols = cols.astype(np.int64) + int(window.col_off)
        else:
            print(f"[sample] large window cells={est_cells:,}; using blockwise random sampling", flush=True)
            rows, cols, vals = _sample_raster_blockwise(src, int(band), window, max_samples, seed, valid_min)
        xs, ys = tf_xy(src.transform, rows, cols, offset='center')
        lon = np.asarray(xs, dtype=np.float32)
        lat = np.asarray(ys, dtype=np.float32)
    vals_target = transform_target_values(vals, target_transform)
    y_mean = float(vals_target.mean())
    y_std = float(vals_target.std() + 1e-6)
    y_norm = ((vals_target - y_mean) / y_std).reshape(-1, 1).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, lon=lon, lat=lat, y=y_norm, raw=vals, bbox=np.asarray(bbox), y_mean=np.asarray([y_mean], dtype=np.float32), y_std=np.asarray([y_std], dtype=np.float32), target_transform=np.asarray([target_transform]))
    return {'lon': lon, 'lat': lat, 'y': y_norm, 'raw': vals, 'bbox': np.asarray(bbox), 'y_mean': np.asarray([y_mean], dtype=np.float32), 'y_std': np.asarray([y_std], dtype=np.float32), 'target_transform': np.asarray([target_transform])}

def _append_reservoir(
    rng: np.random.Generator,
    keep_keys: np.ndarray,
    keep_rows: np.ndarray,
    keep_cols: np.ndarray,
    keep_vals: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    max_samples: int,
):
    keys = rng.random(len(vals), dtype=np.float64)
    keep_keys = np.concatenate([keep_keys, keys])
    keep_rows = np.concatenate([keep_rows, rows])
    keep_cols = np.concatenate([keep_cols, cols])
    keep_vals = np.concatenate([keep_vals, vals])
    if len(keep_keys) > max_samples * 2:
        idx = np.argpartition(keep_keys, max_samples - 1)[:max_samples]
        keep_keys, keep_rows, keep_cols, keep_vals = keep_keys[idx], keep_rows[idx], keep_cols[idx], keep_vals[idx]
    return keep_keys, keep_rows, keep_cols, keep_vals

def load_boundary_geometries(boundary_path: Path, layer: str | None, dst_crs) -> list:
    import geopandas as gpd

    kwargs = {}
    if layer:
        kwargs["layer"] = layer
    gdf = gpd.read_file(boundary_path, **kwargs)
    if gdf.empty:
        raise ValueError(f"Boundary vector has no rows: {boundary_path}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if dst_crs is not None:
        gdf = gdf.to_crs(dst_crs)
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError(f"Boundary vector has no usable geometries: {boundary_path}")
    return geoms

def sample_raster_from_tile_index(
    raster_path: Path,
    tile_index_path: Path,
    max_samples: int,
    seed: int,
    out_path: Path,
    band: int,
    force: bool,
    valid_min: float | None = 0.0,
    target_transform: str = 'identity',
    boundary_path: Path | None = None,
    boundary_layer: str | None = None,
):
    if out_path.exists() and not force:
        data = np.load(out_path)
        return {k: data[k] for k in data.files}
    rng = np.random.default_rng(seed)
    tiles = pd.read_parquet(tile_index_path)
    bbox = (
        float(tiles.left.min()),
        float(tiles.bottom.min()),
        float(tiles.right.max()),
        float(tiles.top.max()),
    )
    keep_keys = np.empty((0,), dtype=np.float64)
    keep_rows = np.empty((0,), dtype=np.int64)
    keep_cols = np.empty((0,), dtype=np.int64)
    keep_vals = np.empty((0,), dtype=np.float32)
    with rasterio.open(raster_path) as src:
        boundary_geoms = load_boundary_geometries(boundary_path, boundary_layer, src.crs) if boundary_path else None
        full_window = Window(0, 0, src.width, src.height)
        for rec in tqdm(tiles.itertuples(index=False), total=len(tiles), desc='sample tile bboxes', ncols=90):
            window = from_bounds(float(rec.left), float(rec.bottom), float(rec.right), float(rec.top), src.transform).round_offsets().round_lengths()
            window = _intersect_windows(window, full_window)
            if window is None:
                continue
            arr = src.read(int(band), window=window, boundless=False)
            nodata = src.nodata
            mask = np.isfinite(arr)
            if nodata is not None:
                mask &= arr != nodata
            if valid_min is not None:
                mask &= arr >= float(valid_min)
            if boundary_geoms is not None:
                inside = geometry_mask(
                    boundary_geoms,
                    out_shape=arr.shape,
                    transform=window_transform(window, src.transform),
                    invert=True,
                    all_touched=False,
                )
                mask &= inside
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            vals = arr[rows, cols].astype(np.float32)
            rows = rows.astype(np.int64) + int(window.row_off)
            cols = cols.astype(np.int64) + int(window.col_off)
            keep_keys, keep_rows, keep_cols, keep_vals = _append_reservoir(
                rng, keep_keys, keep_rows, keep_cols, keep_vals, rows, cols, vals, max_samples
            )
        if len(keep_keys) > max_samples:
            idx = np.argpartition(keep_keys, max_samples - 1)[:max_samples]
            keep_rows, keep_cols, keep_vals = keep_rows[idx], keep_cols[idx], keep_vals[idx]
        if len(keep_vals) == 0:
            raise ValueError(f"No valid raster samples found in tile index bounds: {tile_index_path}")
        xs, ys = tf_xy(src.transform, keep_rows, keep_cols, offset='center')
        lon = np.asarray(xs, dtype=np.float32)
        lat = np.asarray(ys, dtype=np.float32)
    vals_target = transform_target_values(keep_vals, target_transform)
    y_mean = float(vals_target.mean())
    y_std = float(vals_target.std() + 1e-6)
    y_norm = ((vals_target - y_mean) / y_std).reshape(-1, 1).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, lon=lon, lat=lat, y=y_norm, raw=keep_vals, bbox=np.asarray(bbox), y_mean=np.asarray([y_mean], dtype=np.float32), y_std=np.asarray([y_std], dtype=np.float32), target_transform=np.asarray([target_transform]), sample_from_tile_index=np.asarray([1], dtype=np.int8))
    return {'lon': lon, 'lat': lat, 'y': y_norm, 'raw': keep_vals, 'bbox': np.asarray(bbox), 'y_mean': np.asarray([y_mean], dtype=np.float32), 'y_std': np.asarray([y_std], dtype=np.float32), 'target_transform': np.asarray([target_transform]), 'sample_from_tile_index': np.asarray([1], dtype=np.int8)}

def integral_image(arr: np.ndarray, dtype) -> np.ndarray:
    ii = np.cumsum(np.cumsum(arr, axis=0, dtype=dtype), axis=1, dtype=dtype)
    return np.pad(ii, ((1, 0), (1, 0)), mode='constant')

def rect_sum(ii, r0, r1, c0, c1):
    return ii[r1 + 1, c1 + 1] - ii[r0, c1 + 1] - ii[r1 + 1, c0] + ii[r0, c0]

def _infer_aether_head(sd: dict, model_types: dict | None, device: torch.device) -> nn.Module:
    ae_type = str((model_types or {}).get("ae_type", ""))
    if ae_type == "AEProjSH" or "pos_proj.0.weight" in sd:
        out_dim = int(sd["out_proj.weight"].shape[0]); hidden = int(sd["out_proj.weight"].shape[1]); pe_dim = int(sd["pos_proj.0.weight"].shape[1]); sh_degree = int(round(math.sqrt(pe_dim) - 1)); pos_hidden = int(sd["pos_proj.0.weight"].shape[0])
        return AEProjSH(d_in=64, d_out=out_dim, hidden=hidden, sh_degree=sh_degree, pos_hidden=pos_hidden).to(device)
    if ae_type not in {"", "AEProj"}:
        raise ValueError(f"Unsupported checkpoint architecture: {ae_type}")
    weight = sd.get("post_mlp.3.weight", torch.empty(128, 256)); return AEProj(d_in=64, d_out=int(weight.shape[0]), hidden=int(weight.shape[1])).to(device)



def load_aether_head(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["ae_state_dict"]
    head = _infer_aether_head(sd, ckpt.get("model_types"), device)
    head.load_state_dict(sd, strict=True)
    head.eval()
    return head

def _get_cached_head(ckpt_path: str, device_str: str) -> nn.Module:
    key = (ckpt_path, device_str)
    head = _AETHER_HEAD_CACHE.get(key)
    if head is None:
        if device_str == "cpu":
            torch.set_num_threads(1)
        head = load_aether_head(Path(ckpt_path), torch.device(device_str))
        _AETHER_HEAD_CACHE[key] = head
    return head


def _aether_out_dim(head: nn.Module, device: torch.device) -> int:
    if isinstance(head, AEProjSH):
        return int(head.out_proj.out_features)
    for layer in reversed(list(head.post_mlp)):
        if isinstance(layer, nn.Linear):
            return int(layer.out_features)
    raise TypeError(f"Unsupported AETHER head: {type(head).__name__}")



def _head_forward(head: nn.Module, xb: torch.Tensor, latlon_np: np.ndarray | None = None) -> torch.Tensor:
    if isinstance(head, AEProjSH):
        if latlon_np is None:
            raise ValueError("AEProjSH inference requires per-pixel coordinates")
        latlon = torch.from_numpy(latlon_np.astype(np.float32, copy=False)).to(xb.device, non_blocking=True)
        return head(xb, latlon=latlon)
    return head(xb)



def _pixel_center_xy(transform, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rr = rows.astype(np.float64, copy=False) + 0.5
    cc = cols.astype(np.float64, copy=False) + 0.5
    x = transform.c + cc * transform.a + rr * transform.b
    y = transform.f + cc * transform.d + rr * transform.e
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)

def _regular_grid_tile_subtasks(x: np.ndarray, y: np.ndarray, tiles: pd.DataFrame):
    widths = (tiles["right"].to_numpy(np.float64) - tiles["left"].to_numpy(np.float64))
    heights = (tiles["top"].to_numpy(np.float64) - tiles["bottom"].to_numpy(np.float64))
    if len(widths) == 0:
        return None
    width = float(widths[0])
    height = float(heights[0])
    if width <= 0 or height <= 0:
        return None
    if not (np.allclose(widths, width) and np.allclose(heights, height)):
        return None

    left0 = float(tiles["left"].min())
    bottom0 = float(tiles["bottom"].min())
    tile_gx = np.rint((tiles["left"].to_numpy(np.float64) - left0) / width).astype(np.int64)
    tile_gy = np.rint((tiles["bottom"].to_numpy(np.float64) - bottom0) / height).astype(np.int64)
    if tile_gx.size == 0 or tile_gy.size == 0:
        return None
    ncols = int(tile_gx.max()) + 1
    tile_keys = tile_gy * ncols + tile_gx
    if len(np.unique(tile_keys)) != len(tile_keys):
        return None

    paths = dict(zip(tile_keys.tolist(), tiles["path"].tolist()))
    gx = np.floor((x.astype(np.float64, copy=False) - left0) / width).astype(np.int64)
    gy = np.floor((y.astype(np.float64, copy=False) - bottom0) / height).astype(np.int64)
    keys = gy * ncols + gx
    valid = np.fromiter((int(k) in paths for k in keys), dtype=bool, count=len(keys))
    if not valid.any():
        return []

    valid_idx = np.nonzero(valid)[0]
    valid_keys = keys[valid]
    order = np.argsort(valid_keys, kind="stable")
    sorted_keys = valid_keys[order]
    starts = np.r_[0, np.nonzero(sorted_keys[1:] != sorted_keys[:-1])[0] + 1]
    stops = np.r_[starts[1:], len(sorted_keys)]
    tasks = []
    for start, stop in zip(starts, stops):
        key = int(sorted_keys[start])
        idx = valid_idx[order[start:stop]].astype(np.int64, copy=False)
        tasks.append((paths[key], idx, x[idx].astype(np.float64, copy=False), y[idx].astype(np.float64, copy=False)))
    return tasks

def _tile_subtasks(x: np.ndarray, y: np.ndarray, tiles: pd.DataFrame):
    fast_tasks = _regular_grid_tile_subtasks(x, y, tiles)
    if fast_tasks is not None:
        for task in fast_tasks:
            yield task
        return

    for rec in tiles.itertuples(index=False):
        mask = (x >= float(rec.left)) & (x < float(rec.right)) & (y >= float(rec.bottom)) & (y < float(rec.top))
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        yield rec.path, idx.astype(np.int64), x[idx].astype(np.float64, copy=False), y[idx].astype(np.float64, copy=False)

def _row_col_from_xy(src, x_local: np.ndarray, y_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = ~src.transform
    cols_f, rows_f = inv * (x_local, y_local)
    rows = np.floor(rows_f).astype(np.int32, copy=False)
    cols = np.floor(cols_f).astype(np.int32, copy=False)
    return rows, cols

def _ae_tile_worker(task):
    tile_path, idx, x_local, y_local, radius_px = task
    with rasterio.open(str(tile_path)) as src:
        rows, cols = _row_col_from_xy(src, x_local, y_local)
        r0 = np.clip(rows - radius_px, 0, src.height - 1)
        r1 = np.clip(rows + radius_px, 0, src.height - 1)
        c0 = np.clip(cols - radius_px, 0, src.width - 1)
        c1 = np.clip(cols + radius_px, 0, src.width - 1)
        out_local = np.zeros((len(idx), 64), dtype=np.float32)
        for band in range(1, 65):
            vals, finite = decode_ae_band(src.read(band), src.nodata)
            sum_ii = integral_image(vals, np.float64)
            cnt_ii = integral_image(finite.astype(np.int32), np.int64)
            s = rect_sum(sum_ii, r0, r1, c0, c1)
            c = rect_sum(cnt_ii, r0, r1, c0, c1)
            out_local[:, band - 1] = np.divide(s, np.clip(c, 1, None), out=np.zeros_like(s, dtype=np.float64), where=c > 0)
    return idx, out_local

def _aether_tile_worker(task):
    if len(task) == 8:
        tile_path, idx, x_local, y_local, radius_px, ckpt_path, batch_pixels, worker_device = task
        latlon_override = None
    else:
        tile_path, idx, x_local, y_local, radius_px, ckpt_path, batch_pixels, worker_device, latlon_override = task
    head = _get_cached_head(ckpt_path, worker_device)
    device = next(head.parameters()).device
    out_dim = _aether_out_dim(head, device)
    out_local = np.zeros((len(idx), out_dim), dtype=np.float32)
    with rasterio.open(str(tile_path)) as src:
        rows, cols = _row_col_from_xy(src, x_local, y_local)
        tile_arr = decode_ae_stack(src.read(indexes=list(range(1, 65))), src.nodata)
        flat = tile_arr.reshape(64, -1)
        offsets = None
        if radius_px > 0:
            rr = np.arange(-radius_px, radius_px + 1, dtype=np.int32)
            cc = np.arange(-radius_px, radius_px + 1, dtype=np.int32)
            dr, dc = np.meshgrid(rr, cc, indexing="ij")
            offsets = np.stack([dr.ravel(), dc.ravel()], axis=1)
        sample_batch = 4096 if radius_px > 0 else 65536
        with torch.inference_mode():
            for start in range(0, len(idx), sample_batch):
                end = min(start + sample_batch, len(idx))
                rb = rows[start:end]
                cb = cols[start:end]
                if radius_px == 0:
                    rr_clip = np.clip(rb, 0, src.height - 1).astype(np.int64)
                    cc_clip = np.clip(cb, 0, src.width - 1).astype(np.int64)
                    lin = rr_clip * src.width + cc_clip
                    pix = flat[:, lin].T
                    pix = l2_normalize(pix)
                    if latlon_override is None:
                        pix_lon, pix_lat = _pixel_center_xy(src.transform, rr_clip, cc_clip)
                        pix_latlon = np.stack([pix_lat, pix_lon], axis=1).astype(np.float32, copy=False)
                    else:
                        pix_latlon = latlon_override[start:end].astype(np.float32, copy=False)
                    zs = []
                    for j in range(0, len(pix), batch_pixels):
                        xb = torch.from_numpy(pix[j:j + batch_pixels]).to(device).float()
                        zs.append(_head_forward(head, xb, pix_latlon[j:j + batch_pixels]).detach().cpu().numpy())
                    out_local[start:end] = np.vstack(zs).astype(np.float32)
                    continue

                if latlon_override is None:
                    wr = np.clip(rb[:, None] + offsets[None, :, 0], 0, src.height - 1).astype(np.int64, copy=False)
                    wc = np.clip(cb[:, None] + offsets[None, :, 1], 0, src.width - 1).astype(np.int64, copy=False)
                    lin = (wr * src.width + wc).ravel()
                    unique_lin, inverse = np.unique(lin, return_inverse=True)
                    pix = flat[:, unique_lin].T
                    pix = l2_normalize(pix)
                    pix_rows = (unique_lin // src.width).astype(np.int64, copy=False)
                    pix_cols = (unique_lin % src.width).astype(np.int64, copy=False)
                    pix_lon, pix_lat = _pixel_center_xy(src.transform, pix_rows, pix_cols)
                    pix_latlon = np.stack([pix_lat, pix_lon], axis=1).astype(np.float32, copy=False)
                    zs = []
                    for j in range(0, len(pix), batch_pixels):
                        xb = torch.from_numpy(pix[j:j + batch_pixels]).to(device).float()
                        zs.append(_head_forward(head, xb, pix_latlon[j:j + batch_pixels]).detach().cpu().numpy())
                    z_unique = np.vstack(zs).astype(np.float32, copy=False)
                    z = z_unique[inverse].reshape(end - start, -1, out_dim).mean(axis=1)
                else:
                    wr = np.clip(rb[:, None] + offsets[None, :, 0], 0, src.height - 1).astype(np.int64, copy=False)
                    wc = np.clip(cb[:, None] + offsets[None, :, 1], 0, src.width - 1).astype(np.int64, copy=False)
                    lin = (wr * src.width + wc).ravel()
                    pix = flat[:, lin].T
                    pix = l2_normalize(pix)
                    pix_latlon = np.repeat(
                        latlon_override[start:end].astype(np.float32, copy=False),
                        offsets.shape[0],
                        axis=0,
                    )
                    zs = []
                    for j in range(0, len(pix), batch_pixels):
                        xb = torch.from_numpy(pix[j:j + batch_pixels]).to(device).float()
                        zs.append(_head_forward(head, xb, pix_latlon[j:j + batch_pixels]).detach().cpu().numpy())
                    z = np.vstack(zs).reshape(end - start, -1, out_dim).mean(axis=1)
                out_local[start:end] = z.astype(np.float32)
        del tile_arr
    return idx, out_local

def _aether_part_path(parts_dir: Path, task_id: int) -> Path:
    return parts_dir / f"part_{task_id:05d}.npz"

def _save_aether_part(part_path: Path, idx: np.ndarray, out_local: np.ndarray) -> None:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = part_path.with_name(f"{part_path.name}.tmp.{os.getpid()}.npz")
    np.savez(tmp, idx=idx.astype(np.int64, copy=False), out=out_local.astype(np.float32, copy=False))
    os.replace(tmp, part_path)

def _load_aether_part(part_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(part_path)
    return data["idx"].astype(np.int64, copy=False), data["out"].astype(np.float32, copy=False)

def window_mean_ae(lon, lat, tile_index_path: Path, radius_px: int, out_path: Path, force: bool, jobs: int = 1):
    if out_path.exists() and not force: return np.load(out_path)
    tiles = pd.read_parquet(tile_index_path)
    x = lon.astype(np.float64); y = lat.astype(np.float64)
    out = np.zeros((len(x), 64), dtype=np.float32)
    assigned = np.zeros(len(x), dtype=bool)
    tasks = [(tile_path, idx, x_local, y_local, radius_px) for tile_path, idx, x_local, y_local in _tile_subtasks(x, y, tiles)]
    for _, idx, _, _, _ in tasks:
        assigned[idx] = True
    if jobs <= 1:
        iterator = map(_ae_tile_worker, tasks)
        iterable = tqdm(iterator, total=len(tasks), desc=f'AE window r{radius_px}', ncols=90)
        for idx, out_local in iterable:
            out[idx] = out_local
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_ae_tile_worker, task) for task in tasks]
            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc=f'AE window r{radius_px}', ncols=90):
                idx, out_local = fut.result()
                out[idx] = out_local
    if not assigned.all():
        print(f'[WARN] unassigned samples: {int((~assigned).sum())}')
    out = l2_normalize(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    return out

def encode_aether(ae, ckpt_path: Path, device: torch.device, out_path: Path, force: bool):
    """Project an AE cache without location conditioning; only valid for non-SH AETHER heads."""
    if out_path.exists() and not force: return np.load(out_path)
    head = load_aether_head(ckpt_path, device)
    if isinstance(head, AEProjSH) or hasattr(head, "pos_proj"):
        raise ValueError("encode_aether cannot project AEProjSH without lat/lon. Use window_mean_aether_exact.")
    outs=[]; bs=8192
    with torch.no_grad():
        for i in tqdm(range(0, len(ae), bs), desc='AETHER encode', ncols=90):
            xb=torch.from_numpy(ae[i:i+bs]).to(device).float()
            outs.append(head(xb).cpu().numpy())
    z=np.vstack(outs).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, z)
    return z

def window_mean_aether_exact(lon, lat, tile_index_path: Path, radius_px: int, ckpt_path: Path, device: torch.device, out_path: Path, force: bool, jobs: int = 1):
    """Mean-pool projected AETHER pixels in each window, matching raster-inference then aggregation."""
    if out_path.exists() and not force: return np.load(out_path)
    parts_dir = out_path.with_name(out_path.name + ".parts")
    if force and parts_dir.exists():
        shutil.rmtree(parts_dir)
    tiles = pd.read_parquet(tile_index_path)
    x = lon.astype(np.float64); y = lat.astype(np.float64)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["ae_state_dict"]
    if "out_proj.weight" in sd:
        out_dim = int(sd["out_proj.weight"].shape[0])
    elif "post_mlp.3.weight" in sd:
        out_dim = int(sd["post_mlp.3.weight"].shape[0])
    else:
        out_dim = 128
    del ckpt
    out = np.zeros((len(x), out_dim), dtype=np.float32)
    assigned = np.zeros(len(x), dtype=bool)
    batch_pixels = int(os.environ.get("AETHER_BATCH_PIXELS", "32768"))
    worker_device = str(device) if torch.cuda.is_available() and device.type == "cuda" else "cpu"
    print(f"[AETHER] worker_device={worker_device} jobs={jobs} batch_pixels={batch_pixels}", flush=True)

    tasks = []
    for tile_path, idx, x_local, y_local in _tile_subtasks(x, y, tiles):
        tasks.append((tile_path, idx, x_local, y_local, radius_px, str(ckpt_path), batch_pixels, worker_device, None))
    for task in tasks:
        idx = task[1]
        assigned[idx] = True
    pending: list[tuple[int, tuple]] = []
    resumed = 0
    for task_id, task in enumerate(tasks):
        part_path = _aether_part_path(parts_dir, task_id)
        if part_path.exists() and not force:
            idx_done, out_done = _load_aether_part(part_path)
            out[idx_done] = out_done
            resumed += 1
        else:
            pending.append((task_id, task))
    if resumed:
        print(f'[resume] AETHER loaded completed tile parts: {resumed}/{len(tasks)}', flush=True)
    if jobs <= 1:
        iterator = ((task_id, _aether_tile_worker(task)) for task_id, task in pending)
        iterable = tqdm(iterator, total=len(pending), desc=f'AETHER window r{radius_px}', ncols=90)
        for task_id, (idx, out_local) in iterable:
            _save_aether_part(_aether_part_path(parts_dir, task_id), idx, out_local)
            out[idx] = out_local
    else:
        mp_context = mp.get_context("spawn") if worker_device != "cpu" else None
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp_context) as ex:
            futs = {ex.submit(_aether_tile_worker, task): task_id for task_id, task in pending}
            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc=f'AETHER window r{radius_px}', ncols=90):
                task_id = futs[fut]
                idx, out_local = fut.result()
                _save_aether_part(_aether_part_path(parts_dir, task_id), idx, out_local)
                out[idx] = out_local
    if not assigned.all():
        print(f'[WARN] unassigned samples: {int((~assigned).sum())}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp_out, out)
    os.replace(tmp_out, out_path)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return out


def sample_mask_values(lon: np.ndarray, lat: np.ndarray, mask_paths: list[Path]) -> np.ndarray:
    vals = np.full(len(lon), -9999, dtype=np.int32)
    x = lon.astype(np.float64); y = lat.astype(np.float64)
    for mp in mask_paths:
        with rasterio.open(mp) as ds:
            left, bottom, right, top = ds.bounds
            mask = (x >= left) & (x < right) & (y >= bottom) & (y < top)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            coords = list(zip(x[idx].tolist(), y[idx].tolist()))
            arr = np.asarray([v[0] for v in ds.sample(coords)], dtype=np.int32)
            vals[idx] = arr
    return vals

def eval_metrics(pred, target):
    pred_np=pred.numpy().reshape(-1); target_np=target.numpy().reshape(-1)
    mse=float(np.mean((pred_np-target_np)**2))
    return {'MAE': float(np.mean(np.abs(pred_np-target_np))), 'RMSE': float(math.sqrt(mse)), 'R2': float(r2_score(target_np, pred_np)), 'MSE': mse}

def train_one(X, Y, seed, device, out_dir, epochs, patience, batch_size, lr, hidden):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx=np.arange(len(X)); idx_tr, idx_te=train_test_split(idx, test_size=0.2, random_state=seed); idx_tr, idx_va=train_test_split(idx_tr, test_size=0.1, random_state=seed)
    dl_tr=DataLoader(ArrayDataset(X[idx_tr], Y[idx_tr]), batch_size=batch_size, shuffle=True)
    dl_va=DataLoader(ArrayDataset(X[idx_va], Y[idx_va]), batch_size=batch_size, shuffle=False)
    dl_te=DataLoader(ArrayDataset(X[idx_te], Y[idx_te]), batch_size=batch_size, shuffle=False)
    model=MLPHead(X.shape[1], hidden=hidden).to(device); opt=torch.optim.Adam(model.parameters(), lr=lr); loss_fn=nn.MSELoss(); best_val=float('inf'); best_state=None; wait=0
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in dl_tr:
            xb,yb=xb.to(device).float(),yb.to(device).float(); loss=loss_fn(model(xb),yb); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval(); val=0.0
        with torch.no_grad():
            for xb,yb in dl_va:
                xb,yb=xb.to(device).float(),yb.to(device).float(); val += float(loss_fn(model(xb),yb).item())
        val /= max(1, len(dl_va))
        if ep == 1 or ep % 10 == 0: print(f'seed={seed} epoch={ep:03d} val={val:.4f}')
        if val + 1e-8 < best_val:
            best_val=val; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
            if wait >= patience:
                print(f'seed={seed} early stop at epoch={ep}'); break
    if best_state: model.load_state_dict(best_state)
    model.eval(); preds=[]; tgts=[]
    with torch.no_grad():
        for xb,yb in dl_te:
            preds.append(model(xb.to(device).float()).cpu()); tgts.append(yb.float())
    metrics=eval_metrics(torch.cat(preds), torch.cat(tgts)); metrics['best_val']=float(best_val); metrics['seed']=int(seed)
    torch.save(model.state_dict(), out_dir/'model.pth')
    with open(out_dir/'metrics.json','w') as f: json.dump(metrics,f,indent=2)
    return metrics

def summarize(all_metrics, out_dir):
    summary={}
    for name, rows in all_metrics.items():
        summary[name]={}
        for k in [kk for kk in rows[0] if kk != 'seed']:
            vals=np.asarray([r[k] for r in rows], dtype=np.float64)
            summary[name][k]={'mean':float(vals.mean()), 'std':float(vals.std(ddof=0))}
    with open(out_dir/'summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary, indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raster', '--worldpop', dest='raster', required=True)
    ap.add_argument('--band', type=int, default=1)
    ap.add_argument('--label-name', default='population')
    ap.add_argument('--tile-index', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--bbox', default=None)
    ap.add_argument('--max-samples', type=int, default=50000)
    ap.add_argument('--sample-seed', type=int, default=42)
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--ae-radius-px', type=int, default=0, help='Square pixel radius for downstream aggregation, e.g. 0 for 100m cell, 5 for 500m on 100m AE.')
    ap.add_argument('--min-raw', type=float, default=None, help='Optional filter on raw raster value after sampling, e.g. 1 for populated WorldPop cells or 0 for positive GDP cells.')
    ap.add_argument('--valid-min', type=float, default=0.0, help='Minimum raster value accepted during initial sampling. Use -100 for LST.')
    ap.add_argument('--target-transform', choices=['log1p', 'identity'], default='identity', help='Transform used for cached y. Use identity for raw-value downstream targets.')
    ap.add_argument('--mask-tifs', default=None, help='Comma-separated land-cover mask GeoTIFFs. Samples are kept when mask value is in --mask-values.')
    ap.add_argument('--mask-values', default='50', help='Comma-separated integer mask values to keep; ESA WorldCover built-up is 50.')
    ap.add_argument('--jobs', type=int, default=max(1, min(16, (os.cpu_count() or 4) // 2)), help='Parallel tile workers for nationwide feature extraction.')
    ap.add_argument('--sample-from-tile-index', action='store_true', help='Sample raster cells from the union of tile-index bounding boxes. Use this for multi-region or antimeridian-crossing countries such as the US.')
    ap.add_argument('--sample-boundary', default=None, help='Optional vector boundary used to keep sampled raster cells inside the study region.')
    ap.add_argument('--sample-boundary-layer', default=None, help='Optional layer name for --sample-boundary, e.g. boundary in a GeoPackage.')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--prepare-only', action='store_true', help='Only build sample/AE/AETHER caches and metadata, skip downstream training.')
    args=ap.parse_args()
    out_dir=Path(args.out_dir); cache_dir=out_dir/'cache'; out_dir.mkdir(parents=True, exist_ok=True)
    bbox=resolve_bbox_arg(args.bbox, Path(args.raster), Path(args.tile_index)); device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    tag=f"{args.label_name}_b{args.band}_n{args.max_samples}_seed{args.sample_seed}_aer{args.ae_radius_px}"
    if args.sample_from_tile_index:
        samples=sample_raster_from_tile_index(
            Path(args.raster),
            Path(args.tile_index),
            args.max_samples,
            args.sample_seed,
            cache_dir/f'{tag}.npz',
            args.band,
            args.force,
            args.valid_min,
            args.target_transform,
            Path(args.sample_boundary) if args.sample_boundary else None,
            args.sample_boundary_layer,
        )
    else:
        samples=sample_raster(Path(args.raster), bbox, args.max_samples, args.sample_seed, cache_dir/f'{tag}.npz', args.band, args.force, args.valid_min, args.target_transform)
    lon,lat,Y=samples['lon'],samples['lat'],samples['y']
    raw = samples['raw'] if 'raw' in samples else samples.get('pop')
    keep = np.ones(len(Y), dtype=bool)
    if args.min_raw is not None:
        keep &= raw > float(args.min_raw)
    if args.mask_tifs:
        mask_paths = [Path(v.strip()) for v in args.mask_tifs.split(',') if v.strip()]
        mask_values = {int(v.strip()) for v in args.mask_values.split(',') if v.strip()}
        mv = sample_mask_values(lon, lat, mask_paths)
        keep &= np.isin(mv, list(mask_values))
        unique, counts = np.unique(mv[mv != -9999], return_counts=True)
        top = dict(zip(unique.tolist()[:20], counts.tolist()[:20]))
        print(f"[mask] keep_values={sorted(mask_values)} valid_masked={(mv != -9999).sum():,} class_counts_head={top}")
    if args.min_raw is not None or args.mask_tifs:
        lon, lat, raw = lon[keep], lat[keep], raw[keep]
        vals_target = transform_target_values(raw, args.target_transform)
        y_mean = float(vals_target.mean())
        y_std = float(vals_target.std() + 1e-6)
        Y = ((vals_target - y_mean) / y_std).reshape(-1, 1).astype(np.float32)
        print(f"[filter] kept={len(Y):,} y_mean={y_mean:.4f} y_std={y_std:.4f}")
    print(f"[samples] n={len(Y):,} y_mean={float(np.mean(Y)):.4f} y_std={float(np.std(Y)):.4f} ae_radius_px={args.ae_radius_px}")
    X_ae=window_mean_ae(lon, lat, Path(args.tile_index), int(args.ae_radius_px), cache_dir/f'{tag}.ae64.npy', args.force, jobs=int(args.jobs))
    X_aether=window_mean_aether_exact(lon, lat, Path(args.tile_index), int(args.ae_radius_px), Path(args.ckpt), device, cache_dir/f'{tag}.aether128_exact.npy', args.force, jobs=int(args.jobs))
    with open(out_dir/'meta.json','w') as f:
        json.dump({'raster': args.raster, 'band': int(args.band), 'label_name': args.label_name, 'ckpt': args.ckpt, 'bbox': list(bbox), 'n': int(len(Y)), 'ae_radius_px': int(args.ae_radius_px), 'min_raw': args.min_raw, 'valid_min': args.valid_min, 'target_transform': args.target_transform, 'mask_tifs': args.mask_tifs, 'mask_values': args.mask_values, 'sample_from_tile_index': bool(args.sample_from_tile_index), 'sample_boundary': args.sample_boundary, 'sample_boundary_layer': args.sample_boundary_layer, 'models':['AlphaEarth64','AETHER128']}, f, indent=2)
    if args.prepare_only:
        print('[prepare-only] cache build finished; skipping downstream training.')
        return
    seeds=[int(s) for s in args.seeds.split(',') if s.strip()]
    all_metrics={'AlphaEarth64':[], 'AETHER128':[]}
    for name,X in [('AlphaEarth64',X_ae),('AETHER128',X_aether)]:
        print(f"\n=== {name} X={X.shape} ===")
        for seed in seeds:
            metrics=train_one(X,Y,seed,device,out_dir/name/f'seed_{seed}',args.epochs,args.patience,args.batch_size,args.lr,args.hidden)
            print(f'[{name}] seed={seed} metrics={metrics}')
            all_metrics[name].append(metrics)
    summarize(all_metrics,out_dir)
if __name__=='__main__': main()
