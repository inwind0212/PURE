#!/usr/bin/env python
"""Build POI-region downstream caches with target-cell area-average alignment.

This is the CityRep-style protocol for the global POI-region downstream tasks:
reuse the existing selected target pixels and labels, but recompute features by
averaging the AE/AETHER raster pixels that overlap each target raster cell.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.aether_raster_io import (  # noqa: E402
    _aether_out_dim,
    _head_forward,
    _pixel_center_xy,
    decode_ae_stack,
    l2_normalize,
    load_aether_head,
)


@dataclass(frozen=True)
class TileRec:
    key: int
    path: str
    left: float
    bottom: float
    right: float
    top: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-task-dir", required=True, help="Existing *_same_poi_pixels task directory.")
    p.add_argument("--out-root", required=True, help="New cityrepavg cache root.")
    p.add_argument("--models", default="ae,aether", help="Comma-separated: ae,aether.")
    p.add_argument("--ckpt", default="", help="AETHER checkpoint. Required when models includes aether.")
    p.add_argument("--aether-suffix", default="aether128_shpos_areaavg_l2_v2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--batch-pixels", type=int, default=65536)
    p.add_argument(
        "--aether-input-radius-px",
        type=int,
        default=0,
        help="Mean-pool AE in a square input window before PURE projection; 1 gives a 3x3 window.",
    )
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--sample-limit", type=int, default=0, help="Debug only: truncate samples and write a debug cache.")
    p.add_argument(
        "--sum-storage",
        choices=["auto", "ram", "disk"],
        default="auto",
        help="Where to keep feature accumulators before writing final .npy files.",
    )
    p.add_argument(
        "--disk-sum-threshold-gb",
        type=float,
        default=48.0,
        help="In auto mode, use disk-backed accumulators when one sum array exceeds this size.",
    )
    p.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Remove completed tile part files after the final averaged cache is written.",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_save_average_npy(
    path: Path,
    sums: np.ndarray,
    weights: np.ndarray,
    *,
    normalize_rows: bool,
    chunk_rows: int = 1_000_000,
) -> None:
    """Write averaged features without materializing a second full array in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=sums.shape)
    for start in range(0, sums.shape[0], int(chunk_rows)):
        end = min(start + int(chunk_rows), sums.shape[0])
        w = weights[start:end]
        ok = w > 0
        block = np.zeros((end - start, sums.shape[1]), dtype=np.float32)
        if ok.any():
            block[ok] = sums[start:end][ok] / w[ok, None]
            if normalize_rows:
                block[ok] = l2_normalize(block[ok])
        out[start:end] = block
    out.flush()
    del out
    os.replace(tmp, path)


def make_sum_array(
    path: Path,
    shape: tuple[int, int] | tuple[int],
    *,
    dtype: np.dtype,
    storage: str,
    threshold_gb: float,
) -> tuple[np.ndarray, Path | None]:
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    use_disk = storage == "disk" or (storage == "auto" and nbytes >= float(threshold_gb) * (1024**3))
    if not use_disk:
        return np.zeros(shape, dtype=dtype), None
    tmp = path.with_name(f".{path.name}.sum.{os.getpid()}.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
    arr.flush()
    print(f"[sum-storage] disk path={tmp} shape={shape} size_gb={nbytes/(1024**3):.1f}", flush=True)
    return arr, tmp


def flush_array(arr: np.ndarray | None) -> None:
    if arr is not None and hasattr(arr, "flush"):
        arr.flush()


def cleanup_paths(paths: list[Path | None]) -> None:
    for path in paths:
        if path is not None and path.exists():
            path.unlink()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_old_meta(old_task_dir: Path) -> dict:
    meta_path = old_task_dir / "poi_region_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def one_file(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern} under {root}, found {len(files)}")
    return files[0]


def require_source_path(path: Path, field: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata field {field!r} points to a missing file: {path}. "
            "Update poi_region_meta.json explicitly; final code does not guess archive paths."
        )
    return path


def prepare_task_dir(old_task_dir: Path, out_root: Path, sample_src: Path) -> tuple[Path, Path, Path]:
    task_dir = out_root / old_task_dir.name
    cache_dir = task_dir / "cache"
    task_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(old_task_dir.glob("*.parquet")) + sorted(old_task_dir.glob("*label_meta.json")):
        link_or_copy(path, task_dir / path.name)

    sample_dst = cache_dir / sample_src.name
    link_or_copy(sample_src, sample_dst)
    return task_dir, cache_dir, sample_dst


def target_cell_bounds(raster_path: Path, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, ...]:
    with rasterio.open(raster_path) as src:
        tr = src.transform
        if not (abs(tr.b) < 1e-12 and abs(tr.d) < 1e-12):
            raise ValueError(f"Rotated/sheared target raster is not supported: {tr}")
        left = tr.c + cols.astype(np.float64) * tr.a
        right = tr.c + (cols.astype(np.float64) + 1.0) * tr.a
        y0 = tr.f + rows.astype(np.float64) * tr.e
        y1 = tr.f + (rows.astype(np.float64) + 1.0) * tr.e
        bottom = np.minimum(y0, y1)
        top = np.maximum(y0, y1)
    return left, bottom, right, top


def sample_rows_cols(samples: np.lib.npyio.NpzFile, raster_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if "row" in samples.files and "col" in samples.files:
        return samples["row"].astype(np.int64), samples["col"].astype(np.int64)
    lon = samples["lon"].astype(np.float64)
    lat = samples["lat"].astype(np.float64)
    with rasterio.open(raster_path) as src:
        inv = ~src.transform
        cols_f, rows_f = inv * (lon, lat)
    return np.floor(rows_f).astype(np.int64), np.floor(cols_f).astype(np.int64)


def build_tile_records(tile_index_path: Path) -> tuple[list[TileRec], dict[int, TileRec], float, float, float, float, int]:
    tiles = pd.read_parquet(tile_index_path)
    required = {"path", "left", "bottom", "right", "top"}
    missing = required - set(tiles.columns)
    if missing:
        raise KeyError(f"{tile_index_path} missing columns: {sorted(missing)}")
    widths = tiles["right"].to_numpy(np.float64) - tiles["left"].to_numpy(np.float64)
    heights = tiles["top"].to_numpy(np.float64) - tiles["bottom"].to_numpy(np.float64)
    tile_w = float(np.median(widths))
    tile_h = float(np.median(heights))
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(f"Invalid tile dimensions in {tile_index_path}")
    left0 = float(tiles["left"].min())
    bottom0 = float(tiles["bottom"].min())
    gx = np.rint((tiles["left"].to_numpy(np.float64) - left0) / tile_w).astype(np.int64)
    gy = np.rint((tiles["bottom"].to_numpy(np.float64) - bottom0) / tile_h).astype(np.int64)
    ncols = int(gx.max()) + 1
    recs: list[TileRec] = []
    rec_by_key: dict[int, TileRec] = {}
    for row, tx, ty in zip(tiles.itertuples(index=False), gx, gy):
        key = int(ty * ncols + tx)
        rec = TileRec(
            key=key,
            path=str(row.path),
            left=float(row.left),
            bottom=float(row.bottom),
            right=float(row.right),
            top=float(row.top),
        )
        recs.append(rec)
        rec_by_key[key] = rec
    return recs, rec_by_key, left0, bottom0, tile_w, tile_h, ncols


def _candidate_key_groups(
    keys: np.ndarray,
    idx: np.ndarray,
    rec_by_key: dict[int, TileRec],
    cell_left: np.ndarray,
    cell_bottom: np.ndarray,
    cell_right: np.ndarray,
    cell_top: np.ndarray,
) -> list[tuple[TileRec, np.ndarray]]:
    if len(keys) == 0:
        return []
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    idx_sorted = idx[order]
    starts = np.r_[0, np.nonzero(keys_sorted[1:] != keys_sorted[:-1])[0] + 1]
    stops = np.r_[starts[1:], len(keys_sorted)]
    out = []
    for start, stop in zip(starts, stops):
        key = int(keys_sorted[start])
        rec = rec_by_key.get(key)
        if rec is None:
            continue
        sub = idx_sorted[start:stop]
        overlaps = (
            (cell_right[sub] > rec.left)
            & (cell_left[sub] < rec.right)
            & (cell_top[sub] > rec.bottom)
            & (cell_bottom[sub] < rec.top)
        )
        if overlaps.any():
            out.append((rec, sub[overlaps].astype(np.int64, copy=False)))
    return out


def build_tile_tasks(
    tile_index_path: Path,
    cell_left: np.ndarray,
    cell_bottom: np.ndarray,
    cell_right: np.ndarray,
    cell_top: np.ndarray,
) -> list[tuple[TileRec, np.ndarray]]:
    _, rec_by_key, left0, bottom0, tile_w, tile_h, ncols = build_tile_records(tile_index_path)
    eps_right = np.nextafter(cell_right, cell_left)
    eps_top = np.nextafter(cell_top, cell_bottom)
    gx0 = np.floor((cell_left - left0) / tile_w).astype(np.int64)
    gx1 = np.floor((eps_right - left0) / tile_w).astype(np.int64)
    gy0 = np.floor((cell_bottom - bottom0) / tile_h).astype(np.int64)
    gy1 = np.floor((eps_top - bottom0) / tile_h).astype(np.int64)
    idx_all = np.arange(len(cell_left), dtype=np.int64)
    groups: list[tuple[TileRec, np.ndarray]] = []

    combos = [
        (gx0, gy0, np.ones(len(idx_all), dtype=bool)),
        (gx1, gy0, gx1 != gx0),
        (gx0, gy1, gy1 != gy0),
        (gx1, gy1, (gx1 != gx0) & (gy1 != gy0)),
    ]
    for gx, gy, mask in combos:
        if not mask.any():
            continue
        ii = idx_all[mask]
        keys = (gy[mask] * np.int64(ncols) + gx[mask]).astype(np.int64)
        groups.extend(_candidate_key_groups(keys, ii, rec_by_key, cell_left, cell_bottom, cell_right, cell_top))

    merged: dict[int, list[np.ndarray]] = {}
    recs: dict[int, TileRec] = {}
    for rec, idx in groups:
        merged.setdefault(rec.key, []).append(idx)
        recs[rec.key] = rec
    tasks = []
    for key in sorted(merged):
        idx = np.unique(np.concatenate(merged[key])).astype(np.int64)
        tasks.append((recs[key], idx))
    return tasks


def overlap_lins_weights(
    src,
    rec: TileRec,
    sample_idx: np.ndarray,
    cell_left: np.ndarray,
    cell_bottom: np.ndarray,
    cell_right: np.ndarray,
    cell_top: np.ndarray,
    valid_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_lins: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    lengths = np.zeros(len(sample_idx), dtype=np.int32)
    tr = src.transform
    xres = float(tr.a)
    yres = abs(float(tr.e))
    src_left = float(tr.c)
    src_top = float(tr.f)
    for j, si in enumerate(sample_idx):
        left = max(float(cell_left[si]), rec.left)
        right = min(float(cell_right[si]), rec.right)
        bottom = max(float(cell_bottom[si]), rec.bottom)
        top = min(float(cell_top[si]), rec.top)
        if not (right > left and top > bottom):
            continue
        c0 = max(0, int(math.floor((left - src_left) / xres)))
        c1 = min(src.width, int(math.ceil((right - src_left) / xres)))
        r0 = max(0, int(math.floor((src_top - top) / yres)))
        r1 = min(src.height, int(math.ceil((src_top - bottom) / yres)))
        if c1 <= c0 or r1 <= r0:
            continue
        cols = np.arange(c0, c1, dtype=np.int64)
        rows = np.arange(r0, r1, dtype=np.int64)
        pix_left = src_left + cols.astype(np.float64) * xres
        pix_right = pix_left + xres
        wx = np.minimum(pix_right, right) - np.maximum(pix_left, left)
        pix_top = src_top - rows.astype(np.float64) * yres
        pix_bottom = pix_top - yres
        wy = np.minimum(pix_top, top) - np.maximum(pix_bottom, bottom)
        wx = np.clip(wx, 0.0, None)
        wy = np.clip(wy, 0.0, None)
        if not (np.any(wx > 0) and np.any(wy > 0)):
            continue
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        ww = (wy[:, None] * wx[None, :]).astype(np.float32, copy=False)
        lin = (rr * np.int64(src.width) + cc).ravel()
        weight = ww.ravel()
        ok = (weight > 0) & valid_flat[lin]
        if not ok.any():
            continue
        lin = lin[ok].astype(np.int64, copy=False)
        weight = weight[ok].astype(np.float32, copy=False)
        all_lins.append(lin)
        all_weights.append(weight)
        lengths[j] = int(len(lin))
    if not all_lins:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), lengths
    return np.concatenate(all_lins), np.concatenate(all_weights), lengths


def infer_aether_pixels(
    head: torch.nn.Module,
    flat_ae: np.ndarray,
    valid_flat: np.ndarray,
    src,
    unique_lins: np.ndarray,
    batch_pixels: int,
    input_radius_px: int,
) -> np.ndarray:
    device = next(head.parameters()).device
    rows = (unique_lins // np.int64(src.width)).astype(np.int64, copy=False)
    cols = (unique_lins % np.int64(src.width)).astype(np.int64, copy=False)
    pix_lon, pix_lat = _pixel_center_xy(src.transform, rows, cols)
    latlon = np.stack([pix_lat, pix_lon], axis=1).astype(np.float32, copy=False)
    radius = int(input_radius_px)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    dr = dr.ravel()
    dc = dc.ravel()
    outs = []
    with torch.inference_mode():
        for start in range(0, len(unique_lins), int(batch_pixels)):
            end = min(start + int(batch_pixels), len(unique_lins))
            if radius == 0:
                pix = flat_ae[:, unique_lins[start:end]].T.astype(np.float32, copy=False)
            else:
                rr = rows[start:end, None] + dr[None, :]
                cc = cols[start:end, None] + dc[None, :]
                inside = (rr >= 0) & (rr < int(src.height)) & (cc >= 0) & (cc < int(src.width))
                rr_safe = np.clip(rr, 0, int(src.height) - 1)
                cc_safe = np.clip(cc, 0, int(src.width) - 1)
                neighbor_lins = rr_safe * np.int64(src.width) + cc_safe
                keep = inside & valid_flat[neighbor_lins]
                values = flat_ae[:, neighbor_lins.ravel()].T.reshape(end - start, -1, 64)
                counts = np.maximum(keep.sum(axis=1, dtype=np.int64), 1)
                pix = (values * keep[..., None]).sum(axis=1) / counts[:, None]
                pix = pix.astype(np.float32, copy=False)
            pix = l2_normalize(pix)
            xb = torch.from_numpy(pix).to(device, non_blocking=True).float()
            outs.append(_head_forward(head, xb, latlon[start:end]).detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32, copy=False)


def aggregate_tile(
    rec: TileRec,
    idx: np.ndarray,
    cell_left: np.ndarray,
    cell_bottom: np.ndarray,
    cell_right: np.ndarray,
    cell_top: np.ndarray,
    *,
    do_ae: bool,
    head: torch.nn.Module | None,
    out_dim: int,
    sample_batch: int,
    batch_pixels: int,
    aether_input_radius_px: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    ae_part = np.zeros((len(idx), 64), dtype=np.float32) if do_ae else None
    aether_part = np.zeros((len(idx), out_dim), dtype=np.float32) if head is not None else None
    w_part = np.zeros(len(idx), dtype=np.float32)
    with rasterio.open(rec.path) as src:
        # Reading all 64 bands for an entire 400-km AE tile is extremely
        # wasteful when only a few settlement cells fall in that tile (common
        # in Alaska, northern Canada and small islands).  Crop to the smallest
        # source-pixel window covering this task's target-cell intersections.
        tr = src.transform
        xres = float(tr.a)
        yres = abs(float(tr.e))
        left = max(float(np.min(cell_left[idx])), float(rec.left))
        right = min(float(np.max(cell_right[idx])), float(rec.right))
        bottom = max(float(np.min(cell_bottom[idx])), float(rec.bottom))
        top = min(float(np.max(cell_top[idx])), float(rec.top))
        col0 = max(0, int(math.floor((left - float(tr.c)) / xres)))
        col1 = min(src.width, int(math.ceil((right - float(tr.c)) / xres)))
        row0 = max(0, int(math.floor((float(tr.f) - top) / yres)))
        row1 = min(src.height, int(math.ceil((float(tr.f) - bottom) / yres)))
        if col1 <= col0 or row1 <= row0:
            return idx, ae_part, aether_part, w_part
        radius = int(aether_input_radius_px)
        read_col0 = max(0, col0 - radius)
        read_col1 = min(src.width, col1 + radius)
        read_row0 = max(0, row0 - radius)
        read_row1 = min(src.height, row1 + radius)
        window = Window(read_col0, read_row0, read_col1 - read_col0, read_row1 - read_row0)
        raw = src.read(indexes=list(range(1, 65)), window=window)
        nodata = -128 if src.nodata is None and raw.dtype == np.int8 else src.nodata
        if nodata is None:
            valid = np.isfinite(raw[0])
        else:
            valid = raw[0] != nodata
        tile = decode_ae_stack(raw, nodata)
        del raw
        flat = tile.reshape(64, -1)
        valid_flat = valid.ravel()
        cropped_transform = src.window_transform(window)
        core_window = Window(col0, row0, col1 - col0, row1 - row0)
        cropped_bounds = rasterio.windows.bounds(core_window, src.transform)
        cropped_rec = TileRec(
            key=rec.key,
            path=rec.path,
            left=float(cropped_bounds[0]),
            bottom=float(cropped_bounds[1]),
            right=float(cropped_bounds[2]),
            top=float(cropped_bounds[3]),
        )

        cropped_src = SimpleNamespace(
            transform=cropped_transform,
            width=int(window.width),
            height=int(window.height),
        )
        for start in range(0, len(idx), int(sample_batch)):
            end = min(start + int(sample_batch), len(idx))
            batch_idx = idx[start:end]
            lins, weights, lengths = overlap_lins_weights(
                cropped_src,
                cropped_rec,
                batch_idx,
                cell_left,
                cell_bottom,
                cell_right,
                cell_top,
                valid_flat,
            )
            if len(lins) == 0:
                continue
            unique_lins, inverse = np.unique(lins, return_inverse=True)
            if head is not None:
                z_unique = infer_aether_pixels(
                    head,
                    flat,
                    valid_flat,
                    cropped_src,
                    unique_lins,
                    batch_pixels,
                    radius,
                )
            else:
                z_unique = None
            offset = 0
            for j, n_pix in enumerate(lengths):
                n_pix = int(n_pix)
                if n_pix <= 0:
                    continue
                sl = slice(offset, offset + n_pix)
                inv = inverse[sl]
                ww = weights[sl]
                w_sum = float(ww.sum())
                if w_sum <= 0:
                    offset += n_pix
                    continue
                if ae_part is not None:
                    ae_vals = flat[:, unique_lins[inv]].T
                    ae_part[start + j] += (ae_vals * ww[:, None]).sum(axis=0)
                if aether_part is not None and z_unique is not None:
                    aether_part[start + j] += (z_unique[inv] * ww[:, None]).sum(axis=0)
                w_part[start + j] += w_sum
                offset += n_pix
        del tile
    return idx, ae_part, aether_part, w_part


def part_path(parts_dir: Path, tile_key: int) -> Path:
    return parts_dir / f"tile_{int(tile_key):06d}.npz"


def save_part(path: Path, idx: np.ndarray, ae: np.ndarray | None, aether: np.ndarray | None, w: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.npz")
    payload = {"idx": idx.astype(np.int64, copy=False), "w": w.astype(np.float32, copy=False)}
    if ae is not None:
        payload["ae"] = ae.astype(np.float32, copy=False)
    if aether is not None:
        payload["aether"] = aether.astype(np.float32, copy=False)
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def load_part(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    z = np.load(path)
    return (
        z["idx"].astype(np.int64, copy=False),
        z["ae"].astype(np.float32, copy=False) if "ae" in z.files else None,
        z["aether"].astype(np.float32, copy=False) if "aether" in z.files else None,
        z["w"].astype(np.float32, copy=False),
    )


def main() -> None:
    args = parse_args()
    if int(args.aether_input_radius_px) < 0:
        raise ValueError("--aether-input-radius-px must be non-negative")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    old_task_dir = Path(args.old_task_dir)
    out_root = Path(args.out_root)
    old_meta = load_old_meta(old_task_dir)
    if old_meta.get("raster"):
        old_meta["raster"] = str(require_source_path(Path(old_meta["raster"]), "raster"))
    if old_meta.get("tile_index"):
        old_meta["tile_index"] = str(require_source_path(Path(old_meta["tile_index"]), "tile_index"))
    sample_meta = old_meta.get("sample_npz")
    sample_src = Path(sample_meta) if sample_meta else one_file(old_task_dir / "cache", "*.npz")
    if not sample_src.exists():
        # Archived experiment directories may contain metadata with absolute paths
        # from the pre-archive location. Prefer the active task cache when present.
        sample_src = one_file(old_task_dir / "cache", "*.npz")
    task_dir, cache_dir, sample_dst = prepare_task_dir(old_task_dir, out_root, sample_src)

    models = {x.strip().lower() for x in args.models.split(",") if x.strip()}
    do_ae = "ae" in models
    do_aether = "aether" in models
    if not (do_ae or do_aether):
        raise ValueError("--models must include ae and/or aether")
    if do_aether and not args.ckpt:
        raise ValueError("--ckpt is required when --models includes aether")

    stem = sample_dst.name[:-4] if sample_dst.name.endswith(".npz") else sample_dst.stem
    if int(args.sample_limit) > 0:
        stem = f"{stem}_debugn{int(args.sample_limit)}"
    ae_path = cache_dir / f"{stem}.ae64.npy"
    aether_path = cache_dir / f"{stem}.{args.aether_suffix}.npy"
    done_ae = (not do_ae) or (ae_path.exists() and not args.force)
    done_aether = (not do_aether) or (aether_path.exists() and not args.force)
    if done_ae and done_aether:
        print(f"[cache] exists task={task_dir}", flush=True)
        return

    samples = np.load(sample_src)
    n_all = int(len(samples["lon"]))
    n = min(n_all, int(args.sample_limit)) if int(args.sample_limit) > 0 else n_all
    rows, cols = sample_rows_cols(samples, Path(old_meta["raster"]))
    rows = rows[:n]
    cols = cols[:n]
    cell_left, cell_bottom, cell_right, cell_top = target_cell_bounds(Path(old_meta["raster"]), rows, cols)

    head = None
    out_dim = 0
    if do_aether and not done_aether:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        head = load_aether_head(Path(args.ckpt), device)
        out_dim = _aether_out_dim(head, device)
        print(f"[aether] ckpt={args.ckpt} device={device} out_dim={out_dim}", flush=True)

    tasks = build_tile_tasks(Path(old_meta["tile_index"]), cell_left, cell_bottom, cell_right, cell_top)
    assigned = np.zeros(n, dtype=bool)
    for _, idx in tasks:
        assigned[idx] = True
    if not assigned.all():
        print(f"[WARN] unassigned target cells: {int((~assigned).sum()):,}/{n:,}", flush=True)

    sum_tmp_paths: list[Path | None] = []
    if do_ae and not done_ae:
        ae_sum, ae_tmp = make_sum_array(
            ae_path,
            (n, 64),
            dtype=np.float32,
            storage=args.sum_storage,
            threshold_gb=float(args.disk_sum_threshold_gb),
        )
        sum_tmp_paths.append(ae_tmp)
    else:
        ae_sum = None
    if do_aether and not done_aether:
        aether_sum, aether_tmp = make_sum_array(
            aether_path,
            (n, out_dim),
            dtype=np.float32,
            storage=args.sum_storage,
            threshold_gb=float(args.disk_sum_threshold_gb),
        )
        sum_tmp_paths.append(aether_tmp)
    else:
        aether_sum = None
    w_sum, w_tmp = make_sum_array(
        cache_dir / f"{stem}.weights.npy",
        (n,),
        dtype=np.float32,
        storage=args.sum_storage,
        threshold_gb=float(args.disk_sum_threshold_gb),
    )
    sum_tmp_paths.append(w_tmp)
    part_tag = args.aether_suffix if do_aether else "ae64"
    parts_dir = cache_dir / f"{stem}.{part_tag}.cityrepavg.parts"
    if args.force and parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for rec, idx in tqdm(tasks, desc=f"cityrepavg {old_task_dir.name}", ncols=100):
        pp = part_path(parts_dir, rec.key)
        if pp.exists() and not args.force:
            idx_p, ae_p, aether_p, w_p = load_part(pp)
        else:
            idx_p, ae_p, aether_p, w_p = aggregate_tile(
                rec,
                idx,
                cell_left,
                cell_bottom,
                cell_right,
                cell_top,
                do_ae=(ae_sum is not None),
                head=head,
                out_dim=out_dim,
                sample_batch=int(args.sample_batch),
                batch_pixels=int(args.batch_pixels),
                aether_input_radius_px=int(args.aether_input_radius_px),
            )
            save_part(pp, idx_p, ae_p, aether_p, w_p)
        w_sum[idx_p] += w_p
        if ae_sum is not None and ae_p is not None:
            ae_sum[idx_p] += ae_p
        if aether_sum is not None and aether_p is not None:
            aether_sum[idx_p] += aether_p

    ok = w_sum > 0
    flush_array(ae_sum)
    flush_array(aether_sum)
    flush_array(w_sum)
    if ae_sum is not None:
        atomic_save_average_npy(ae_path, ae_sum, w_sum, normalize_rows=True)
        print(f"[write] {ae_path} shape={ae_sum.shape}", flush=True)
    if aether_sum is not None:
        atomic_save_average_npy(aether_path, aether_sum, w_sum, normalize_rows=True)
        print(f"[write] {aether_path} shape={aether_sum.shape}", flush=True)
    cleanup_paths(sum_tmp_paths)
    parts_cleaned = False
    if args.cleanup_parts and parts_dir.exists():
        shutil.rmtree(parts_dir)
        parts_cleaned = True

    elapsed = time.time() - started
    new_meta = {
        "protocol": (
            "cityrep_target_cell_area_average_l2_v2"
            if int(args.aether_input_radius_px) == 0
            else f"cityrep_target_cell_area_average_l2_v2_aether_input_window_r{int(args.aether_input_radius_px)}"
        ),
        "source_task_dir": str(old_task_dir),
        "source_sample_npz": str(sample_src),
        "sample_npz": str(sample_dst),
        "source_meta": str(old_task_dir / "poi_region_meta.json"),
        "region": old_meta.get("region"),
        "task_dir_name": old_task_dir.name,
        "raster": old_meta.get("raster"),
        "band": old_meta.get("band"),
        "label_name": old_meta.get("label_name"),
        "tile_index": old_meta.get("tile_index"),
        "n_source": n_all,
        "n": n,
        "models_requested": sorted(models),
        "ae64": str(ae_path) if do_ae else None,
        "aether": str(aether_path) if do_aether else None,
        "ckpt": args.ckpt if do_aether else None,
        "aether_suffix": args.aether_suffix if do_aether else None,
        "aether_input_radius_px": int(args.aether_input_radius_px) if do_aether else None,
        "aggregation": {
            "source_grid": "AlphaEarth 100m pixels",
            "target_grid": "downstream target raster cells from cached row/col",
            "weight": "overlap area in EPSG:4326 degree^2",
            "ae_feature": "area-weighted decoded AE64, then L2-normalized per target cell",
            "aether_feature": (
                "mean-pool AE in a square input window at each overlapping pixel, infer PURE at the center position, "
                "then area-weighted average and L2-normalize per target cell"
                if int(args.aether_input_radius_px) > 0
                else "infer AETHER per overlapping AE pixel, then area-weighted average and L2-normalize per target cell"
            ),
        },
        "n_tiles": len(tasks),
        "n_with_source_overlap": int(ok.sum()),
        "parts_dir": str(parts_dir),
        "parts_cleaned": parts_cleaned,
        "elapsed_sec": elapsed,
        "sample_batch": int(args.sample_batch),
        "batch_pixels": int(args.batch_pixels),
        "torch_threads": int(args.torch_threads),
        "sum_storage": args.sum_storage,
        "disk_sum_threshold_gb": float(args.disk_sum_threshold_gb),
    }
    (task_dir / "cityrepavg_meta.json").write_text(json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(new_meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
