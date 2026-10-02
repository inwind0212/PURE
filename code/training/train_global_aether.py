#!/usr/bin/env python
"""Train joint AETHER directly from a spatially sorted feature store."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import AEProjSH, TextProj  # noqa: E402
from training_utils import ensure_dir  # noqa: E402
from spatial_sampling import JointGridRingBatchSampler, iter_dataset_batches  # noqa: E402


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--resume-ckpt", default="")
    ap.add_argument("--resume-auto", action="store_true")
    ap.add_argument("--resume-model-only", action="store_true", help="Load model weights only and restart optimizer/scheduler.")
    return ap.parse_args()


def relational_preservation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 0.1,
    sample_size: int = 0,
) -> torch.Tensor:
    """Preserve the teacher's within-batch neighbour distribution.

    The student and teacher may have different feature dimensions.  Only
    pairwise cosine relations are distilled, so the deployed representation
    remains the single student embedding.
    """
    if student.ndim != 2 or teacher.ndim != 2:
        raise ValueError("student and teacher must both have shape [batch, dim]")
    if student.shape[0] != teacher.shape[0]:
        raise ValueError("student and teacher batch sizes must match")
    if student.shape[0] < 2:
        raise ValueError("relational preservation requires at least two samples")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    if sample_size and student.shape[0] > sample_size:
        indices = torch.randperm(student.shape[0], device=student.device)[:sample_size]
        student = student.index_select(0, indices)
        teacher = teacher.index_select(0, indices)

    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    student_logits = (student @ student.T) / temperature
    teacher_logits = (teacher @ teacher.T) / temperature
    diagonal = torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
    student_logits = student_logits.masked_fill(diagonal, -1e4)
    teacher_logits = teacher_logits.masked_fill(diagonal, -1e4)
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    )


class SpatialSortedFeatureStoreDataset:
    def __init__(
        self,
        store_dir: Path,
        text_dim: int,
        *,
        return_spatial: bool = False,
        include_regions: list[str] | None = None,
        exclude_regions: list[str] | None = None,
        row_filter_path: str | Path | None = None,
        row_filter_name: str = "",
        io_read_workers: int = 1,
    ):
        self.store_dir = Path(store_dir)
        self.meta_path = self.store_dir / "feature_store_meta.json"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing feature store metadata: {self.meta_path}")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if not bool(self.meta.get("complete", False)):
            raise RuntimeError(f"Feature store is incomplete: {self.meta_path}")
        paths = self.meta["paths"]
        self.ae_base = np.load(paths["ae_base"], mmap_mode="r")
        self.ae_aug = np.load(paths["ae_aug"], mmap_mode="r")
        self.text = np.load(paths["text"], mmap_mode="r")
        # Batches are globally sampled and sorted only within each fetch block.
        # Disable Linux's sequential readahead for these sparse feature reads;
        # otherwise a requested 4-KiB page can pull in a much larger unused
        # window from the 216-GiB store. This changes I/O policy only.
        if hasattr(mmap, "MADV_RANDOM"):
            for arr in (self.ae_base, self.ae_aug, self.text):
                mapped = getattr(arr, "_mmap", None)
                if mapped is not None and hasattr(mapped, "madvise"):
                    mapped.madvise(mmap.MADV_RANDOM)
        self.spatial_x = np.load(paths["sp_x"], mmap_mode="r")
        self.spatial_y = np.load(paths["sp_y"], mmap_mode="r")
        self.return_spatial = bool(return_spatial)
        self.row_index: np.ndarray | None = None
        self.row_filter: dict | None = None
        self.io_read_workers = max(1, int(io_read_workers))
        self._explicit_component_names: list[str] | None = None
        self._explicit_component_counts: np.ndarray | None = None
        if self.text.shape[1] != int(text_dim):
            raise ValueError(f"text_dim mismatch: store={self.text.shape[1]} cfg={text_dim}")
        n = int(self.meta["rows"])
        for name, arr in [
            ("ae_base", self.ae_base),
            ("ae_aug", self.ae_aug),
            ("text", self.text),
            ("sp_x", self.spatial_x),
            ("sp_y", self.spatial_y),
        ]:
            if arr.shape[0] != n:
                raise ValueError(f"{name} row count mismatch: {arr.shape[0]:,} vs {n:,}")
        self._all_component_names, self._all_component_counts = self._read_component_names_and_counts()
        self._component_keep_mask = np.ones(len(self._all_component_names), dtype=bool)
        self._init_row_filter(include_regions=include_regions, exclude_regions=exclude_regions)
        self._init_explicit_row_filter(row_filter_path, row_filter_name=row_filter_name)
        print(
            f"[FEATURE-STORE] {self.store_dir} rows={len(self):,}/{n:,} sort={self.meta.get('sort')} "
            f"ae_dim={self.ae_base.shape[1]} text_dim={self.text.shape[1]}"
            + (f" filter={self.row_filter}" if self.row_filter else ""),
            flush=True,
        )

    def __len__(self) -> int:
        if self.row_index is not None:
            return int(self.row_index.shape[0])
        return int(self.meta["rows"])

    def summary(self) -> dict:
        return {
            "store_dir": str(self.store_dir),
            "meta": str(self.meta_path),
            "rows": int(len(self)),
            "source_rows": int(self.meta["rows"]),
            "sort": self.meta.get("sort"),
            "hilbert_bits": self.meta.get("hilbert_bits"),
            "row_filter": self.row_filter,
        }

    def _read_component_names_and_counts(self) -> tuple[list[str], np.ndarray]:
        components = self.meta.get("components", [])
        if not components:
            raise ValueError(f"Feature store metadata has no components: {self.meta_path}")
        names = [str(c.get("region", f"component_{i}")) for i, c in enumerate(components)]
        counts = np.asarray([int(c.get("n", c.get("n_source", 0))) for c in components], dtype=np.int64)
        if int(counts.sum()) != len(self):
            raise ValueError(
                f"Feature store component counts sum to {int(counts.sum()):,}, expected rows={len(self):,}"
            )
        return names, counts

    def all_component_names_and_counts(self) -> tuple[list[str], np.ndarray]:
        return self._all_component_names, self._all_component_counts

    def component_names_and_counts(self) -> tuple[list[str], np.ndarray]:
        if self._explicit_component_names is not None and self._explicit_component_counts is not None:
            return self._explicit_component_names, self._explicit_component_counts
        names = [n for n, keep in zip(self._all_component_names, self._component_keep_mask) if bool(keep)]
        counts = self._all_component_counts[self._component_keep_mask]
        return names, counts

    @staticmethod
    def _as_region_set(values) -> set[str]:
        if values is None:
            return set()
        if isinstance(values, str):
            values = [values]
        return {str(v) for v in values}

    def _init_row_filter(self, *, include_regions, exclude_regions) -> None:
        include = self._as_region_set(include_regions)
        exclude = self._as_region_set(exclude_regions)
        if not include and not exclude:
            return
        known = set(self._all_component_names)
        unknown = (include | exclude) - known
        if unknown:
            raise ValueError(f"Unknown feature-store regions in row filter: {sorted(unknown)}; known={self._all_component_names}")
        keep_names = set(self._all_component_names)
        if include:
            keep_names &= include
        keep_names -= exclude
        if not keep_names:
            raise ValueError("Feature-store row filter removed all regions.")
        self._component_keep_mask = np.asarray([name in keep_names for name in self._all_component_names], dtype=bool)
        filter_payload = {
            "include_regions": sorted(include),
            "exclude_regions": sorted(exclude),
            "keep_regions": [name for name in self._all_component_names if name in keep_names],
        }
        digest = hashlib.sha1(json.dumps(filter_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        cache_dir = self.store_dir / "row_filters"
        cache_dir.mkdir(parents=True, exist_ok=True)
        row_index_path = cache_dir / f"rows_{digest}.npy"
        manifest_path = cache_dir / f"rows_{digest}.json"
        if not row_index_path.exists():
            source_path = self.meta.get("paths", {}).get("source_global_index")
            if not source_path:
                raise FileNotFoundError("Feature-store region filtering requires paths.source_global_index.")
            source = np.load(str(source_path), mmap_mode="r")
            if source.shape[0] != int(self.meta["rows"]):
                raise ValueError(f"source_global_index length mismatch: {source.shape[0]:,} vs {int(self.meta['rows']):,}")
            boundaries = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(self._all_component_counts)])
            keep_parts: list[np.ndarray] = []
            chunk_rows = 4_000_000
            print(
                f"[ROW-FILTER] build cache={row_index_path} keep={filter_payload['keep_regions']} "
                f"source_rows={int(self.meta['rows']):,}",
                flush=True,
            )
            for start in tqdm(range(0, int(self.meta["rows"]), chunk_rows), desc="Build row filter", ncols=90):
                end = min(int(self.meta["rows"]), start + chunk_rows)
                src = np.asarray(source[start:end], dtype=np.int64)
                comp = np.searchsorted(boundaries[1:], src, side="right")
                valid = (comp >= 0) & (comp < len(self._all_component_names))
                keep = np.zeros(end - start, dtype=bool)
                keep[valid] = self._component_keep_mask[comp[valid]]
                if np.any(keep):
                    keep_parts.append(np.arange(start, end, dtype=np.int64)[keep])
            rows = np.concatenate(keep_parts).astype(np.int64, copy=False) if keep_parts else np.empty(0, dtype=np.int64)
            if len(rows) == 0:
                raise RuntimeError(f"Row filter produced no rows: {filter_payload}")
            tmp_path = row_index_path.with_suffix(".tmp.npy")
            np.save(tmp_path, rows)
            tmp_path.replace(row_index_path)
            manifest_path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        **filter_payload,
                        "rows": int(len(rows)),
                        "source_rows": int(self.meta["rows"]),
                        "created_at": now_iso(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.row_index = np.load(row_index_path, mmap_mode="r")
        self.row_filter = {
            **filter_payload,
            "row_index": str(row_index_path),
            "rows": int(self.row_index.shape[0]),
        }
        self.spatial_x = np.asarray(self.spatial_x[self.row_index], dtype=np.int32)
        self.spatial_y = np.asarray(self.spatial_y[self.row_index], dtype=np.int32)

    def _init_explicit_row_filter(self, row_filter_path, *, row_filter_name: str = "") -> None:
        if row_filter_path is None or str(row_filter_path).strip() == "":
            return
        if self.row_index is not None:
            raise ValueError("Use either include/exclude region filters or paths.row_filter, not both.")
        path = Path(str(row_filter_path))
        if not path.exists():
            raise FileNotFoundError(f"Explicit row filter not found: {path}")
        rows = np.load(path, mmap_mode="r")
        if rows.ndim != 1:
            raise ValueError(f"Explicit row filter must be a 1D npy array: {path} shape={rows.shape}")
        if rows.shape[0] == 0:
            raise ValueError(f"Explicit row filter is empty: {path}")
        if rows.dtype.kind not in {"i", "u"}:
            raise ValueError(f"Explicit row filter must contain integer row indices: {path} dtype={rows.dtype}")
        max_row = int(np.max(rows))
        min_row = int(np.min(rows))
        if min_row < 0 or max_row >= int(self.meta["rows"]):
            raise ValueError(
                f"Explicit row filter indices out of range: min={min_row:,} max={max_row:,} "
                f"source_rows={int(self.meta['rows']):,}"
            )
        self.row_index = rows
        name = str(row_filter_name).strip() or path.stem
        self.row_filter = {
            "mode": "explicit_row_filter",
            "name": name,
            "row_index": str(path),
            "rows": int(rows.shape[0]),
        }
        self._explicit_component_names = [name]
        self._explicit_component_counts = np.asarray([int(rows.shape[0])], dtype=np.int64)
        self.spatial_x = np.asarray(self.spatial_x[self.row_index], dtype=np.int32)
        self.spatial_y = np.asarray(self.spatial_y[self.row_index], dtype=np.int32)

    def source_rows_for_local(self, local_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(local_rows, dtype=np.int64)
        if self.row_index is None:
            return rows
        return np.asarray(self.row_index[rows], dtype=np.int64)

    def get_batch(self, indices: list[int] | np.ndarray):
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("Expected non-empty 1D batch indices.")
        order = np.argsort(idx, kind="mergesort")
        sorted_idx = idx[order]
        source_sorted_idx = self.source_rows_for_local(sorted_idx)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order), dtype=order.dtype)
        ae_base = self._read_rows(self.ae_base, source_sorted_idx)[inv]
        ae_aug = self._read_rows(self.ae_aug, source_sorted_idx)[inv]
        text = self._read_rows(self.text, source_sorted_idx)[inv]
        batch = (torch.from_numpy(ae_base), torch.from_numpy(ae_aug), torch.from_numpy(text))
        if not self.return_spatial:
            return batch
        sp_x = np.asarray(self.spatial_x[sorted_idx], dtype=np.float32)[inv]
        sp_y = np.asarray(self.spatial_y[sorted_idx], dtype=np.float32)[inv]
        return (*batch, torch.from_numpy(sp_x), torch.from_numpy(sp_y))

    def _read_rows(self, array: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
        """Gather sorted mmap rows with enough concurrent reads to queue the NVMe.

        Every worker writes a disjoint output slice, so this is numerically
        identical to ``np.asarray(array[sorted_indices], dtype=np.float32)``.
        """
        idx = np.asarray(sorted_indices, dtype=np.int64)
        workers = min(self.io_read_workers, int(idx.size))
        if workers <= 1:
            return np.asarray(array[idx], dtype=np.float32)
        output = np.empty((idx.size, *array.shape[1:]), dtype=np.float32)
        edges = np.linspace(0, idx.size, workers + 1, dtype=np.int64)

        def read_slice(lo: int, hi: int) -> None:
            output[lo:hi] = array[idx[lo:hi]]

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmap-read") as pool:
            futures = [
                pool.submit(read_slice, int(lo), int(hi))
                for lo, hi in zip(edges[:-1], edges[1:])
                if hi > lo
            ]
            for future in futures:
                future.result()
        return output


def iter_prefetched_dataset_batches(dataset, sampler, fetch_chunk_batches: int):
    """Overlap one feature-store chunk read with GPU work on the prior chunk."""
    chunk_size = max(1, int(fetch_chunk_batches))
    sampler_iter = iter(sampler)

    def fetch_next():
        items = []
        for _ in range(chunk_size):
            try:
                items.append(next(sampler_iter))
            except StopIteration:
                break
        if not items:
            return None
        sizes = [len(rows) for rows in items]
        flat = np.concatenate(
            [np.asarray(rows, dtype=np.int64) for rows in items]
        )
        return sizes, dataset.get_batch(flat)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="feature-prefetch") as pool:
        future = pool.submit(fetch_next)
        while True:
            fetched = future.result()
            if fetched is None:
                break
            sizes, flat_batch = fetched
            future = pool.submit(fetch_next)
            start = 0
            for size in sizes:
                end = start + size
                yield tuple(value[start:end] for value in flat_batch)
                start = end


def iter_pipelined_dataset_batches(dataset, sampler, fetch_chunk_batches: int):
    """Pipeline deterministic index generation, mmap reads, and GPU work.

    The sampler is consumed by exactly one producer thread, so its RNG stream,
    batch order, and sampled rows are identical to the single-stage prefetch
    path.  A second thread reads the already-generated indices while the
    producer constructs the following chunk.
    """
    chunk_size = max(1, int(fetch_chunk_batches))
    sampler_iter = iter(sampler)
    index_queue: Queue = Queue(maxsize=1)
    data_queue: Queue = Queue(maxsize=1)

    def produce_indices() -> None:
        try:
            while True:
                items = []
                for _ in range(chunk_size):
                    try:
                        items.append(next(sampler_iter))
                    except StopIteration:
                        break
                if not items:
                    index_queue.put(("end",))
                    return
                sizes = [len(rows) for rows in items]
                flat = np.concatenate(
                    [np.asarray(rows, dtype=np.int64) for rows in items]
                )
                index_queue.put(("data", sizes, flat))
        except BaseException as exc:
            index_queue.put(("error", exc))

    def read_features() -> None:
        try:
            while True:
                item = index_queue.get()
                if item[0] == "end":
                    data_queue.put(("end",))
                    return
                if item[0] == "error":
                    data_queue.put(item)
                    return
                _, sizes, flat = item
                data_queue.put(("data", sizes, dataset.get_batch(flat)))
        except BaseException as exc:
            data_queue.put(("error", exc))

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="feature-pipeline") as pool:
        producer = pool.submit(produce_indices)
        reader = pool.submit(read_features)
        while True:
            fetched = data_queue.get()
            if fetched[0] == "end":
                break
            if fetched[0] == "error":
                raise fetched[1]
            _, sizes, flat_batch = fetched
            start = 0
            for size in sizes:
                end = start + size
                yield tuple(value[start:end] for value in flat_batch)
                start = end
        producer.result()
        reader.result()


def _anchor_region_weights(mode: str, names: list[str], counts: np.ndarray, sampling_cfg) -> np.ndarray:
    mode = str(mode)
    if mode == "balanced":
        weights = np.ones(len(names), dtype=np.float64)
    elif mode == "sqrt_count":
        weights = np.sqrt(np.maximum(counts.astype(np.float64), 1.0))
    elif mode == "count":
        weights = np.maximum(counts.astype(np.float64), 1.0)
    elif mode == "custom":
        raw = sampling_cfg.get("anchor_region_weights", {})
        weights = np.asarray([float(raw.get(name, 0.0)) for name in names], dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported sampling.anchor_region_mode={mode!r}; expected balanced, sqrt_count, count, custom, or density/none."
        )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"Invalid anchor region weights for mode={mode!r}: {dict(zip(names, weights.tolist()))}")
    return weights / weights.sum()


def build_anchor_region_groups(
    cfg,
    dataset: SpatialSortedFeatureStoreDataset,
    *,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, list[str], dict] | None:
    sampling_cfg = cfg.sampling
    mode = str(sampling_cfg.get("anchor_region_mode", "density"))
    if mode in {"", "none", "density", "global_density"}:
        return None

    paths = dataset.meta.get("paths", {})
    source_path = paths.get("source_global_index")
    if not source_path:
        raise FileNotFoundError("Region-balanced anchors require source_global_index in feature_store_meta.json.")
    source_path = str(source_path)
    all_names, all_counts = dataset.all_component_names_and_counts()
    names, counts = dataset.component_names_and_counts()
    boundaries = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(all_counts)])
    active_pos = np.full(len(all_names), -1, dtype=np.int64)
    active_orig = [all_names.index(name) for name in names]
    active_pos[np.asarray(active_orig, dtype=np.int64)] = np.arange(len(names), dtype=np.int64)
    max_per_component = int(sampling_cfg.get("anchor_region_max_per_component", 2_000_000))
    if max_per_component <= 0:
        raise ValueError("sampling.anchor_region_max_per_component must be positive.")
    min_per_component = int(sampling_cfg.get("anchor_region_min_per_component", 0))
    chunk_rows = int(sampling_cfg.get("anchor_region_build_chunk_rows", 4_000_000))
    chunk_rows = max(1, chunk_rows)
    cache_dir_raw = str(sampling_cfg.get("anchor_region_cache_dir", "")).strip()
    if cache_dir_raw:
        cache_dir = Path(cache_dir_raw)
    else:
        cache_name = f"anchor_region_{mode}_seed{int(seed)}_max{max_per_component}_min{min_per_component}"
        if dataset.row_filter:
            cache_name += f"_{Path(str(dataset.row_filter['row_index'])).stem}"
        cache_dir = dataset.store_dir / cache_name
    manifest_path = cache_dir / "manifest.json"
    weights = _anchor_region_weights(mode, names, counts, sampling_cfg)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if bool(manifest.get("complete")) and manifest.get("mode") == mode and manifest.get("names") == names:
            arrays = [np.load(cache_dir / item["path"], mmap_mode="r") for item in manifest["groups"]]
            cached_weights = np.asarray(manifest.get("weights", weights.tolist()), dtype=np.float64)
            print(
                f"[ANCHOR-REGION] reuse cache={cache_dir} "
                + " ".join(f"{g['name']}:{g['n']:,}" for g in manifest["groups"]),
                flush=True,
            )
            return arrays, cached_weights / cached_weights.sum(), names, manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_manifest = {
        "complete": False,
        "mode": mode,
        "seed": int(seed),
        "source_global_index": source_path,
        "names": names,
        "counts": counts.astype(int).tolist(),
        "weights": weights.tolist(),
        "max_per_component": int(max_per_component),
        "min_per_component": int(min_per_component),
        "chunk_rows": int(chunk_rows),
        "created_at": now_iso(),
    }
    manifest_path.write_text(json.dumps(tmp_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    target_counts = np.minimum(counts, int(max_per_component)).astype(np.int64)
    if min_per_component > 0:
        target_counts = np.minimum(counts, np.maximum(target_counts, int(min_per_component))).astype(np.int64)
    take_probs = target_counts.astype(np.float64) / np.maximum(counts.astype(np.float64), 1.0)
    source = np.load(source_path, mmap_mode="r")
    if source.shape[0] != int(dataset.meta["rows"]):
        raise ValueError(f"source_global_index length mismatch: {source.shape[0]:,} vs {int(dataset.meta['rows']):,}")

    rng = np.random.default_rng(int(seed) + 17017)
    parts: list[list[np.ndarray]] = [[] for _ in names]
    n_rows = len(dataset)
    print(
        f"[ANCHOR-REGION] build mode={mode} cache={cache_dir} rows={n_rows:,} "
        f"targets=" + ",".join(f"{n}:{int(t):,}" for n, t in zip(names, target_counts)),
        flush=True,
    )
    for start in tqdm(range(0, n_rows, chunk_rows), desc="Build anchor regions", ncols=90):
        end = min(n_rows, start + chunk_rows)
        rows = np.arange(start, end, dtype=np.int64)
        source_rows = dataset.source_rows_for_local(rows)
        src = np.asarray(source[source_rows], dtype=np.int64)
        comp = np.searchsorted(boundaries[1:], src, side="right")
        valid = (comp >= 0) & (comp < len(all_names))
        valid[valid] = active_pos[comp[valid]] >= 0
        if not np.any(valid):
            continue
        draw = rng.random(end - start)
        prob = np.zeros(end - start, dtype=np.float64)
        comp_active = np.full(end - start, -1, dtype=np.int64)
        comp_active[valid] = active_pos[comp[valid]]
        prob[valid] = take_probs[comp_active[valid]]
        take = valid & (draw < prob)
        if not np.any(take):
            continue
        take_comp = comp_active[take]
        take_rows = rows[take]
        for comp_i in np.unique(take_comp):
            parts[int(comp_i)].append(take_rows[take_comp == comp_i])

    arrays = []
    groups = []
    for comp_i, name in enumerate(names):
        arr = np.concatenate(parts[comp_i]).astype(np.int64, copy=False) if parts[comp_i] else np.empty(0, dtype=np.int64)
        if len(arr) == 0:
            raise RuntimeError(f"Region-balanced anchor pool is empty for {name}.")
        path = f"{comp_i:02d}_{name}.npy"
        np.save(cache_dir / path, arr)
        arrays.append(np.load(cache_dir / path, mmap_mode="r"))
        groups.append(
            {
                "name": name,
                "path": path,
                "n": int(len(arr)),
                "source_count": int(counts[comp_i]),
                "target_count": int(target_counts[comp_i]),
                "weight": float(weights[comp_i]),
            }
        )
    manifest = {
        **tmp_manifest,
        "complete": True,
        "groups": groups,
        "updated_at": now_iso(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[ANCHOR-REGION] complete cache={cache_dir} "
        + " ".join(f"{g['name']}:{g['n']:,}" for g in groups),
        flush=True,
    )
    return arrays, weights, names, manifest


def build_ae_head(cfg) -> tuple[torch.nn.Module, str]:
    ae_type = str(cfg.model.get("ae_type", "AEProjSH"))
    if ae_type != "AEProjSH":
        raise ValueError("The manuscript release supports model.ae_type=AEProjSH only")
    pos_cfg = cfg.model.get("pos", {})
    world_width_px = int(cfg.sampling.get("world_width_px", 400752))
    resolution_deg = float(pos_cfg.get("resolution_deg", 360.0 / float(world_width_px)))
    return AEProjSH(d_in=64, d_out=int(cfg.model.out_dim), hidden=int(cfg.model.hidden_dim), sh_degree=int(pos_cfg.get("sh_degree", 8)), pos_hidden=int(pos_cfg.get("hidden_dim", 256)), rho_init=float(pos_cfg.get("rho_init", 1e-3)), origin_left=float(pos_cfg.get("origin_left", -180.0)), origin_bottom=float(pos_cfg.get("origin_bottom", -84.0)), resolution_deg=resolution_deg), "AEProjSH"



def build_sampler(cfg, dataset: SpatialSortedFeatureStoreDataset, steps_per_epoch: int, seed: int):
    sampling_type = str(cfg.sampling.get("type", "joint_grid_ring"))
    if sampling_type != "joint_grid_ring":
        raise ValueError(f"Feature-store training currently supports joint_grid_ring only, got {sampling_type!r}")
    anchor_groups = build_anchor_region_groups(cfg, dataset, seed=seed)
    anchor_group_indices = anchor_group_weights = anchor_group_names = None
    if anchor_groups is not None:
        anchor_group_indices, anchor_group_weights, anchor_group_names, _ = anchor_groups
    return JointGridRingBatchSampler(
        dataset.spatial_x,
        dataset.spatial_y,
        batch_size=int(cfg.train.batch_size),
        ring_half_side_px=list(cfg.sampling.ring_half_side_px),
        ring_ratios=list(cfg.sampling.ring_ratios),
        anchor_cell_px=int(cfg.sampling.get("anchor_cell_px", 10000)),
        anchor_alpha=float(cfg.sampling.get("anchor_alpha", 0.5)),
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        world_width_px=int(cfg.sampling.get("world_width_px", 0)) or None,
        oversample_factor=float(cfg.sampling.get("oversample_factor", 2.0)),
        fallback_mode=str(cfg.sampling.get("fallback_mode", "outward_then_global")),
        exclude_center_px=bool(cfg.sampling.get("exclude_center_px", True)),
        max_ring_attempts=int(cfg.sampling.get("max_ring_attempts", 12)),
        max_other_attempts=int(cfg.sampling.get("max_other_attempts", 16)),
        max_topup_attempts=int(cfg.sampling.get("max_topup_attempts", 24)),
        max_exact_topup_scan=int(cfg.sampling.get("max_exact_topup_scan", 5_000_000)),
        ring_sampling_mode=str(cfg.sampling.get("ring_sampling_mode", "draw_retry")),
        max_exact_ring_scan=int(cfg.sampling.get("max_exact_ring_scan", 500_000)),
        anchor_group_indices=anchor_group_indices,
        anchor_group_weights=anchor_group_weights,
        anchor_group_names=anchor_group_names,
        same_anchor_group_other_ratio=float(cfg.sampling.get("same_anchor_group_other_ratio", 0.0)),
        other_sampling_mode=str(cfg.sampling.get("other_sampling_mode", "random")),
        other_cluster_half_side_px=int(cfg.sampling.get("other_cluster_half_side_px", 0)) or None,
        other_clusters=int(cfg.sampling.get("other_clusters", 4)),
    )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    xt_mode = str(cfg.get("loss", {}).get("xt_mode", "one_hot"))
    if xt_mode != "one_hot":
        raise ValueError("The manuscript release supports one-hot text alignment only")
    relation_cfg = cfg.get("loss", {})
    ae_relation_weight = float(relation_cfg.get("ae_relation_weight", 0.0))
    ae_relation_temperature = float(relation_cfg.get("ae_relation_temperature", 0.1))
    ae_relation_sample_size = int(relation_cfg.get("ae_relation_sample_size", 0))
    alignment_weight = float(cfg.model["lambda"])
    ii_weight = float(relation_cfg.get("ii_weight", alignment_weight))
    xt_weight = float(relation_cfg.get("xt_weight", 1.0 - alignment_weight))
    if ae_relation_weight < 0:
        raise ValueError("loss.ae_relation_weight must be non-negative")
    if ae_relation_temperature <= 0:
        raise ValueError("loss.ae_relation_temperature must be positive")
    if ae_relation_sample_size < 0:
        raise ValueError("loss.ae_relation_sample_size must be non-negative")
    if ii_weight < 0 or xt_weight < 0:
        raise ValueError("loss.ii_weight and loss.xt_weight must be non-negative")
    if ii_weight == 0 and xt_weight == 0 and ae_relation_weight == 0:
        raise ValueError("At least one training loss weight must be positive")

    seed = int(getattr(cfg.train, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_type = str(cfg.model.get("ae_type", "AEProjSH"))
    if model_type != "AEProjSH":
        raise ValueError("The manuscript release supports AEProjSH only")
    use_spatial_conditioning = True
    dataset = SpatialSortedFeatureStoreDataset(
        Path(args.store_dir),
        int(cfg.text.emb_dim),
        return_spatial=use_spatial_conditioning,
        include_regions=cfg.get("data", {}).get("include_regions", None),
        exclude_regions=cfg.get("data", {}).get("exclude_regions", None),
        row_filter_path=str(cfg.paths.get("row_filter", "")).strip(),
        row_filter_name=str(cfg.paths.get("row_filter_name", "")).strip(),
        io_read_workers=int(getattr(cfg.train, "io_read_workers", 1)),
    )
    bs = int(cfg.train.batch_size)
    steps_per_epoch = int(getattr(cfg.train, "steps_per_epoch", len(dataset) // bs))
    sampler = build_sampler(cfg, dataset, steps_per_epoch, seed)
    steps_per_epoch = int(len(sampler))
    fetch_chunk_batches = int(getattr(cfg.train, "batch_fetch_chunk", 1))
    prefetch_pipeline_stages = int(
        getattr(cfg.train, "prefetch_pipeline_stages", 1)
    )
    if prefetch_pipeline_stages not in {1, 2}:
        raise ValueError("train.prefetch_pipeline_stages must be 1 or 2")

    device = torch.device(str(cfg.train.device) if torch.cuda.is_available() else "cpu")
    img_head, ae_type = build_ae_head(cfg)
    img_head = img_head.to(device)
    text_type = str(cfg.text.get("projector_type", "TextProj"))
    if text_type != "TextProj":
        raise ValueError("The manuscript release supports the linear TextProj only")
    txt_head = TextProj(d_in=int(cfg.text.emb_dim), d_out=int(cfg.model.out_dim)).to(device)
    opt = torch.optim.AdamW(
        list(img_head.parameters()) + list(txt_head.parameters()),
        lr=float(cfg.model.lr),
        weight_decay=float(cfg.model.weight_decay),
        betas=tuple(cfg.model.betas),
    )
    lr_schedule_epochs = int(getattr(cfg.train, "lr_schedule_epochs", cfg.train.epochs))
    if lr_schedule_epochs <= 0:
        raise ValueError("train.lr_schedule_epochs must be positive")
    if lr_schedule_epochs < int(cfg.train.epochs):
        raise ValueError("train.lr_schedule_epochs cannot be shorter than train.epochs")
    total_steps = lr_schedule_epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(
        opt,
        T_max=max(1, total_steps),
        eta_min=float(cfg.model.lr) * float(getattr(cfg.train, "eta_min_ratio", 0.01)),
    )

    run_name = str(cfg.logging.run_name)
    run_dir = Path(cfg.logging.output_dir) / run_name
    ckpt_dir = run_dir / "ckpts"
    ensure_dir(ckpt_dir)

    start_epoch = 1
    resume_path: Path | None = None
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-ckpt not found: {resume_path}")
    elif args.resume_auto and (ckpt_dir / "last.pth").exists():
        resume_path = ckpt_dir / "last.pth"
    if resume_path is not None:
        ckpt = torch.load(str(resume_path), map_location=device)
        img_head.load_state_dict(ckpt["ae_state_dict"])
        txt_head.load_state_dict(ckpt["txt_state_dict"])
        if not args.resume_model_only and "opt_state_dict" in ckpt:
            opt.load_state_dict(ckpt["opt_state_dict"])
        if not args.resume_model_only and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if args.resume_model_only:
            start_epoch = 1
            print(f"[RESUME-MODEL-ONLY] checkpoint={resume_path} restart_epochs=1..{int(cfg.train.epochs)}", flush=True)
        else:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[RESUME] checkpoint={resume_path} start_epoch={start_epoch}", flush=True)

    meta_path = run_dir / "joint_run_inputs.json"
    current_resume = {
        "checkpoint": str(resume_path) if resume_path is not None else "",
        "model_only": bool(args.resume_model_only),
    }
    initial_resume = current_resume
    resume_history: list[dict] = []
    if meta_path.is_file():
        try:
            previous_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            initial_resume = previous_meta.get(
                "initial_resume", previous_meta.get("resume", current_resume)
            )
            resume_history = list(previous_meta.get("resume_history", []))
        except (OSError, ValueError, TypeError):
            pass
    if resume_path is not None:
        resume_history.append({**current_resume, "resumed_at": now_iso()})

    meta = {
        "run_dir": str(run_dir),
        "feature_store": dataset.summary(),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": bs,
        "fetch_chunk_batches": fetch_chunk_batches,
        "prefetch_pipeline_stages": prefetch_pipeline_stages,
        "sampling_type": str(cfg.sampling.get("type", "joint_grid_ring")),
        "model_types": {"ae_type": ae_type, "text_type": text_type},
        "position_conditioning_enabled": position_enabled,
        "loss": {
            "xt_mode": xt_mode,
            "ii_weight": ii_weight,
            "xt_weight": xt_weight,
            "ae_relation_weight": ae_relation_weight,
            "ae_relation_temperature": ae_relation_temperature,
            "ae_relation_sample_size": ae_relation_sample_size,
        },
        "initial_resume": initial_resume,
        "resume": current_resume,
        "resume_history": resume_history,
        "hyper_parameters": OmegaConf.to_container(cfg, resolve=True),
        "lr_schedule_epochs": lr_schedule_epochs,
        "updated_at": now_iso(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[RUN] {run_dir} rows={len(dataset):,} batch_size={bs} "
        f"steps_per_epoch={steps_per_epoch:,} batch_fetch_chunk={fetch_chunk_batches}",
        flush=True,
    )

    max_epochs = int(cfg.train.epochs)
    if start_epoch > max_epochs:
        print(f"[DONE] already reached epoch {max_epochs}", flush=True)
        return

    tau_ii = max(float(cfg.model.tau_img), 1e-8)
    tau_xt = max(float(cfg.model.tau_xt), 1e-8)
    scale_ii = float(np.clip(1.0 / tau_ii, 1.0, 100.0))
    scale_xt = float(np.clip(1.0 / tau_xt, 1.0, 100.0))
    for epoch in range(start_epoch, max_epochs + 1):
        sampler.set_epoch(epoch)
        img_head.train()
        txt_head.train()
        loss_sum = xt_sum = ii_sum = relation_sum = 0.0
        if bool(getattr(cfg.train, "prefetch_chunks", False)):
            if prefetch_pipeline_stages == 2:
                batch_iter = iter_pipelined_dataset_batches(
                    dataset, sampler, fetch_chunk_batches
                )
            else:
                batch_iter = iter_prefetched_dataset_batches(
                    dataset, sampler, fetch_chunk_batches
                )
        else:
            batch_iter = iter_dataset_batches(dataset, sampler, fetch_chunk_batches)
        for batch in tqdm(batch_iter, total=steps_per_epoch, desc=f"Epoch {epoch} [store]", ncols=90):
            a64, a64_aug, tvec, sp_x, sp_y = batch
            spatial_xy = torch.stack((sp_x, sp_y), dim=-1).to(device, non_blocking=True)
            a64 = a64.to(device, non_blocking=True)
            if ii_weight:
                a64_aug = a64_aug.to(device, non_blocking=True)
            tvec = tvec.to(device, non_blocking=True)

            pos_hidden = img_head.encode_position(spatial_xy=spatial_xy, dtype=a64.dtype, device=device)
            ae1 = F.normalize(img_head(a64, pos_hidden=pos_hidden), dim=-1)
            if ii_weight:
                ae2 = F.normalize(img_head(a64_aug, pos_hidden=pos_hidden), dim=-1)
            target = torch.arange(ae1.size(0), device=device)
            l_ii = torch.zeros((), dtype=ae1.dtype, device=device)
            if ii_weight:
                l_ii = 0.5 * F.cross_entropy((ae2 @ ae1.t()) * scale_ii, target)
                l_ii = l_ii + 0.5 * F.cross_entropy((ae1 @ ae2.t()) * scale_ii, target)

            t1 = F.normalize(txt_head(tvec), dim=-1)
            logits_at = (ae1 @ t1.t()) * scale_xt
            logits_ta = (t1 @ ae1.t()) * scale_xt
            l_xt = 0.5 * F.cross_entropy(logits_at, target)
            l_xt = l_xt + 0.5 * F.cross_entropy(logits_ta, target)
            l_ae_relation = torch.zeros((), dtype=ae1.dtype, device=device)
            if ae_relation_weight:
                l_ae_relation = relational_preservation_loss(
                    ae1,
                    a64,
                    temperature=ae_relation_temperature,
                    sample_size=ae_relation_sample_size,
                )
            loss = (
                ii_weight * l_ii
                + xt_weight * l_xt
                + ae_relation_weight * l_ae_relation
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            loss_sum += float(loss.item())
            xt_sum += float(l_xt.item())
            ii_sum += float(l_ii.item())
            relation_sum += float(l_ae_relation.item())

        metrics = {
            "loss": loss_sum / max(1, steps_per_epoch),
            "L_xt": xt_sum / max(1, steps_per_epoch),
            "L_ii": ii_sum / max(1, steps_per_epoch),
            "L_ae_relation": relation_sum / max(1, steps_per_epoch),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        if hasattr(img_head, "rho"):
            metrics["rho"] = float(img_head.rho.detach().cpu().item())
        print(
            f"Epoch {epoch:03d} | loss {metrics['loss']:.4f} | "
            f"L_xt {metrics['L_xt']:.4f} | L_ii {metrics['L_ii']:.4f} | "
            f"L_ae_rel {metrics['L_ae_relation']:.4f} | lr {metrics['lr']:.2e}"
            + (f" | rho {metrics['rho']:.4f}" if "rho" in metrics else ""),
            flush=True,
        )
        payload = {
            "ae_state_dict": img_head.state_dict(),
            "txt_state_dict": txt_head.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "train_metrics": metrics,
            "temps": {"tau_xt": tau_xt, "tau_ii": tau_ii},
            "loss": {
                "xt_mode": xt_mode,
                "ii_weight": ii_weight,
                "xt_weight": xt_weight,
                "ae_relation_weight": ae_relation_weight,
                "ae_relation_temperature": ae_relation_temperature,
                "ae_relation_sample_size": ae_relation_sample_size,
            },
            "model_types": {"ae_type": ae_type, "text_type": text_type},
            "hyper_parameters": OmegaConf.to_container(cfg, resolve=True),
            "feature_store": str(dataset.meta_path),
            "created_at": now_iso(),
        }
        if bool(cfg.logging.save_last):
            torch.save(payload, ckpt_dir / "last.pth")
        if int(cfg.logging.save_every) > 0 and epoch % int(cfg.logging.save_every) == 0:
            torch.save(payload, ckpt_dir / f"epoch_{epoch:04d}.pth")
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "last_epoch": int(epoch),
                    "paths": {"ckpt_dir": str(ckpt_dir), "last": str(ckpt_dir / "last.pth")},
                    "feature_store": str(dataset.meta_path),
                    "updated_at": now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"[DONE] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
