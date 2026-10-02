"""Spatial ring sampler used by PURE manuscript training."""

from __future__ import annotations

import math

import numpy as np
from torch.utils.data import Sampler


class SpatialCellIndex:
    """Sparse sorted cell index for O(nearby cells) rectangular sampling."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cell_size: int,
        world_width_px: int | None = None,
        valid_indices: np.ndarray | None = None,
    ):
        self.x = x.astype(np.int32, copy=False)
        self.y = y.astype(np.int32, copy=False)
        self.cell_size = int(cell_size)
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        valid = (self.x >= 0) & (self.y >= 0)
        if valid_indices is None:
            self.valid_indices = np.flatnonzero(valid).astype(np.int64)
        else:
            idx = np.asarray(valid_indices, dtype=np.int64)
            in_bounds = (idx >= 0) & (idx < len(self.x))
            if not np.all(in_bounds):
                idx = idx[in_bounds]
            idx = idx[valid[idx]]
            self.valid_indices = idx.astype(np.int64, copy=False)
        if len(self.valid_indices) == 0:
            raise ValueError("No valid spatial indices for grid-ring sampler.")
        self.world_width_px = int(world_width_px or (int(self.x[valid].max()) + 1))
        self.cell_width = int(math.ceil(float(self.world_width_px) / float(self.cell_size)))

        cx = self.x[self.valid_indices].astype(np.int64) // self.cell_size
        cy = self.y[self.valid_indices].astype(np.int64) // self.cell_size
        cell_id = cy * np.int64(self.cell_width) + cx
        order_pos = np.argsort(cell_id, kind="mergesort")
        self.order = self.valid_indices[order_pos]
        sorted_cell = cell_id[order_pos]
        self.cell_ids, self.starts, self.counts = np.unique(sorted_cell, return_index=True, return_counts=True)
        self.counts_f = self.counts.astype(np.float64)

    def _cell_ranges_for_rect(self, center_x: int, center_y: int, radius: int) -> tuple[np.ndarray, np.ndarray] | None:
        r = int(radius)
        x0 = int(center_x) - r
        x1 = int(center_x) + r
        y0 = max(0, int(center_y) - r)
        y1 = int(center_y) + r
        if y1 < 0:
            return None
        cy0 = max(0, y0 // self.cell_size)
        cy1 = max(0, y1 // self.cell_size)
        cx0 = math.floor(x0 / self.cell_size)
        cx1 = math.floor(x1 / self.cell_size)
        cell_x_width = self.cell_width

        ids = []
        for cy in range(cy0, cy1 + 1):
            for cx_raw in range(cx0, cx1 + 1):
                cx = cx_raw % cell_x_width
                ids.append(int(cy) * cell_x_width + int(cx))
        if not ids:
            return None
        ids_arr = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.cell_ids, ids_arr)
        found = (pos < len(self.cell_ids)) & (self.cell_ids[np.clip(pos, 0, max(0, len(self.cell_ids) - 1))] == ids_arr)
        if not np.any(found):
            return None
        pos = pos[found]
        return self.starts[pos], self.counts[pos]

    def sample_rect(self, rng: np.random.Generator, center_x: int, center_y: int, radius: int, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        ranges = self._cell_ranges_for_rect(center_x, center_y, radius)
        if ranges is None:
            return np.empty(0, dtype=np.int64)
        starts, counts = ranges
        weights = counts.astype(np.float64)
        weights = weights / weights.sum()
        chosen = rng.choice(len(starts), size=int(n), replace=True, p=weights)
        offsets = (rng.random(int(n)) * counts[chosen]).astype(np.int64)
        return self.order[starts[chosen] + offsets].astype(np.int64, copy=False)

    def sample_anchor(self, rng: np.random.Generator, weights: np.ndarray) -> int:
        chosen = int(rng.choice(len(self.starts), p=weights))
        offset = int(rng.integers(0, int(self.counts[chosen])))
        return int(self.order[int(self.starts[chosen]) + offset])


class JointGridRingBatchSampler(Sampler[list[int]]):
    """Sample batches from square spatial windows/rings on 100m-equivalent POI indices."""

    def __init__(
        self,
        spatial_x: np.ndarray,
        spatial_y: np.ndarray,
        batch_size: int,
        ring_half_side_px: list[int],
        ring_ratios: list[float],
        *,
        anchor_cell_px: int = 10000,
        anchor_alpha: float = 0.5,
        steps_per_epoch: int | None = None,
        seed: int = 42,
        world_width_px: int | None = None,
        oversample_factor: float = 2.0,
        fallback_mode: str = "outward_then_global",
        exclude_center_px: bool = True,
        max_ring_attempts: int = 12,
        max_other_attempts: int = 16,
        max_topup_attempts: int = 24,
        max_exact_topup_scan: int = 5_000_000,
        ring_sampling_mode: str = "draw_retry",
        max_exact_ring_scan: int = 500_000,
        anchor_group_indices: list[np.ndarray] | None = None,
        anchor_group_weights: list[float] | np.ndarray | None = None,
        anchor_group_names: list[str] | None = None,
        same_anchor_group_other_ratio: float = 0.0,
        other_sampling_mode: str = "random",
        other_cluster_half_side_px: int | None = None,
        other_clusters: int = 4,
    ):
        if len(ring_half_side_px) < 1:
            raise ValueError("ring_half_side_px must contain at least one radius.")
        if len(ring_ratios) != len(ring_half_side_px) + 1:
            raise ValueError("ring_ratios must have len(ring_half_side_px) + 1 entries, including others.")
        self.x = np.asarray(spatial_x, dtype=np.int32)
        self.y = np.asarray(spatial_y, dtype=np.int32)
        if self.x.shape != self.y.shape:
            raise ValueError(f"spatial x/y shape mismatch: {self.x.shape} vs {self.y.shape}")
        self.valid = (self.x >= 0) & (self.y >= 0)
        self.all_indices = np.flatnonzero(self.valid).astype(np.int64)
        if len(self.all_indices) == 0:
            raise ValueError("No valid spatial rows for joint_grid_ring sampling.")
        self.batch_size = int(batch_size)
        if self.batch_size > len(self.all_indices):
            raise ValueError(
                f"joint_grid_ring requires unique batch rows, but batch_size={self.batch_size:,} "
                f"> n_valid={len(self.all_indices):,}"
            )
        self.radii = [int(x) for x in ring_half_side_px]
        if any(r <= 0 for r in self.radii) or any(b <= a for a, b in zip(self.radii, self.radii[1:])):
            raise ValueError(f"ring_half_side_px must be strictly increasing positive integers: {self.radii}")
        ratios = np.asarray(ring_ratios, dtype=np.float64)
        if np.any(ratios < 0) or ratios.sum() <= 0:
            raise ValueError(f"Invalid ring ratios: {ring_ratios}")
        ratios = ratios / ratios.sum()
        counts = np.floor(ratios * self.batch_size).astype(np.int64)
        remainder = self.batch_size - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(ratios * self.batch_size - counts))
            counts[order[:remainder]] += 1
        self.bin_counts = counts.astype(int).tolist()
        self.steps_per_epoch = int(steps_per_epoch or (len(self.all_indices) // self.batch_size))
        self.seed = int(seed)
        self.epoch = 0
        self.world_width_px = int(world_width_px or (int(self.x[self.valid].max()) + 1))
        self.oversample_factor = max(1.0, float(oversample_factor))
        self.fallback_mode = str(fallback_mode)
        if self.fallback_mode != "outward_then_global":
            raise ValueError(f"Unsupported grid-ring fallback_mode={self.fallback_mode!r}")
        self.exclude_center_px = bool(exclude_center_px)
        self.inner_bounds = ([0] if self.exclude_center_px else [-1]) + self.radii[:-1]
        self.max_ring_attempts = max(1, int(max_ring_attempts))
        self.max_other_attempts = max(1, int(max_other_attempts))
        self.max_topup_attempts = max(1, int(max_topup_attempts))
        self.max_exact_topup_scan = max(0, int(max_exact_topup_scan))
        self.ring_sampling_mode = str(ring_sampling_mode)
        if self.ring_sampling_mode not in {"draw_retry", "exact_pool"}:
            raise ValueError(f"Unsupported ring_sampling_mode={ring_sampling_mode!r}")
        self.max_exact_ring_scan = max(0, int(max_exact_ring_scan))
        self.same_anchor_group_other_ratio = float(same_anchor_group_other_ratio)
        if not (0.0 <= self.same_anchor_group_other_ratio <= 1.0):
            raise ValueError(
                f"same_anchor_group_other_ratio must be in [0, 1], got {same_anchor_group_other_ratio}"
            )
        self.other_sampling_mode = str(other_sampling_mode)
        if self.other_sampling_mode not in {"random", "clustered"}:
            raise ValueError(f"Unsupported other_sampling_mode={other_sampling_mode!r}")
        self.other_cluster_half_side_px = int(other_cluster_half_side_px or self.radii[-1])
        if self.other_cluster_half_side_px <= 0:
            raise ValueError("other_cluster_half_side_px must be positive.")
        self.other_clusters = max(1, int(other_clusters))
        self._fallback_events: dict[str, int] = {}

        self.ring_indexes = {r: SpatialCellIndex(self.x, self.y, r, self.world_width_px) for r in self.radii}
        self.anchor_group_indexes: list[SpatialCellIndex] = []
        self.anchor_group_cell_weights: list[np.ndarray] = []
        self.anchor_group_pools: list[np.ndarray] = []
        self.anchor_group_names: list[str] = []
        self.anchor_group_p: np.ndarray | None = None
        if anchor_group_indices:
            raw_group_weights = (
                np.ones(len(anchor_group_indices), dtype=np.float64)
                if anchor_group_weights is None
                else np.asarray(anchor_group_weights, dtype=np.float64)
            )
            if raw_group_weights.shape[0] != len(anchor_group_indices):
                raise ValueError(
                    f"anchor_group_weights length mismatch: {raw_group_weights.shape[0]} vs {len(anchor_group_indices)}"
                )
            raw_group_names = anchor_group_names or [f"group_{i}" for i in range(len(anchor_group_indices))]
            if len(raw_group_names) != len(anchor_group_indices):
                raise ValueError(f"anchor_group_names length mismatch: {len(raw_group_names)} vs {len(anchor_group_indices)}")

            kept_weights = []
            kept_summaries = []
            for group_i, group_indices in enumerate(anchor_group_indices):
                group_idx = np.asarray(group_indices, dtype=np.int64)
                if len(group_idx) == 0:
                    continue
                group_index = SpatialCellIndex(
                    self.x,
                    self.y,
                    int(anchor_cell_px),
                    self.world_width_px,
                    valid_indices=group_idx,
                )
                cell_counts = np.power(np.maximum(group_index.counts_f, 1.0), float(anchor_alpha))
                self.anchor_group_indexes.append(group_index)
                self.anchor_group_cell_weights.append(cell_counts / cell_counts.sum())
                self.anchor_group_pools.append(group_index.valid_indices)
                self.anchor_group_names.append(str(raw_group_names[group_i]))
                kept_weights.append(float(raw_group_weights[group_i]))
                kept_summaries.append(f"{raw_group_names[group_i]}:{len(group_index.valid_indices):,}")
            if not self.anchor_group_indexes:
                raise ValueError("All anchor_group_indices are empty after spatial validity filtering.")
            weights = np.asarray(kept_weights, dtype=np.float64)
            if np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError(f"Invalid anchor_group_weights: {anchor_group_weights}")
            self.anchor_group_p = weights / weights.sum()
            self.anchor_index = None
            self.anchor_weights = None
            print(
                "[GRID-RING-ANCHOR-GROUPS] "
                + " ".join(kept_summaries)
                + " weights="
                + ",".join(f"{n}:{w:.4f}" for n, w in zip(self.anchor_group_names, self.anchor_group_p)),
                flush=True,
            )
        else:
            self.anchor_index = SpatialCellIndex(self.x, self.y, int(anchor_cell_px), self.world_width_px)
            anchor_counts = np.power(np.maximum(self.anchor_index.counts_f, 1.0), float(anchor_alpha))
            self.anchor_weights = anchor_counts / anchor_counts.sum()

        print(
            f"[GRID-RING] n_valid={len(self.all_indices):,}/{len(self.x):,} "
            f"half_side_px={self.radii} bin_counts={self.bin_counts} anchor_cell_px={int(anchor_cell_px)} "
            f"exclude_center_px={self.exclude_center_px} fallback={self.fallback_mode} "
            f"same_anchor_group_other_ratio={self.same_anchor_group_other_ratio:.3f} "
            f"other_sampling={self.other_sampling_mode}"
            + (
                f"(half_side_px={self.other_cluster_half_side_px},clusters={self.other_clusters}) "
                if self.other_sampling_mode == "clustered"
                else " "
            )
            + f"ring_sampling={self.ring_sampling_mode} "
            + f"attempts=ring:{self.max_ring_attempts}/other:{self.max_other_attempts}/topup:{self.max_topup_attempts}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_anchor(self, rng: np.random.Generator) -> tuple[int, int | None]:
        if self.anchor_group_indexes:
            assert self.anchor_group_p is not None
            group_i = int(rng.choice(len(self.anchor_group_indexes), p=self.anchor_group_p))
            anchor_idx = self.anchor_group_indexes[group_i].sample_anchor(rng, self.anchor_group_cell_weights[group_i])
            return anchor_idx, group_i
        return self.anchor_index.sample_anchor(rng, self.anchor_weights), None

    def _dx_abs(self, idx: np.ndarray, center_x: int) -> np.ndarray:
        dx = np.abs(self.x[idx].astype(np.int64) - int(center_x))
        if self.world_width_px > 0:
            dx = np.minimum(dx, int(self.world_width_px) - dx)
        return dx

    def _ring_filter(self, idx: np.ndarray, center_x: int, center_y: int, inner: int, outer: int) -> np.ndarray:
        if len(idx) == 0:
            return idx
        dx = self._dx_abs(idx, center_x)
        dy = np.abs(self.y[idx].astype(np.int64) - int(center_y))
        d = np.maximum(dx, dy)
        keep = (d > int(inner)) & (d <= int(outer))
        return idx[keep]

    def _note_fallback(self, key: str, n: int) -> None:
        if n > 0:
            self._fallback_events[key] = self._fallback_events.get(key, 0) + int(n)

    @staticmethod
    def _take_unique(cand: np.ndarray, used: set[int], n: int) -> np.ndarray:
        """Take up to n candidates that are not already present in the batch.

        The one-hot contrastive loss assumes each row appears once in a batch.
        Repeated global indices create false negatives against their own
        duplicated positives, so grid-ring batches must be unique.
        """
        if n <= 0 or len(cand) == 0:
            return np.empty(0, dtype=np.int64)
        out: list[int] = []
        for v in cand:
            iv = int(v)
            if iv in used:
                continue
            used.add(iv)
            out.append(iv)
            if len(out) >= int(n):
                break
        return np.asarray(out, dtype=np.int64)

    def _sample_ring_unique(
        self,
        rng: np.random.Generator,
        anchor_idx: int,
        inner: int,
        outer: int,
        n: int,
        used: set[int],
    ) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if self.ring_sampling_mode == "exact_pool":
            exact = self._sample_ring_unique_exact_pool(rng, anchor_idx, inner, outer, n, used)
            if exact is not None:
                return exact
        center_x = int(self.x[anchor_idx])
        center_y = int(self.y[anchor_idx])
        index = self.ring_indexes[int(outer)]
        out_parts: list[np.ndarray] = []
        remaining = int(n)
        draw_factor = self.oversample_factor
        for _ in range(self.max_ring_attempts):
            draw = int(math.ceil(remaining * draw_factor)) + 64
            draw = max(draw, remaining + 32)
            cand = index.sample_rect(rng, center_x, center_y, int(outer), draw)
            cand = self._ring_filter(cand, center_x, center_y, int(inner), int(outer))
            if len(cand) == 0:
                draw_factor *= 1.5
                continue
            part = self._take_unique(cand, used, remaining)
            if len(part) == 0:
                draw_factor *= 1.5
                continue
            out_parts.append(part)
            remaining -= int(len(part))
            if remaining <= 0:
                break
            draw_factor *= 1.25
        if out_parts:
            return np.concatenate(out_parts).astype(np.int64, copy=False)
        return np.empty(0, dtype=np.int64)

    def _sample_ring_unique_exact_pool(
        self,
        rng: np.random.Generator,
        anchor_idx: int,
        inner: int,
        outer: int,
        n: int,
        used: set[int],
    ) -> np.ndarray | None:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        center_x = int(self.x[anchor_idx])
        center_y = int(self.y[anchor_idx])
        index = self.ring_indexes[int(outer)]
        ranges = index._cell_ranges_for_rect(center_x, center_y, int(outer))
        if ranges is None:
            return np.empty(0, dtype=np.int64)
        starts, counts = ranges
        total = int(np.sum(counts, dtype=np.int64))
        if total <= 0:
            return np.empty(0, dtype=np.int64)
        if self.max_exact_ring_scan > 0 and total > self.max_exact_ring_scan:
            return None
        cand = np.concatenate([index.order[int(s) : int(s) + int(c)] for s, c in zip(starts, counts)])
        cand = self._ring_filter(cand.astype(np.int64, copy=False), center_x, center_y, int(inner), int(outer))
        if len(cand) == 0:
            return np.empty(0, dtype=np.int64)
        rng.shuffle(cand)
        return self._take_unique(cand, used, int(n))

    def _sample_other_unique(
        self,
        rng: np.random.Generator,
        anchor_idx: int,
        excluded_radius: int,
        n: int,
        used: set[int],
        pool_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        pool = self.all_indices if pool_indices is None else pool_indices
        if len(pool) == 0:
            return np.empty(0, dtype=np.int64)
        center_x = int(self.x[anchor_idx])
        center_y = int(self.y[anchor_idx])
        out_parts: list[np.ndarray] = []
        remaining = int(n)
        draw_factor = max(2.0, self.oversample_factor)
        for _ in range(self.max_other_attempts):
            draw = int(math.ceil(remaining * draw_factor)) + 128
            cand = rng.choice(pool, size=draw, replace=len(pool) < draw).astype(np.int64)
            dx = self._dx_abs(cand, center_x)
            dy = np.abs(self.y[cand].astype(np.int64) - center_y)
            cand = cand[np.maximum(dx, dy) > int(excluded_radius)]
            if len(cand) == 0:
                draw_factor *= 1.5
                continue
            part = self._take_unique(cand, used, remaining)
            if len(part) == 0:
                draw_factor *= 1.5
                continue
            out_parts.append(part)
            remaining -= int(len(part))
            if remaining <= 0:
                break
            draw_factor *= 1.25
        if out_parts:
            return np.concatenate(out_parts).astype(np.int64, copy=False)
        return np.empty(0, dtype=np.int64)

    def _sample_other_clustered_unique(
        self,
        rng: np.random.Generator,
        anchor_idx: int,
        excluded_radius: int,
        n: int,
        used: set[int],
        *,
        spatial_index: SpatialCellIndex,
        center_pool: np.ndarray | None = None,
    ) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        pool = self.all_indices if center_pool is None else center_pool
        if len(pool) == 0:
            return np.empty(0, dtype=np.int64)
        center_x = int(self.x[anchor_idx])
        center_y = int(self.y[anchor_idx])
        out_parts: list[np.ndarray] = []
        remaining = int(n)
        clusters_left = self.other_clusters
        for _ in range(self.max_other_attempts):
            if remaining <= 0:
                break
            center_draw = max(16, clusters_left * 4)
            centers = rng.choice(pool, size=center_draw, replace=len(pool) < center_draw).astype(np.int64)
            dx = self._dx_abs(centers, center_x)
            dy = np.abs(self.y[centers].astype(np.int64) - center_y)
            centers = centers[np.maximum(dx, dy) > int(excluded_radius)]
            if len(centers) == 0:
                clusters_left = min(clusters_left * 2, self.other_clusters * 8)
                continue
            for center in centers[:clusters_left]:
                per_cluster = int(math.ceil(remaining / max(1, clusters_left)))
                draw = int(math.ceil(per_cluster * self.oversample_factor)) + 64
                cand = spatial_index.sample_rect(
                    rng,
                    int(self.x[int(center)]),
                    int(self.y[int(center)]),
                    self.other_cluster_half_side_px,
                    draw,
                )
                if len(cand) == 0:
                    continue
                dx = self._dx_abs(cand, center_x)
                dy = np.abs(self.y[cand].astype(np.int64) - center_y)
                cand = cand[np.maximum(dx, dy) > int(excluded_radius)]
                part = self._take_unique(cand, used, remaining)
                if len(part) == 0:
                    continue
                out_parts.append(part)
                remaining -= int(len(part))
                if remaining <= 0:
                    break
            clusters_left = min(clusters_left * 2, self.other_clusters * 8)
        if out_parts:
            return np.concatenate(out_parts).astype(np.int64, copy=False)
        return np.empty(0, dtype=np.int64)

    def _sample_other_mixed_unique(
        self,
        rng: np.random.Generator,
        anchor_idx: int,
        anchor_group_i: int | None,
        excluded_radius: int,
        n: int,
        used: set[int],
    ) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if self.other_sampling_mode == "clustered":
            return self._sample_other_mixed_clustered_unique(
                rng,
                anchor_idx,
                anchor_group_i,
                excluded_radius,
                n,
                used,
            )
        same_ratio = self.same_anchor_group_other_ratio
        if same_ratio <= 0.0 or anchor_group_i is None or not self.anchor_group_pools:
            return self._sample_other_unique(rng, anchor_idx, excluded_radius, n, used)

        out_parts: list[np.ndarray] = []
        same_n = int(round(float(n) * same_ratio))
        same_n = min(max(0, same_n), int(n))
        remaining = int(n)
        if same_n > 0:
            same_pool = self.anchor_group_pools[int(anchor_group_i)]
            part = self._sample_other_unique(
                rng,
                anchor_idx,
                excluded_radius,
                same_n,
                used,
                pool_indices=same_pool,
            )
            if len(part) > 0:
                out_parts.append(part)
                remaining -= int(len(part))
        if remaining > 0:
            part = self._sample_other_unique(rng, anchor_idx, excluded_radius, remaining, used)
            if len(part) > 0:
                out_parts.append(part)
        if not out_parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(out_parts).astype(np.int64, copy=False)

    def _sample_other_mixed_clustered_unique(
        self,
        rng: np.random.Generator,
        anchor_idx: int,
        anchor_group_i: int | None,
        excluded_radius: int,
        n: int,
        used: set[int],
    ) -> np.ndarray:
        same_ratio = self.same_anchor_group_other_ratio
        global_index = self.ring_indexes[self.radii[-1]]
        if same_ratio <= 0.0 or anchor_group_i is None or not self.anchor_group_indexes:
            part = self._sample_other_clustered_unique(
                rng,
                anchor_idx,
                excluded_radius,
                n,
                used,
                spatial_index=global_index,
            )
            if len(part) < int(n):
                tail = self._sample_other_unique(rng, anchor_idx, excluded_radius, int(n) - len(part), used)
                if len(tail) > 0:
                    part = np.concatenate([part, tail]).astype(np.int64, copy=False)
            return part

        out_parts: list[np.ndarray] = []
        same_n = min(max(0, int(round(float(n) * same_ratio))), int(n))
        remaining = int(n)
        if same_n > 0:
            same_part = self._sample_other_clustered_unique(
                rng,
                anchor_idx,
                excluded_radius,
                same_n,
                used,
                spatial_index=self.anchor_group_indexes[int(anchor_group_i)],
                center_pool=self.anchor_group_pools[int(anchor_group_i)],
            )
            if len(same_part) > 0:
                out_parts.append(same_part)
                remaining -= int(len(same_part))
        if remaining > 0:
            global_part = self._sample_other_clustered_unique(
                rng,
                anchor_idx,
                excluded_radius,
                remaining,
                used,
                spatial_index=global_index,
            )
            if len(global_part) > 0:
                out_parts.append(global_part)
                remaining -= int(len(global_part))
        if remaining > 0:
            tail = self._sample_other_unique(rng, anchor_idx, excluded_radius, remaining, used)
            if len(tail) > 0:
                out_parts.append(tail)
        if not out_parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(out_parts).astype(np.int64, copy=False)

    def _sample_any(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(self.all_indices, size=int(n), replace=len(self.all_indices) < int(n)).astype(np.int64)

    def _sample_any_unique(self, rng: np.random.Generator, n: int, used: set[int]) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        out_parts: list[np.ndarray] = []
        remaining = int(n)
        draw_factor = 3.0
        for _ in range(self.max_topup_attempts):
            draw = int(math.ceil(remaining * draw_factor)) + 128
            part = self._take_unique(self._sample_any(rng, draw), used, remaining)
            if len(part) > 0:
                out_parts.append(part)
                remaining -= int(len(part))
            if remaining <= 0:
                break
            draw_factor *= 1.35
        if out_parts:
            return np.concatenate(out_parts).astype(np.int64, copy=False)
        return np.empty(0, dtype=np.int64)

    def _sample_bin_with_fallback(
        self,
        rng: np.random.Generator,
        anchor_idx: int,
        anchor_group_i: int | None,
        bin_i: int,
        n: int,
        used: set[int],
    ) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        out_parts: list[np.ndarray] = []
        remaining = int(n)

        def take(part: np.ndarray, source: str) -> None:
            nonlocal remaining
            if remaining <= 0 or len(part) == 0:
                return
            out_parts.append(part)
            if source != f"bin{bin_i}":
                self._note_fallback(f"bin{bin_i}->{source}", len(part))
            remaining -= int(len(part))

        if bin_i < len(self.radii):
            take(
                self._sample_ring_unique(rng, anchor_idx, self.inner_bounds[bin_i], self.radii[bin_i], remaining, used),
                f"bin{bin_i}",
            )
            for j in range(bin_i + 1, len(self.radii)):
                if remaining <= 0:
                    break
                take(
                    self._sample_ring_unique(rng, anchor_idx, self.inner_bounds[j], self.radii[j], remaining, used),
                    f"bin{j}",
                )
            if remaining > 0:
                take(
                    self._sample_other_mixed_unique(
                        rng, anchor_idx, anchor_group_i, self.radii[-1], remaining, used
                    ),
                    "other",
                )
        else:
            take(
                self._sample_other_mixed_unique(rng, anchor_idx, anchor_group_i, self.radii[-1], remaining, used),
                f"bin{bin_i}",
            )

        if remaining > 0:
            take(self._sample_any_unique(rng, remaining, used), "any")
        if not out_parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(out_parts).astype(np.int64, copy=False)

    def _top_up_unique(self, rng: np.random.Generator, parts: list[np.ndarray], used: set[int]) -> np.ndarray:
        batch = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        remaining = self.batch_size - int(len(batch))
        if remaining <= 0:
            return batch[: self.batch_size].astype(np.int64, copy=False)

        fill_parts: list[np.ndarray] = []
        part = self._sample_any_unique(rng, remaining, used)
        if len(part) > 0:
            fill_parts.append(part)
            remaining -= int(len(part))
        if remaining > 0:
            if len(self.all_indices) <= self.max_exact_topup_scan:
                mask = np.fromiter((int(v) not in used for v in self.all_indices), dtype=bool, count=len(self.all_indices))
                pool = self.all_indices[mask]
                if len(pool) < remaining:
                    raise RuntimeError(f"Unable to build unique grid-ring batch: missing {remaining} rows.")
                part = rng.choice(pool, size=remaining, replace=False).astype(np.int64)
                used.update(int(v) for v in part)
                fill_parts.append(part)
                self._note_fallback("topup->exact_scan", len(part))
                remaining = 0
            else:
                raise RuntimeError(
                    f"Unable to build unique grid-ring batch without scanning {len(self.all_indices):,} rows; "
                    f"missing {remaining} rows. Increase max_topup_attempts/oversample_factor or reduce batch_size."
                )
        if fill_parts:
            batch = np.concatenate([batch, *fill_parts]).astype(np.int64, copy=False)
            self._note_fallback("topup->any", int(sum(len(x) for x in fill_parts)))
        return batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self._fallback_events = {}
        for _ in range(self.steps_per_epoch):
            anchor_idx, anchor_group_i = self._sample_anchor(rng)
            parts = []
            used: set[int] = set()
            for bin_i, n in enumerate(self.bin_counts):
                parts.append(self._sample_bin_with_fallback(rng, anchor_idx, anchor_group_i, bin_i, int(n), used))
            batch = self._top_up_unique(rng, parts, used)
            rng.shuffle(batch)
            yield batch.tolist()
        if self._fallback_events:
            print(f"[GRID-RING-FALLBACK] epoch={self.epoch} {self._fallback_events}", flush=True)


    def iter_with_bin_ids(self):
        """Yield rows together with their requested spatial-scale bin.

        This opt-in interface leaves the normal ``__iter__`` path unchanged.
        Outward-fallback samples retain the requested bin; unplanned top-up
        rows use the final/global bin.
        """
        rng = np.random.default_rng(self.seed + self.epoch)
        self._fallback_events = {}
        global_bin = len(self.bin_counts) - 1
        for _ in range(self.steps_per_epoch):
            anchor_idx, anchor_group_i = self._sample_anchor(rng)
            parts: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            used: set[int] = set()
            for bin_i, n in enumerate(self.bin_counts):
                part = self._sample_bin_with_fallback(
                    rng, anchor_idx, anchor_group_i, bin_i, int(n), used
                )
                parts.append(part)
                labels.append(np.full(len(part), bin_i, dtype=np.int8))

            requested_rows = int(sum(len(part) for part in parts))
            batch = self._top_up_unique(rng, parts, used)
            if len(batch) > requested_rows:
                labels.append(
                    np.full(len(batch) - requested_rows, global_bin, dtype=np.int8)
                )
            bin_ids = np.concatenate(labels).astype(np.int8, copy=False)
            if len(batch) != self.batch_size or len(bin_ids) != self.batch_size:
                raise RuntimeError(
                    f"Labeled grid-ring batch has invalid size: "
                    f"rows={len(batch)} labels={len(bin_ids)} expected={self.batch_size}"
                )
            order = np.arange(self.batch_size)
            rng.shuffle(order)
            yield batch[order].tolist(), bin_ids[order]
        if self._fallback_events:
            print(
                f"[GRID-RING-FALLBACK] epoch={self.epoch} {self._fallback_events}",
                flush=True,
            )


def iter_dataset_batches(dataset, sampler, fetch_chunk_batches: int):
    """Yield training batches, optionally fetching several batches at once.

    Large full-region caches are stored as memmaps. Fetching a single random
    batch at a time causes heavy disk seek overhead. Chunked fetching preserves
    the sampler's batch boundaries while allowing get_batch() to sort a much
    larger index set before reading from disk.
    """
    fetch_chunk_batches = max(1, int(fetch_chunk_batches))
    chunk: list[list[int]] = []

    def flush_chunk(items: list[list[int]]):
        sizes = [len(x) for x in items]
        flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in items])
        flat_batch = dataset.get_batch(flat)
        start = 0
        for size in sizes:
            end = start + size
            yield tuple(x[start:end] for x in flat_batch)
            start = end

    for batch_indices in sampler:
        chunk.append(batch_indices)
        if len(chunk) >= fetch_chunk_batches:
            yield from flush_chunk(chunk)
            chunk = []
    if chunk:
        yield from flush_chunk(chunk)
