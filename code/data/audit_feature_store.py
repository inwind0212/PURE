#!/usr/bin/env python3
"""Audit the global training feature store."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_ROWS = 84_618_817
EXPECTED_REGIONS = [
    "china", "africa", "asia_ex_china", "europe", "us",
    "middle_east_central_asia", "latin_america_caribbean", "oceania",
    "europe_non32", "north_america_ex_us",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=500000)
    args = parser.parse_args()
    store = args.store.resolve()
    meta = json.loads((store / "feature_store_meta.json").read_text(encoding="utf-8"))
    if not meta.get("complete") or int(meta["rows"]) != EXPECTED_ROWS:
        raise ValueError(f"Incomplete or wrong row count: {meta.get('complete')} {meta.get('rows')}")
    if int(meta["ae_dim"]) != 64 or int(meta["text_dim"]) != 512:
        raise ValueError(f"Wrong dimensions: ae={meta['ae_dim']} text={meta['text_dim']}")
    regions = [str(item["region"]) for item in meta["components"]]
    counts = [int(item.get("n", item.get("n_source", -1))) for item in meta["components"]]
    if regions != EXPECTED_REGIONS or sum(counts) != EXPECTED_ROWS:
        raise ValueError(f"Component mismatch: {regions}, sum={sum(counts)}")
    specs = {
        "ae_base": ((EXPECTED_ROWS, 64), np.float32),
        "ae_aug": ((EXPECTED_ROWS, 64), np.float32),
        "text": ((EXPECTED_ROWS, 512), np.float32),
        "sp_x": ((EXPECTED_ROWS,), np.int32),
        "sp_y": ((EXPECTED_ROWS,), np.int32),
        "hilbert": ((EXPECTED_ROWS,), np.uint64),
        "source_global_index": ((EXPECTED_ROWS,), np.int64),
    }
    arrays = {}
    for key, (shape, dtype) in specs.items():
        arr = np.load(Path(meta["paths"][key]), mmap_mode="r")
        if arr.shape != shape or arr.dtype != np.dtype(dtype):
            raise ValueError(f"{key}: got {arr.shape}/{arr.dtype}, expected {shape}/{np.dtype(dtype)}")
        arrays[key] = arr
    seen = np.zeros(EXPECTED_ROWS, dtype=np.bool_)
    previous = None
    for start in range(0, EXPECTED_ROWS, args.chunk_rows):
        end = min(EXPECTED_ROWS, start + args.chunk_rows)
        x = np.asarray(arrays["sp_x"][start:end])
        y = np.asarray(arrays["sp_y"][start:end])
        h = np.asarray(arrays["hilbert"][start:end])
        source = np.asarray(arrays["source_global_index"][start:end])
        if np.any(x < 0) or np.any(y < 0):
            raise ValueError(f"Negative spatial coordinate at [{start}, {end})")
        if previous is not None and int(h[0]) < previous or np.any(h[1:] < h[:-1]):
            raise ValueError(f"Hilbert order failure at [{start}, {end})")
        previous = int(h[-1])
        if np.any(source < 0) or np.any(source >= EXPECTED_ROWS) or np.any(seen[source]):
            raise ValueError(f"Source permutation failure at [{start}, {end})")
        seen[source] = True
        for key in ("ae_base", "ae_aug", "text"):
            if not np.isfinite(np.asarray(arrays[key][start:end])).all():
                raise ValueError(f"Non-finite {key} at [{start}, {end})")
        if start % 5_000_000 == 0:
            print(f"[AUDIT] {end:,}/{EXPECTED_ROWS:,}", flush=True)
    if not seen.all():
        raise ValueError("Source index is not a full permutation")
    rng = np.random.default_rng(45)
    idx = np.unique(np.r_[0, EXPECTED_ROWS // 2, EXPECTED_ROWS - 1, rng.integers(0, EXPECTED_ROWS, 20000)])
    text_norms = np.linalg.norm(np.asarray(arrays["text"][idx]), axis=1)
    if np.max(np.abs(text_norms - 1.0)) > 2e-4:
        raise ValueError(f"Text norms invalid: [{text_norms.min()}, {text_norms.max()}]")
    report = {
        "passed": True, "created_at": datetime.now(timezone.utc).isoformat(),
        "store": str(store), "rows": EXPECTED_ROWS, "regions": regions,
        "full_finite_scan": ["ae_base", "ae_aug", "text"],
        "source_index_full_permutation": True, "hilbert_monotonic": True,
        "text_norm_min": float(text_norms.min()), "text_norm_max": float(text_norms.max()),
    }
    out = store / "feature_store_audit.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(f"[PASS] {out}", flush=True)


if __name__ == "__main__":
    main()
