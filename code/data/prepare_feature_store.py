#!/usr/bin/env python3
"""Create aligned component links and a build config for 2024 POI text."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


TEXT_FORMULA = "A place of {category_leaf}, a type of {category_top}, named {name}."
EXPECTED_TOTAL = 84_618_817
TEXT_ROOT = Path(os.environ["PURE_TEXT_EMBEDDING_ROOT"])
POI_ROOT = Path(os.environ["PURE_POI_ROOT"])
COMPONENT_ROOT = Path(os.environ["PURE_COMPONENT_ROOT"])
FINAL_STORE = Path(os.environ["PURE_FEATURE_STORE"])
TILE_MANIFEST = Path(os.environ["PURE_AEF_TILE_MANIFEST"])
REGIONS = (
    "china", "africa", "asia_ex_china", "europe", "us",
    "middle_east_central_asia", "latin_america_caribbean", "oceania",
    "europe_non32", "north_america_ex_us",
)


def poi_path(region):
    if region == "china":
        return Path(os.environ["PURE_CHINA_POI"])
    return POI_ROOT / f"{region}_poi_name_quality_ae100m_tiles_spatial_validated.parquet"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main():
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    components = []
    total = 0
    for region in REGIONS:
        poi = poi_path(region).resolve(strict=True)
        rows = int(pq.ParquetFile(poi).metadata.num_rows)
        array_path = (TEXT_ROOT / f"{region}_qwen512.npy").resolve(strict=True)
        meta_path = TEXT_ROOT / f"{region}_qwen512.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_meta = {
            "status": "complete",
            "rows": rows,
            "embedding_dimension": 512,
            "dtype": "float32",
            "model": "Qwen/Qwen3-Embedding-8B",
            "prompt_name": "query",
            "max_seq_length": 128,
            "batch_size": 1024,
            "text_formula": TEXT_FORMULA,
            "preexisting_description_used": False,
            "normalization": "MRL prefix truncation followed by L2 normalization",
            "china_name_extracted_from_description": region == "china",
        }
        mismatch = {key: (meta.get(key), value) for key, value in expected_meta.items() if meta.get(key) != value}
        if mismatch:
            raise ValueError(f"{region} metadata mismatch: {mismatch}")
        expected_software = {
            "torch": "2.11.0+cu130",
            "transformers": "5.9.0",
            "sentence_transformers": "5.5.1",
        }
        software_mismatch = {
            key: (meta.get("software", {}).get(key), value)
            for key, value in expected_software.items()
            if meta.get("software", {}).get(key) != value
        }
        if software_mismatch:
            raise ValueError(f"{region} software mismatch: {software_mismatch}")
        if Path(meta["source"]).name != poi.name:
            raise ValueError(f"{region} source basename mismatch: {meta['source']} vs {poi}")
        if int(meta["source_size_bytes"]) != poi.stat().st_size:
            raise ValueError(f"{region} source size mismatch: {meta['source_size_bytes']} vs {poi.stat().st_size}")
        arr = np.load(array_path, mmap_mode="r")
        if arr.shape != (rows, 512) or arr.dtype != np.float32:
            raise ValueError(f"{region} array mismatch: {arr.shape} {arr.dtype}")
        rng = np.random.default_rng(45)
        idx = np.unique(np.r_[0, rows // 2, rows - 1, rng.integers(0, rows, size=min(10000, rows))])
        sample = np.asarray(arr[idx], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1)
        if not np.isfinite(sample).all() or np.max(np.abs(norms - 1.0)) > 2e-4:
            raise ValueError(f"{region} invalid embedding sample norms [{norms.min()}, {norms.max()}]")
        digest = sha256(array_path)
        target = COMPONENT_ROOT / region / "text_qwen_d512.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != array_path:
                raise FileExistsError(f"Wrong existing link: {target}")
        elif target.exists():
            raise FileExistsError(f"Refusing existing non-link: {target}")
        else:
            target.symlink_to(array_path)
        components.append({"region": region, "poi_sample": str(poi), "cache_dir": str(target.parent)})
        records.append({
            "region": region, "rows": rows, "poi": str(poi), "text": str(array_path),
            "sha256": digest, "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "text_formula": TEXT_FORMULA,
        })
        total += rows
        print(f"[VALID] {region}: {rows:,} sha256={digest}", flush=True)
    if total != EXPECTED_TOTAL:
        raise ValueError(f"Total rows {total:,} != {EXPECTED_TOTAL:,}")
    config = {
        "paths": {"components": components},
        "joint": {"sample_frac_per_component": 1.0, "sample_seed": 45},
        "ae": {"crs": "EPSG:4326", "pix_radius": 0, "aug_pix_radius": 1},
        "text": {"backend": "qwen", "hf_model": "Qwen/Qwen3-Embedding-8B", "emb_dim": 512},
        "sampling": {"world_width_px": 400752},
        "train": {"device": "cuda:0", "batch_size": 2048, "epochs": 10, "seed": 45},
        "model": {"hidden_dim": 2048, "out_dim": 128},
        "loss": {"xt_mode": "one_hot"},
        "logging": {"output_dir": str(COMPONENT_ROOT / "logs"), "run_name": "prepare_leaf_top"},
    }
    config_path = COMPONENT_ROOT / "joint_overture2024_leaf_top_qwen512.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(COMPONENT_ROOT / "component_alignment_audit.json", {
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "poi_release": "2024-12-18.0",
        "rows": total,
        "text_dim": 512,
        "text_formula": TEXT_FORMULA,
        "regions": records,
        "config": str(config_path),
        "tile_manifest": str(TILE_MANIFEST),
        "planned_store": str(FINAL_STORE),
    })
    print(f"[DONE] {total:,} rows; config={config_path}", flush=True)


if __name__ == "__main__":
    main()
