#!/usr/bin/env python
"""Infer one AETHER model and fit downstream heads for a transfer city."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CITYREP = str(PROJECT_ROOT / "evaluation" / "build_area_average_cache.py")
TRAIN_HEAD = str(PROJECT_ROOT / "evaluation" / "train_downstream_heads.py")
TARGET_NORM_PROTOCOL = "train_only_per_seed_head_v1"
CACHE_PROTOCOL = "cityrep_target_cell_area_average_l2_v2"
HEAD_EPOCHS = 100
HEAD_PATIENCE = 10
HEAD_HIDDEN = 1024
HEAD_LR = 1e-3
HEAD_BATCH_DEFAULT = 128
HEAD_BATCH_POP = 1024


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source)


def metrics_cover_requested_seeds(
    metrics_path: Path, *, models: list[str], seeds: list[int]
) -> bool:
    """Return true only when every requested global head/seed is present."""
    if not metrics_path.is_file():
        return False
    observed: dict[str, set[int]] = {model: set() for model in models}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model")
            if row.get("method") != "global_head" or model not in observed:
                continue
            observed[model].add(int(row["seed"]))
    expected = set(seeds)
    return all(observed[model] >= expected for model in models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-tag")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--sample-batch", type=int, default=4096)
    parser.add_argument("--aether-input-radius-px", type=int, default=0)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--cache-tag", default="smod13_settlement")
    parser.add_argument("--head-batch-default", type=int, default=HEAD_BATCH_DEFAULT)
    parser.add_argument("--head-batch-pop", type=int, default=HEAD_BATCH_POP)
    parser.add_argument(
        "--result-root-name",
        default="results",
        help="Directory name below downstream used for versioned head results.",
    )
    parser.add_argument(
        "--cache-base",
        type=Path,
        help="Optional NVMe root for derived AE/AETHER feature caches.",
    )
    parser.add_argument("--aether-only", action="store_true")
    parser.add_argument(
        "--ae-source-cache",
        type=Path,
        help="Cache root from an earlier arm whose AlphaEarth files should be reused.",
    )
    args = parser.parse_args()
    if args.head_batch_default < 1 or args.head_batch_pop < 1:
        raise ValueError("Downstream batch sizes must be positive")
    if Path(args.result_root_name).name != args.result_root_name:
        raise ValueError("--result-root-name must be a single directory name")

    manifest = json.loads((args.root / "experiment_manifest.json").read_text(encoding="utf-8"))
    city_record = manifest["cities"][args.city]
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        model_record = city_record["models"][args.arm]
        checkpoint = Path(model_record["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    tasks = args.task or ["gdp", "pop", "ntl", "pm25", "lst"]
    model_tag = args.model_tag or args.arm
    seeds = args.seeds or ",".join(str(v) for v in manifest["downstream"]["seeds"])
    requested_seeds = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    suffix = f"aether128_{model_tag}_{args.cache_tag}_cityrepavg"
    model_name = f"AETHER_{model_tag}"
    city_dir = args.root / "cities" / args.city
    raw_root = city_dir / "downstream" / "raw_samples"
    cache_root = (
        args.cache_base / args.city / model_tag
        if args.cache_base is not None
        else city_dir / "downstream" / "cityrepavg" / model_tag
    )
    result_root = city_dir / "downstream" / args.result_root_name / model_tag
    log_root = args.root / "logs" / "downstream" / args.city / model_tag

    for task in tasks:
        head_batch_size = (
            args.head_batch_pop if task == "pop" else args.head_batch_default
        )
        raw_task = raw_root / task
        if not raw_task.is_dir():
            raise FileNotFoundError(raw_task)
        cache_task = cache_root / task
        result_task = result_root / task
        cache_task.mkdir(parents=True, exist_ok=True)
        result_task.mkdir(parents=True, exist_ok=True)
        log = log_root / f"{task}.log"

        npz_files = sorted((raw_task / "cache").glob("*.npz"))
        if len(npz_files) != 1:
            raise RuntimeError(f"Expected one raw NPZ for {args.city}/{task}, found {len(npz_files)}")
        npz = npz_files[0]
        stem = npz.stem
        aether_cache = cache_task / "cache" / f"{stem}.{suffix}.npy"
        if not aether_cache.is_file():
            sample_batch = 8192 if task == "pop" else args.sample_batch
            inference_models = "aether" if args.aether_only else "ae,aether"
            run(
                [
                    PYTHON,
                    "-u",
                    CITYREP,
                    "--old-task-dir",
                    str(raw_task),
                    "--out-root",
                    str(cache_root),
                    "--models",
                    inference_models,
                    "--ckpt",
                    str(checkpoint),
                    "--aether-suffix",
                    suffix,
                    "--device",
                    args.device,
                    "--sample-batch",
                    str(sample_batch),
                    "--batch-pixels",
                    "65536",
                    "--torch-threads",
                    "8",
                    "--aether-input-radius-px",
                    str(args.aether_input_radius_px),
                ],
                log,
            )
        cache_meta_path = cache_task / "cityrepavg_meta.json"
        if not cache_meta_path.is_file():
            raise RuntimeError(f"Missing cache protocol metadata: {cache_meta_path}")
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        expected_cache_protocol = (
            CACHE_PROTOCOL
            if args.aether_input_radius_px == 0
            else f"{CACHE_PROTOCOL}_aether_input_window_r{args.aether_input_radius_px}"
        )
        if cache_meta.get("protocol") != expected_cache_protocol:
            raise RuntimeError(
                f"Refusing incompatible cache for {args.city}/{task}: "
                f"expected={expected_cache_protocol!r} actual={cache_meta.get('protocol')!r}"
            )

        if args.aether_only:
            if args.ae_source_cache is None:
                raise ValueError("--ae-source-cache is required with --aether-only")
            source_files = sorted((args.ae_source_cache / task / "cache").glob("*.ae64.npy"))
            if len(source_files) != 1:
                raise RuntimeError(
                    f"Expected one shared AlphaEarth cache for {task}, found {len(source_files)}"
                )
            link(source_files[0], cache_task / "cache" / source_files[0].name)

        label_files = sorted(raw_task.glob("*_admin_labels_with_country.parquet"))
        if len(label_files) != 1:
            raise RuntimeError(
                f"Expected one labels parquet for {args.city}/{task}, found {len(label_files)}"
            )
        labels = label_files[0]
        label_stem = labels.name.removesuffix("_admin_labels_with_country.parquet")
        link(npz, cache_task / "cache" / npz.name)
        for ending in [
            "_admin_labels.parquet",
            "_admin_labels_with_country.parquet",
            "_label_meta.json",
        ]:
            source = raw_task / f"{label_stem}{ending}"
            if source.exists():
                link(source, cache_task / source.name)

        metrics = result_task / "metrics.csv"
        meta_path = result_task / "meta.json"
        if metrics.is_file():
            if not meta_path.is_file():
                raise RuntimeError(
                    f"Existing metrics have no protocol metadata: {result_task}. "
                    "Archive the incomplete result directory before resuming."
                )
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            existing_protocol = existing_meta.get("target_norm_protocol")
            if existing_protocol != TARGET_NORM_PROTOCOL:
                raise RuntimeError(
                    f"Refusing to reuse incompatible downstream results in {result_task}: "
                    f"target_norm_protocol={existing_protocol!r}, "
                    f"target_norm={existing_meta.get('target_norm')!r}"
                )
            expected_head = {
                "epochs": HEAD_EPOCHS,
                "patience": HEAD_PATIENCE,
                "batch_size": head_batch_size,
                "lr": HEAD_LR,
                "hidden": HEAD_HIDDEN,
                "head_type": "mlp",
                "split_mode": "spatial_block",
                "spatial_block_km": 1.0,
            }
            mismatches = {
                key: (old_meta.get(key), expected)
                for key, expected in expected_head.items()
                if old_meta.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"Refusing to reuse incompatible downstream head settings in {result_task}: "
                    f"{mismatches}"
                )
        requested_model_names = (
            [model_name] if args.aether_only else ["AlphaEarth64", model_name]
        )
        result_complete = metrics_cover_requested_seeds(
            metrics,
            models=requested_model_names,
            seeds=requested_seeds,
        )
        if not result_complete:
            requested_models = ",".join(requested_model_names)
            run(
                [
                    PYTHON,
                    "-u",
                    TRAIN_HEAD,
                    "--cache-dir",
                    str(cache_task / "cache"),
                    "--labels-parquet",
                    str(cache_task / labels.name),
                    "--group-col",
                    "country_code",
                    "--group-name-col",
                    "country_name",
                    "--out-dir",
                    str(result_task),
                    "--task",
                    task,
                    "--region",
                    args.city,
                    "--head-scope-label",
                    "city",
                    "--seeds",
                    seeds,
                    "--models",
                    requested_models,
                    "--aether-model",
                    model_name,
                    "--aether-cache-glob",
                    f"*.{suffix}.npy",
                    "--skip-group-models",
                    f"AlphaEarth64,{model_name}",
                    "--device",
                    args.device,
                    "--epochs",
                    str(HEAD_EPOCHS),
                    "--patience",
                    str(HEAD_PATIENCE),
                    "--batch-size",
                    str(head_batch_size),
                    "--lr",
                    str(HEAD_LR),
                    "--hidden",
                    str(HEAD_HIDDEN),
                    "--head-type",
                    "mlp",
                    "--split-mode",
                    "spatial_block",
                    "--spatial-block-km",
                    "1.0",
                    "--sample-frac-per-group",
                    "1.0",
                    "--min-effective-samples",
                    "1",
                    "--save-predictions",
                    "--resume",
                ],
                log,
            )
        print(f"[done] city={args.city} arm={args.arm} task={task}", flush=True)


if __name__ == "__main__":
    main()
