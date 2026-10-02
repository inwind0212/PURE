#!/usr/bin/env python3
"""Run the final 15-city POI-availability and cross-country transfer experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from spatial_support import split_region_rows


PROJECT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT / "code"
STORE = Path(os.environ["PURE_FEATURE_STORE"])
ROOT = Path(os.environ["PURE_TRANSFER_ROOT"])
BASE_CONFIG = PROJECT / (
    "code/configs/train_pure.yaml"
)
TRAIN = CODE_ROOT / "training/train_global_aether.py"
EVALUATE = CODE_ROOT / "transfer/run_city_evaluation.py"
PYTHON = Path(os.environ.get("PURE_PYTHON", sys.executable))
PCTS = (1, 2, 5, 10, 20, 40, 60, 80, 100)
TASKS = ("gdp", "pop", "ntl", "pm25", "lst")
TRAIN_SEED = 45

SUPPORT_ROOT = Path(os.environ["PURE_SETTLEMENT_SUPPORT_ROOT"])
RAW_SAMPLE_ROOT = Path(os.environ["PURE_CITY_RAW_ROOT"])
AE_CACHE_ROOT = Path(os.environ["PURE_AE_CACHE_ROOT"])

# city: display name, country, ISO2, feature-store component, support, raw samples
CITY_SPECS = {
    "bangkok": ("Bangkok", "Thailand", "TH", "asia_ex_china"),
    "chengdu": ("Chengdu", "China", None, "china"),
    "osaka": ("Osaka", "Japan", "JP", "asia_ex_china"),
    "mumbai": ("Mumbai", "India", "IN", "asia_ex_china"),
    "melbourne": ("Melbourne", "Australia", "AU", "oceania"),
    "johannesburg": ("Johannesburg", "South Africa", "ZA", "africa"),
    "nairobi": ("Nairobi", "Kenya", "KE", "africa"),
    "luanda": ("Luanda", "Angola", "AO", "africa"),
    "lisbon": ("Lisbon", "Portugal", "PT", "europe"),
    "paris": ("Paris", "France", "FR", "europe"),
    "manchester": ("Manchester", "United Kingdom", "GB", "europe"),
    "montreal": ("Montreal", "Canada", "CA", "north_america_ex_us"),
    "los_angeles": ("Los Angeles", "United States", None, "us"),
    "rio_de_janeiro": ("Rio de Janeiro", "Brazil", "BR", "latin_america_caribbean"),
    "buenos_aires": ("Buenos Aires", "Argentina", "AR", "latin_america_caribbean"),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def support_path(city: str) -> Path:
    return SUPPORT_ROOT / f"{city}.gpkg"



def raw_samples_path(city: str) -> Path:
    return RAW_SAMPLE_ROOT / city / "downstream/raw_samples"



def ae_source_cache(city: str) -> Path:
    return AE_CACHE_ROOT / city / "city_full"



def component_records() -> tuple[dict[str, dict], int]:
    meta = json.loads((STORE / "feature_store_meta.json").read_text(encoding="utf-8"))
    offset = 0
    records = {}
    for component in meta["components"]:
        row = copy.deepcopy(component)
        row["offset"] = offset
        records[row["region"]] = row
        offset += int(row["n"])
    if offset != int(meta["rows"]):
        raise RuntimeError("Component row counts do not match feature-store rows")
    return records, offset


def build_inverse_source_index(total_rows: int) -> np.ndarray:
    source = np.load(STORE / "source_global_index.npy", mmap_mode="r")
    if source.shape != (total_rows,):
        raise RuntimeError(f"Unexpected source index shape: {source.shape}")
    inverse = np.empty(total_rows, dtype=np.int64)
    chunk = 4_000_000
    for start in range(0, total_rows, chunk):
        stop = min(total_rows, start + chunk)
        values = np.asarray(source[start:stop], dtype=np.int64)
        inverse[values] = np.arange(start, stop, dtype=np.int64)
    return inverse


def country_local_rows(component: dict, iso2: str | None) -> np.ndarray:
    n = int(component["n"])
    if iso2 is None:
        return np.arange(n, dtype=np.int64)
    parquet = pq.ParquetFile(component["paths"]["poi_sample"])
    if parquet.metadata.num_rows != n:
        raise RuntimeError(
            f"{component['region']}: parquet rows {parquet.metadata.num_rows} != {n}"
        )
    parts = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=1_000_000, columns=["country_code"]):
        values = pc.utf8_upper(
            pc.utf8_trim_whitespace(pc.cast(batch.column(0), pa.string()))
        )
        keep = pc.fill_null(pc.equal(values, iso2), False).to_numpy(zero_copy_only=False)
        parts.append(np.flatnonzero(keep).astype(np.int64) + offset)
        offset += batch.num_rows
    if offset != n:
        raise RuntimeError(f"{component['region']}: incomplete country scan")
    selected = np.concatenate(parts)
    if selected.size == 0:
        raise RuntimeError(f"No rows found for ISO2={iso2} in {component['region']}")
    return selected


def nested_filter(base_rows: np.ndarray, pct: int, permutation: np.ndarray) -> np.ndarray:
    rows = max(1, int(base_rows.size * pct / 100))
    return np.sort(base_rows[permutation[:rows]]).astype(np.int64, copy=False)


def make_train_job(
    *, city: str, scope: str, pct: int, row_filter: Path, rows: int, base: dict,
) -> dict:
    model_tag = f"pure2024_{scope}_{city}_p{pct:03d}"
    run_name = f"{model_tag}_ep10_fixedlr1e4_b512_s45"
    model_root = ROOT / "models" / scope / city
    config_path = ROOT / "configs" / scope / city / f"train_{run_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base)
    cfg["paths"]["feature_store"] = str(STORE)
    cfg["paths"]["row_filter"] = str(row_filter)
    cfg["paths"]["row_filter_name"] = row_filter.stem
    cfg["sampling"]["anchor_region_mode"] = "none"
    batch_size = min(512, rows)
    cfg["train"].update({
        "batch_size": batch_size,
        "epochs": 10,
        "lr_schedule_epochs": 10,
        "steps_per_epoch": max(1, rows // batch_size),
        "seed": TRAIN_SEED,
        "eta_min_ratio": 1.0,
        "batch_fetch_chunk": 2048,
        "prefetch_chunks": True,
        "io_read_workers": 24,
        "num_workers": 0,
    })
    cfg["model"]["lr"] = 1.0e-4
    cfg["loss"]["xt_mode"] = "one_hot"
    cfg["logging"].update({
        "output_dir": str(model_root),
        "run_name": run_name,
        "save_every": 10,
        "save_last": True,
    })
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    checkpoint = model_root / run_name / "ckpts/epoch_0010.pth"
    return {
        "city": city,
        "scope": scope,
        "pct": pct,
        "rows": rows,
        "steps": int(cfg["train"]["steps_per_epoch"]) * 10,
        "model_tag": model_tag,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "row_filter": str(row_filter),
    }


def prepare() -> None:
    for path in (STORE, BASE_CONFIG, TRAIN, EVALUATE):
        if not path.exists():
            raise FileNotFoundError(path)
    store_audit_path = STORE / "feature_store_audit.json"
    if not store_audit_path.is_file():
        raise FileNotFoundError(store_audit_path)
    store_audit = json.loads(store_audit_path.read_text(encoding="utf-8"))
    if not store_audit.get("passed"):
        raise RuntimeError(f"Feature-store audit did not pass: {store_audit_path}")
    china_audit = store_audit.get("ae_component_audit", {}).get("china", {})
    for key in ("ae_base_nonzero_fraction", "ae_aug_nonzero_fraction"):
        if float(china_audit.get(key, 0.0)) < 0.95:
            raise RuntimeError(f"China AE audit failed for {key}: {china_audit}")
    ROOT.mkdir(parents=True, exist_ok=True)
    components, total_rows = component_records()
    inverse = build_inverse_source_index(total_rows)
    sp_x = np.load(STORE / "sp_x_100m.npy", mmap_mode="r")
    sp_y = np.load(STORE / "sp_y_100m.npy", mmap_mode="r")
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))

    country_cache: dict[tuple[str, str | None], np.ndarray] = {}
    country_paths: dict[str, Path] = {}
    city_paths: dict[str, Path] = {}
    loco_paths: dict[str, Path] = {}
    cities_manifest = {}
    sampling_audit = {}

    for city, (name, country, iso2, region) in CITY_SPECS.items():
        component = components[region]
        key = (region, iso2)
        if key not in country_cache:
            local = country_local_rows(component, iso2)
            source_rows = int(component["offset"]) + local
            country_cache[key] = np.sort(inverse[source_rows]).astype(np.int64, copy=False)
            del local, source_rows
        country_rows = country_cache[key]
        filter_dir = ROOT / "row_filters" / city
        filter_dir.mkdir(parents=True, exist_ok=True)
        country_path = filter_dir / "country_full.npy"
        np.save(country_path, country_rows)
        city_rows, loco_rows, bounds = split_region_rows(
            country_rows, sp_x, sp_y, support_path(city), chunk_rows=4_000_000
        )
        city_path = filter_dir / "city_full.npy"
        loco_path = filter_dir / "country_loco_full.npy"
        np.save(city_path, city_rows)
        np.save(loco_path, loco_rows)
        country_paths[city] = country_path
        city_paths[city] = city_path
        loco_paths[city] = loco_path

        raw_link = ROOT / "cities" / city / "downstream/raw_samples"
        raw_link.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(raw_link):
            raw_link.symlink_to(raw_samples_path(city), target_is_directory=True)
        cities_manifest[city] = {
            "name": name,
            "country": country,
            "country_iso2": iso2,
            "component": region,
            "support": str(support_path(city)),
            "support_layer": "settlement_support",
            "bounds_4326": bounds,
            "raw_samples": str(raw_link),
            "models": {},
        }
        sampling_audit[city] = {
            "country": country,
            "component": region,
            "country_rows": int(country_rows.size),
            "city_rows": int(city_rows.size),
            "country_loco_rows": int(loco_rows.size),
            "removed_for_loco": int(country_rows.size - loco_rows.size),
            "partition_exact": int(city_rows.size + loco_rows.size) == int(country_rows.size),
            "city_country_overlap": int(np.intersect1d(city_rows, country_rows).size),
        }

    del inverse, sp_x, sp_y, country_cache

    train_jobs = []
    for city in CITY_SPECS:
        scopes = {
            "city": np.load(city_paths[city], mmap_mode="r"),
            "country": np.load(country_paths[city], mmap_mode="r"),
            "country_loco": np.load(loco_paths[city], mmap_mode="r"),
        }
        for scope, base_rows_mmap in scopes.items():
            base_rows = np.asarray(base_rows_mmap, dtype=np.int64)
            permutation = np.random.default_rng(TRAIN_SEED).permutation(base_rows.size)
            for pct in PCTS:
                selected = nested_filter(base_rows, pct, permutation)
                path = ROOT / "row_filters" / city / f"{scope}_p{pct:03d}_nested_s45.npy"
                np.save(path, selected)
                job = make_train_job(
                    city=city, scope=scope, pct=pct, row_filter=path,
                    rows=int(selected.size), base=base,
                )
                train_jobs.append(job)
                cities_manifest[city]["models"][job["model_tag"]] = job
            del base_rows, permutation

    train_by_key = {(j["city"], j["scope"], j["pct"]): j for j in train_jobs}
    eval_jobs = []
    for target in CITY_SPECS:
        for scope in ("city", "country_loco"):
            for pct in PCTS:
                job = train_by_key[(target, scope, pct)]
                eval_jobs.append({
                    "source": target,
                    "target": target,
                    "scope": scope,
                    "pct": pct,
                    "model_tag": job["model_tag"],
                    "checkpoint": job["checkpoint"],
                    "role": "quantity_curve",
                })
        for pct in PCTS[:-1]:
            job = train_by_key[(target, "country", pct)]
            eval_jobs.append({
                "source": target,
                "target": target,
                "scope": "country",
                "pct": pct,
                "model_tag": job["model_tag"],
                "checkpoint": job["checkpoint"],
                "role": "quantity_curve",
            })
    for source in CITY_SPECS:
        job = train_by_key[(source, "country", 100)]
        for target in CITY_SPECS:
            eval_jobs.append({
                "source": source,
                "target": target,
                "scope": "country",
                "pct": 100,
                "model_tag": job["model_tag"],
                "checkpoint": job["checkpoint"],
                "role": "quantity_curve_and_transfer" if source == target else "cross_country_transfer",
            })

    manifest = {
        "protocol_id": "pure_transferability_15city_ep10_b512_s45",
        "created_at": now_iso(),
        "feature_store": str(STORE),
        "feature_store_audit": str(store_audit_path),
        "poi_release": "Overture 2024-12-18.0",
        "poi_text": "A place of {category_leaf}, a type of {category_top}, named {name}.",
        "experiments": {
            "quantity": {
                "scopes": ["city", "country", "country_loco"],
                "percentages": list(PCTS),
                "nested_sampling_seed": TRAIN_SEED,
            },
            "cross_country": {
                "sources": len(CITY_SPECS),
                "targets": len(CITY_SPECS),
                "full_country_poi": True,
                "matrix_evaluations": len(CITY_SPECS) ** 2,
            },
        },
        "training": {
            "architecture": "PURE AEProjSH h2048 out128",
            "epochs": 10,
            "batch_size": 512,
            "learning_rate": 1.0e-4,
            "learning_rate_schedule": "fixed",
            "text_loss": "one_hot",
            "ae_relation_weight": 0.5,
            "seed": TRAIN_SEED,
            "trainer": str(TRAIN),
        },
        "downstream": {
            "tasks": list(TASKS),
            "seeds": [42, 24, 7, 0, 100],
            "head": "MLP hidden1024",
            "epochs": 100,
            "patience": 10,
            "split": "fixed 1-km spatial blocks",
        },
        "cities": cities_manifest,
        "counts": {"training_models": len(train_jobs), "unique_evaluations": len(eval_jobs)},
    }
    (ROOT / "train_queue.json").write_text(json.dumps(train_jobs, indent=2) + "\n")
    (ROOT / "eval_queue.json").write_text(json.dumps(eval_jobs, indent=2) + "\n")
    (ROOT / "sampling_audit.json").write_text(json.dumps(sampling_audit, indent=2) + "\n")
    (ROOT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "root": str(ROOT), "training_models": len(train_jobs),
        "unique_evaluations": len(eval_jobs), "sampling_audit": sampling_audit,
    }, indent=2), flush=True)


def balanced_assignment(jobs: list[dict], worker_index: int, workers: int, weight: str) -> list[dict]:
    assignments = [[] for _ in range(workers)]
    loads = [0] * workers
    for job in sorted(jobs, key=lambda item: int(item.get(weight, 1)), reverse=True):
        target = min(range(workers), key=lambda index: loads[index])
        assignments[target].append(job)
        loads[target] += int(job.get(weight, 1))
    print(json.dumps({"worker": worker_index, "workers": workers, "load": loads[worker_index],
                      "jobs": len(assignments[worker_index])}), flush=True)
    return assignments[worker_index]


def train_worker(worker_index: int, workers: int, physical_gpu: int) -> None:
    jobs = json.loads((ROOT / "train_queue.json").read_text())
    selected = balanced_assignment(jobs, worker_index, workers, "steps")
    status_dir = ROOT / "status/train"
    log_dir = ROOT / "logs/train"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(physical_gpu), "OMP_NUM_THREADS": "12",
                "MKL_NUM_THREADS": "12", "OPENBLAS_NUM_THREADS": "12",
                "NUMEXPR_NUM_THREADS": "12"})
    for job in selected:
        status = status_dir / f"{job['model_tag']}.done"
        checkpoint = Path(job["checkpoint"])
        if status.is_file() and checkpoint.is_file():
            continue
        log = log_dir / f"{job['model_tag']}.log"
        command = [str(PYTHON), "-u", str(TRAIN), "--cfg", job["config"],
                   "--store-dir", str(STORE), "--resume-auto"]
        with log.open("a", encoding="utf-8") as stream:
            print("$ " + " ".join(command), file=stream, flush=True)
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        status.write_text("done\n", encoding="utf-8")


def eval_worker(worker_index: int, workers: int, physical_gpu: int) -> None:
    jobs = json.loads((ROOT / "eval_queue.json").read_text())
    selected = [job for index, job in enumerate(jobs) if index % workers == worker_index]
    status_dir = ROOT / "status/eval"
    log_dir = ROOT / "logs/eval"
    cache_root = ROOT / "downstream_caches"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(physical_gpu), "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
                "NUMEXPR_NUM_THREADS": "8"})
    for job in selected:
        key = f"{job['model_tag']}__to__{job['target']}"
        status = status_dir / f"{key}.done"
        if status.is_file():
            continue
        checkpoint = Path(job["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        ae_cache = ae_source_cache(job["target"])
        for task in TASKS:
            source_files = list((ae_cache / task / "cache").glob("*.ae64.npy"))
            if len(source_files) != 1:
                raise RuntimeError(
                    f"Expected one AlphaEarth cache for {job['target']}/{task}, "
                    f"found {len(source_files)} below {ae_cache}"
                )
        log = log_dir / f"{key}.log"
        command = [
            str(PYTHON), "-u", str(EVALUATE), "--root", str(ROOT),
            "--city", job["target"], "--arm", job["model_tag"],
            "--checkpoint", str(checkpoint), "--model-tag", job["model_tag"],
            "--device", "cuda:0", "--cache-base", str(cache_root),
            "--head-batch-default", "128", "--head-batch-pop", "1024",
            "--result-root-name", "results_transferability_2024",
            "--aether-input-radius-px", "0",
        ]
        if job["role"] != "quantity_curve_and_transfer":
            command.extend(["--aether-only", "--ae-source-cache", str(ae_cache)])
        with log.open("a", encoding="utf-8") as stream:
            print("$ " + " ".join(command), file=stream, flush=True)
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status.write_text("done\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-worker", action="store_true")
    parser.add_argument("--eval-worker", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.train_worker:
        train_worker(args.worker_index, args.workers, args.physical_gpu)
    elif args.eval_worker:
        eval_worker(args.worker_index, args.workers, args.physical_gpu)
    elif args.summarize:
        from summarize_transfer_results import summarize
        summarize(ROOT)
    else:
        parser.error("Choose --prepare, --train-worker, --eval-worker, or --summarize")


if __name__ == "__main__":
    main()
