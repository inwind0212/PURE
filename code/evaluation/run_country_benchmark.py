#!/usr/bin/env python
"""Run final country-level downstream jobs from an explicit YAML manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_HEAD = PROJECT_ROOT / "scripts" / "train_downstream_heads.py"


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def validate_cache(cache_dir: Path, required_protocol: str) -> None:
    meta_path = cache_dir.parent / "cityrepavg_meta.json"
    if not meta_path.is_file():
        raise RuntimeError(f"Missing area-average metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = metadata.get("protocol")
    if actual != required_protocol:
        raise RuntimeError(
            f"Refusing incompatible cache {cache_dir}: expected={required_protocol!r}, "
            f"actual={actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--region", action="append", default=[])
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    jobs_doc = yaml.safe_load(args.jobs.read_text(encoding="utf-8"))
    task_filter = set(args.task)
    region_filter = set(args.region)
    required_cache_protocol = protocol["features"]["cache_protocol"]
    head = protocol["head"]
    default_models = list(protocol["features"]["models"])

    for job in jobs_doc["jobs"]:
        task = str(job["task"])
        region = str(job["region"])
        if task_filter and task not in task_filter:
            continue
        if region_filter and region not in region_filter:
            continue
        if task not in protocol["tasks"]:
            raise KeyError(f"Task {task!r} is absent from the final protocol")

        task_spec = protocol["tasks"][task]
        requested_models = job.get("models", default_models)
        aether_model = str(job.get("aether_model", protocol["features"]["models"][1]))
        if not isinstance(requested_models, list) or not requested_models:
            raise ValueError(f"Job models must be a non-empty list: {requested_models!r}")
        models = ",".join(str(model) for model in requested_models)
        cache_dir = Path(job["cache_dir"])
        labels = Path(job["labels_parquet"])
        output = Path(job["out_dir"])
        validate_cache(cache_dir, required_cache_protocol)
        if not labels.is_file():
            raise FileNotFoundError(labels)

        command = [
            sys.executable,
            "-u",
            str(TRAIN_HEAD),
            "--cache-dir",
            str(cache_dir),
            "--labels-parquet",
            str(labels),
            "--group-col",
            str(job.get("group_col", "country_code")),
            "--group-name-col",
            str(job.get("group_name_col", "country_name")),
            "--out-dir",
            str(output),
            "--task",
            task,
            "--region",
            region,
            "--head-scope-label",
            "country",
            "--seeds",
            ",".join(str(seed) for seed in head["seeds"]),
            "--models",
            models,
            "--aether-model",
            aether_model,
            "--aether-cache-glob",
            str(job["aether_cache_glob"]),
            "--skip-global-models",
            models,
            "--device",
            args.device,
            "--epochs",
            str(head["epochs"]),
            "--patience",
            str(head["patience"]),
            "--batch-size",
            str(task_spec["head_batch_size"]),
            "--lr",
            str(head["learning_rate"]),
            "--hidden",
            str(head["hidden_dim"]),
            "--head-type",
            str(head["type"]),
            "--split-mode",
            str(head["split_mode"]),
            "--spatial-block-km",
            str(head["spatial_block_km"]),
            "--sample-frac-per-group",
            "1.0",
            "--min-effective-samples",
            str(protocol["eligibility"]["training_min_effective_samples_per_country_task"]),
            "--save-predictions",
            "--resume",
        ]
        if region == "china":
            command.extend(
                [
                    "--merge-group-codes",
                    ",".join(protocol["territories"]["china_merge_before_split_and_z"]),
                    "--merge-group-name",
                    "China",
                ]
            )
        run(command, output / "run.log")
        print(f"[done] region={region} task={task}", flush=True)


if __name__ == "__main__":
    main()
