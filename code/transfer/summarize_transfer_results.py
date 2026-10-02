#!/usr/bin/env python3
"""Summarize the final POI-fraction and cross-country transfer evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


TASKS = ("gdp", "pop", "ntl", "pm25", "lst")
RESULT_ROOT_NAME = "results_transferability_2024"


def read_summary(path: Path, model: str) -> tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("method") == "global_head" and row.get("model") == model
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one global_head row for {model} in {path}, found {len(rows)}")
    return float(rows[0]["R2_mean"]), float(rows[0]["R2_std"])


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path) -> None:
    root = Path(root)
    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    jobs = json.loads((root / "eval_queue.json").read_text(encoding="utf-8"))
    cities = manifest["cities"]
    output = root / "analysis"

    def summary_path(job: dict, task: str) -> Path:
        return (
            root / "cities" / job["target"] / "downstream" / RESULT_ROOT_NAME
            / job["model_tag"] / task / "summary_mean.csv"
        )

    matched = {
        job["target"]: job for job in jobs
        if job["role"] == "quantity_curve_and_transfer"
    }
    if set(matched) != set(cities):
        raise RuntimeError("Each target must have one matched full-country evaluation")

    aef = {}
    for city, job in matched.items():
        for task in TASKS:
            aef[(city, task)] = read_summary(summary_path(job, task), "AlphaEarth64")[0]

    quantity_rows = []
    for job in jobs:
        if not job["role"].startswith("quantity_curve"):
            continue
        city = job["target"]
        for task in TASKS:
            score, score_sd = read_summary(
                summary_path(job, task), f"AETHER_{job['model_tag']}"
            )
            baseline = aef[(city, task)]
            quantity_rows.append({
                "city": city,
                "city_label": cities[city]["name"],
                "country": cities[city]["country"],
                "scope": job["scope"],
                "poi_retained_pct": job["pct"],
                "task": task,
                "r2": score,
                "r2_seed_sd": score_sd,
                "aef_r2": baseline,
                "delta_r2": score - baseline,
            })
    quantity_rows.sort(key=lambda row: (row["city"], row["scope"], row["poi_retained_pct"], row["task"]))
    write_csv(
        output / "poi_fraction_city_task.csv",
        ["city", "city_label", "country", "scope", "poi_retained_pct", "task", "r2", "r2_seed_sd", "aef_r2", "delta_r2"],
        quantity_rows,
    )

    city_groups = defaultdict(list)
    for row in quantity_rows:
        city_groups[(row["city"], row["scope"], row["poi_retained_pct"])].append(row)
    by_city = []
    for city in cities:
        baseline = fmean(aef[(city, task)] for task in TASKS)
        for scope in ("city", "country", "country_loco"):
            by_city.append({"city": city, "scope": scope, "poi_retained_pct": 0, "r2": baseline, "delta_r2": 0.0})
    for (city, scope, pct), rows in city_groups.items():
        by_city.append({
            "city": city, "scope": scope, "poi_retained_pct": pct,
            "r2": fmean(row["r2"] for row in rows),
            "delta_r2": fmean(row["delta_r2"] for row in rows),
        })
    by_city.sort(key=lambda row: (row["city"], row["scope"], row["poi_retained_pct"]))
    write_csv(output / "poi_fraction_three_scopes_by_city.csv", ["city", "scope", "poi_retained_pct", "r2", "delta_r2"], by_city)

    mean_groups = defaultdict(list)
    for row in by_city:
        mean_groups[(row["scope"], row["poi_retained_pct"])].append(row)
    means = [{
        "scope": scope, "poi_retained_pct": pct,
        "r2": fmean(row["r2"] for row in rows),
        "delta_r2": fmean(row["delta_r2"] for row in rows),
    } for (scope, pct), rows in mean_groups.items()]
    means.sort(key=lambda row: (row["scope"], row["poi_retained_pct"]))
    write_csv(output / "poi_fraction_three_scopes_mean.csv", ["scope", "poi_retained_pct", "r2", "delta_r2"], means)

    transfer_rows = []
    transfer_jobs = [job for job in jobs if job["role"] in {"quantity_curve_and_transfer", "cross_country_transfer"}]
    for job in transfer_jobs:
        for task in TASKS:
            score, score_sd = read_summary(summary_path(job, task), f"AETHER_{job['model_tag']}")
            baseline = aef[(job["target"], task)]
            transfer_rows.append({
                "source": job["source"],
                "source_country": cities[job["source"]]["country"],
                "target": job["target"],
                "target_city": cities[job["target"]]["name"],
                "matched": job["source"] == job["target"],
                "task": task,
                "r2": score,
                "r2_seed_sd": score_sd,
                "aef_r2": baseline,
                "delta_r2": score - baseline,
            })
    transfer_rows.sort(key=lambda row: (row["source"], row["target"], row["task"]))
    write_csv(output / "country_transfer_task.csv", ["source", "source_country", "target", "target_city", "matched", "task", "r2", "r2_seed_sd", "aef_r2", "delta_r2"], transfer_rows)

    pair_groups = defaultdict(list)
    for row in transfer_rows:
        pair_groups[(row["source"], row["target"])].append(row)
    pairs = []
    for (source, target), rows in pair_groups.items():
        pairs.append({
            "source": source, "source_country": cities[source]["country"],
            "target": target, "target_city": cities[target]["name"],
            "matched": source == target,
            "r2": fmean(row["r2"] for row in rows),
            "aef_r2": fmean(row["aef_r2"] for row in rows),
            "delta_r2": fmean(row["delta_r2"] for row in rows),
        })
    pairs.sort(key=lambda row: (row["source"], row["target"]))
    write_csv(output / "country_transfer_pairs.csv", ["source", "source_country", "target", "target_city", "matched", "r2", "aef_r2", "delta_r2"], pairs)

    pair_lookup = {(row["source"], row["target"]): row["delta_r2"] for row in pairs}
    matrix_rows = []
    for source in cities:
        row = {"source_country": cities[source]["country"]}
        row.update({cities[target]["name"]: pair_lookup[(source, target)] for target in cities})
        matrix_rows.append(row)
    write_csv(output / "country_transfer_15x15_delta_matrix.csv", ["source_country", *[record["name"] for record in cities.values()]], matrix_rows)

    off_diagonal = [row["delta_r2"] for row in pairs if not row["matched"]]
    diagonal = [row["delta_r2"] for row in pairs if row["matched"]]
    summary = {
        "cities": len(cities), "source_target_pairs": len(pairs),
        "off_diagonal_pairs": len(off_diagonal),
        "positive_off_diagonal_pairs": sum(value > 0 for value in off_diagonal),
        "mean_off_diagonal_delta_r2": fmean(off_diagonal),
        "mean_matched_delta_r2": fmean(diagonal),
    }
    (output / "country_transfer_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root)


if __name__ == "__main__":
    main()
