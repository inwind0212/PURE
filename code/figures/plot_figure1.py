#!/usr/bin/env python3
"""Regenerate Figure 1 from released country results and country boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


RELEASE = Path(__file__).resolve().parents[2]
TASKS = ("gdp", "pop", "ntl", "pm25", "lst")
TASK_LABELS = {
    "gdp": "GDP",
    "pop": "Population",
    "ntl": "Night-time lights",
    "pm25": r"PM$_{2.5}$",
    "lst": "Land-surface temperature",
}
ROBINSON = "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs"
NO_DATA = "#d9d9d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--country-seed-results",
        type=Path,
        default=RELEASE / "results/main/global_five_task_country_seed.csv",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        required=True,
        help="Natural Earth-compatible country boundary file.",
    )
    parser.add_argument("--out-dir", type=Path, default=RELEASE / "outputs/figures")
    parser.add_argument("--stem", default="Figure1")
    return parser.parse_args()


def country_codes(world: gpd.GeoDataFrame) -> pd.Series:
    candidates = (
        "ADM0_A3_CN", "ADM0_A3", "ISO_A3", "ADM0_A3_US", "SOV_A3",
        "BRK_A3", "GU_A3", "SU_A3",
    )
    code = pd.Series(pd.NA, index=world.index, dtype="object")
    for column in candidates:
        if column in world.columns:
            values = world[column].astype("string")
            valid = values.str.fullmatch(r"[A-Z]{3}", na=False) & values.ne("-99")
            code = code.mask(code.isna() & valid, values)
    if "SOVEREIGNT" in world.columns:
        code = code.mask(world["SOVEREIGNT"].eq("China"), "CHN")
    return code


def load_world(path: Path) -> gpd.GeoDataFrame:
    world = gpd.read_file(path)
    for column in ("ADMIN", "NAME", "NAME_LONG"):
        if column in world.columns:
            world = world.loc[world[column].ne("Antarctica")].copy()
            break
    world["country_code"] = country_codes(world)
    return world.to_crs(ROBINSON)


def load_task_means(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    required = {"task", "group", "seed", "AEF_R2"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    results["AEF_R2"] = pd.to_numeric(results["AEF_R2"], errors="coerce")
    return (
        results.loc[results["task"].isin(TASKS)]
        .groupby(["task", "group"], as_index=False)["AEF_R2"]
        .mean()
    )


def main() -> None:
    args = parse_args()
    world = load_world(args.boundaries)
    means = load_task_means(args.country_seed_results)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.05, 5.72), dpi=600)
    grid = fig.add_gridspec(
        3, 4, left=0.018, right=0.982, top=0.965, bottom=0.105,
        wspace=0.05, hspace=0.155,
    )
    axes = (
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[2, 1:3]),
    )
    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis_r")

    summary = []
    for letter, task, axis in zip("abcde", TASKS, axes):
        values = means.loc[means["task"].eq(task)].set_index("group")["AEF_R2"]
        panel = world.copy()
        panel["AEF_R2"] = panel["country_code"].map(values)
        panel["display_R2"] = panel["AEF_R2"].clip(0.0, 1.0)
        panel.plot(ax=axis, color=NO_DATA, edgecolor="white", linewidth=0.026)
        observed = panel["display_R2"].notna()
        panel.loc[observed].plot(
            ax=axis, column="display_R2", cmap=cmap, norm=norm,
            edgecolor="white", linewidth=0.020,
        )
        panel.boundary.plot(
            ax=axis, color="#626b73", linewidth=0.016, alpha=0.24
        )
        axis.set_xlim(-16_850_000, 16_850_000)
        axis.set_ylim(-5_450_000, 8_250_000)
        axis.set_title(TASK_LABELS[task], pad=4)
        axis.text(
            -0.015, 1.015, letter, transform=axis.transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold",
        )
        axis.set_axis_off()
        summary.append({
            "task": task,
            "countries": int(values.notna().sum()),
            "negative_r2_clipped": int((values < 0).sum()),
        })

    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    color_axis = fig.add_axes([0.350, 0.052, 0.300, 0.010])
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label("AEF performance ($R^2$)", labelpad=1.3)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    fig.legend(
        handles=[Patch(facecolor=NO_DATA, edgecolor="#b8b8b5", label="No data")],
        loc="center left", bbox_to_anchor=(0.675, 0.057), frameon=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.out_dir / f"{args.stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)
    pd.DataFrame(summary).to_csv(
        args.out_dir / f"{args.stem}_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
