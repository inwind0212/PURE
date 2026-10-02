#!/usr/bin/env python
"""Combined Nature-style POI availability and performance figure (panels a-i)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


MM = 1 / 25.4
TASKS = ["gdp", "pop", "ntl", "pm25", "lst"]
TASK_LABELS = {
    "gdp": "GDP",
    "pop": "Population",
    "ntl": "Night-time lights",
    "pm25": r"PM$_{2.5}$",
    "lst": "Land surface temperature",
}
TASK_SHORT = ["GDP", "POP", "NTL", r"PM$_{2.5}$", "LST"]
TASK_COLORS = {
    "gdp": "#E69F00",
    "pop": "#0072B2",
    "ntl": "#D55E00",
    "pm25": "#009E73",
    "lst": "#CC79A7",
}
SUMMARY_TASK_COLORS = {
    "gdp": "#0072B2",
    "pop": "#D55E00",
    "ntl": "#6F4BA8",
    "pm25": "#CC79A7",
    "lst": "#009E73",
}
SUMMARY_TASK_MARKERS = {
    "gdp": "v",
    "pop": "s",
    "ntl": "X",
    "pm25": "P",
    "lst": "^",
}
CONTINENT_ORDER = [
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "Latin America & Caribbean",
    "Oceania",
]
SUMMARY_ORDER = [
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "Latin America & Caribbean",
    "Oceania",
]
SUMMARY_LABELS = {
    "Africa": "Africa",
    "Asia": "Asia",
    "Europe": "Europe",
    "North America": "N. America",
    "Latin America & Caribbean": "LAC",
    "Oceania": "Oceania",
}
CONTINENT_COLORS = {
    "Africa": "#0072B2",
    "Asia": "#E69F00",
    "Europe": "#009E73",
    "North America": "#CC79A7",
    "Latin America & Caribbean": "#D55E00",
    "Oceania": "#56B4E9",
}
POI_ORDER = [
    "North America",
    "Oceania",
    "Europe",
    "Latin America & Caribbean",
    "Asia",
    "Africa",
]
POI_LABELS = [
    "North America",
    "Oceania",
    "Europe",
    "Latin America\n& Caribbean",
    "Asia",
    "Africa",
]
POI_COLORS = {
    "Africa": "#0072B2",
    "Asia": "#E69F00",
    "Europe": "#009E73",
    "North America": "#CC79A7",
    "Latin America & Caribbean": "#D55E00",
    "Oceania": "#56B4E9",
}
AE_COLOR = "#4C78A8"
AETHER_COLOR = "#E6812D"
Y_LIMITS = (-0.20, 0.40)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    tables = root / "results/main"
    p = argparse.ArgumentParser()
    p.add_argument("--country-mad", type=Path, default=tables / "figure2_country_r2_mad.csv")
    p.add_argument("--continent-poi", type=Path, default=tables / "figure2_continent_poi_summary.csv")
    p.add_argument("--continent-summary", type=Path, default=tables / "figure2_continent_task_gain.csv")
    p.add_argument(
        "--country-points",
        type=Path,
        default=tables / "figure2_country_scatter_points.csv",
    )
    p.add_argument("--out-dir", type=Path, default=root / "outputs/figures")
    p.add_argument("--stem", default="Figure2")
    return p.parse_args()


def style_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#5E6368")
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.tick_params(colors="#3F454B", width=0.55, length=2.5, pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E4E7EA", linewidth=0.45, zorder=0)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.2, fontweight="bold", pad=6)


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_beta(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"


def plot_variability(ax: plt.Axes, variability: pd.DataFrame) -> None:
    ae = variability.loc["AlphaEarth", TASKS].astype(float).to_numpy()
    aligned = variability.loc["POI-aligned", TASKS].astype(float).to_numpy()
    change = 100 * (aligned - ae) / ae
    y = np.arange(len(TASKS))
    for yi, x0, x1 in zip(y, ae, aligned):
        ax.plot([x0, x1], [yi, yi], color="#A9AFB5", linewidth=1, zorder=1)
    ax.scatter(ae, y, s=18, color=AE_COLOR, label="AEF", zorder=3)
    ax.scatter(aligned, y, s=20, marker="s", color=AETHER_COLOR, label="PURE", zorder=3)
    for yi, x0, x1, pct in zip(y, ae, aligned, change):
        ax.text(max(x0, x1) + 0.004, yi, f"{pct:.1f}%", va="center", fontsize=5.8, color="#30353A")
    ax.set_yticks(y, TASK_SHORT)
    ax.invert_yaxis()
    ax.set_xlim(0.025, 0.098)
    ax.set_xticks([0.03, 0.05, 0.07, 0.09])
    ax.set_xlabel(r"Cross-country variability (MAD of $R^2$)")
    panel_title(ax, "a", "Performance variability")
    legend = ax.legend(
        frameon=True,
        loc="lower right",
        handletextpad=0.3,
        labelspacing=0.2,
        fontsize=5.8,
        facecolor="white",
        edgecolor="#C7CCD1",
        framealpha=0.95,
        fancybox=False,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(0.4)
    style_axis(ax, "x")


def plot_poi_availability(ax_pc: plt.Axes, ax_total: plt.Axes, poi: pd.DataFrame) -> None:
    poi = poi.set_index("continent_summary").loc[POI_ORDER]
    y = np.arange(len(POI_ORDER))
    colors = [POI_COLORS[c] for c in POI_ORDER]
    panels = [
        (ax_pc, poi["poi_per_1000_people"].astype(float).to_numpy(), "b", "Per-capita POIs", "POIs per 1,000 people"),
        (ax_total, poi["poi_million"].astype(float).to_numpy(), "c", "Total POIs", "Million POIs"),
    ]
    for j, (ax, values, letter, title, xlabel) in enumerate(panels):
        ax.barh(y, values, color=colors, height=0.56, edgecolor="none", zorder=2)
        for yi, value in zip(y, values):
            ax.text(value + 0.75, yi, f"{value:.1f}", va="center", fontsize=5.9, color="#30353A")
        ax.set_yticks(y, POI_LABELS if j == 0 else [])
        if j == 1:
            ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel(xlabel)
        panel_title(ax, letter, title)
        style_axis(ax, "x")


def plot_continent_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    y_pos = {name: i for i, name in enumerate(SUMMARY_ORDER)}
    offsets = np.linspace(-0.25, 0.25, len(TASKS))
    for off, task in zip(offsets, TASKS):
        part = summary[summary["task"] == task]
        names = [r for r in SUMMARY_ORDER if r in set(part["region"])]
        xs = [float(part.loc[part["region"] == r, "R2_delta"].iloc[0]) for r in names]
        ys = [y_pos[r] + off for r in names]
        ax.scatter(
            xs,
            ys,
            s=17,
            marker=SUMMARY_TASK_MARKERS[task],
            color=SUMMARY_TASK_COLORS[task],
            edgecolor="white",
            linewidth=0.25,
            label=TASK_SHORT[TASKS.index(task)],
            zorder=3,
        )
    means = summary.groupby("region")["R2_delta"].mean()
    ax.scatter(
        [means[r] for r in SUMMARY_ORDER],
        [y_pos[r] for r in SUMMARY_ORDER],
        s=25,
        color="#111111",
        marker=(4, 1, 0),
        edgecolors="none",
        label="Mean",
        zorder=4,
    )
    ax.set_yticks(range(len(SUMMARY_ORDER)), [SUMMARY_LABELS[r] for r in SUMMARY_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-0.025, 0.13)
    ax.set_xlabel(r"Mean $ΔR^2$")
    panel_title(ax, "d", "Continent summary")
    legend = ax.legend(
        frameon=True,
        loc="lower right",
        ncol=2,
        fontsize=5.5,
        handletextpad=0.25,
        columnspacing=0.7,
        labelspacing=0.2,
        facecolor="white",
        edgecolor="#C7CCD1",
        framealpha=0.95,
        fancybox=False,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(0.4)
    style_axis(ax, "x")


def plot_country_scatter(ax: plt.Axes, data: pd.DataFrame, task: str, letter: str) -> None:
    d = data[(data["task"] == task) & data["plot_ok"].astype(bool)].copy()
    for continent in CONTINENT_ORDER:
        part = d[d["continent"] == continent]
        if part.empty:
            continue
        y = part["R2_delta"].astype(float)
        masks = [
            (y.between(*Y_LIMITS), "o", y.clip(*Y_LIMITS)),
            (y < Y_LIMITS[0], "v", pd.Series(Y_LIMITS[0] + 0.01, index=part.index)),
            (y > Y_LIMITS[1], "^", pd.Series(Y_LIMITS[1] - 0.01, index=part.index)),
        ]
        for mask, marker, yp in masks:
            if not mask.any():
                continue
            ax.scatter(part.loc[mask, "poi_per_1000_people"], yp.loc[mask], s=15, marker=marker, color=CONTINENT_COLORS[continent], alpha=0.80, edgecolor="white", linewidth=0.22, zorder=3)
    x = d["poi_per_1000_people"].astype(float).to_numpy()
    y = d["R2_delta"].astype(float).to_numpy()
    fit = linregress(x, y)
    xs = np.linspace(0, max(48, float(np.nanmax(x))), 120)
    ax.plot(xs, fit.intercept + fit.slope * xs, color="#333333", linewidth=1.0, zorder=2)
    ax.axhline(0, color="#747B82", linewidth=0.7, linestyle="--", zorder=1)
    p_text = r"P<0.001" if fit.pvalue < 0.001 else rf"P={fit.pvalue:.3f}"
    ax.text(0.98, 0.95, rf"$\beta={format_beta(fit.slope)},\ {p_text}$", transform=ax.transAxes, ha="right", va="top", fontsize=6.1, color="#374151", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0))
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4])
    ax.set_xlabel("POIs per 1,000 people")
    ax.set_ylabel(r"$ΔR^2$")
    panel_title(ax, letter, TASK_LABELS[task])
    style_axis(ax, "both")


def main() -> None:
    args = parse_args()
    variability = pd.read_csv(args.country_mad).set_index("Model")
    poi = pd.read_csv(args.continent_poi)
    points = pd.read_csv(args.country_points)
    # Use a single regional definition throughout the figure. The POI
    # availability table already includes Russia in Europe.
    points["continent"] = points["continent"].replace({"Russia": "Europe"})
    eligible = points[points["plot_ok"].astype(bool)].copy()
    summary = (
        eligible.groupby(["task", "continent"], as_index=False)
        .agg(R2_delta=("R2_delta", "mean"), n_countries=("group", "nunique"))
        .rename(columns={"continent": "region"})
    )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.7,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(180 * MM, 172 * MM))
    outer = fig.add_gridspec(3, 3, height_ratios=[0.88, 1, 1], hspace=0.48, wspace=0.34)
    ax_a = fig.add_subplot(outer[0, 0])
    bgrid = outer[0, 1:].subgridspec(1, 2, wspace=0.27)
    ax_b1 = fig.add_subplot(bgrid[0, 0])
    ax_b2 = fig.add_subplot(bgrid[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    scatter_axes = [fig.add_subplot(outer[1, 1]), fig.add_subplot(outer[1, 2]), fig.add_subplot(outer[2, 0]), fig.add_subplot(outer[2, 1]), fig.add_subplot(outer[2, 2])]

    plot_variability(ax_a, variability)
    plot_poi_availability(ax_b1, ax_b2, poi)
    plot_continent_summary(ax_c, summary)
    for ax, task, letter in zip(scatter_axes, TASKS, list("efghi")):
        plot_country_scatter(ax, points, task, letter)

    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CONTINENT_COLORS[c], markeredgecolor="white", markeredgewidth=0.3, markersize=5.4, label=c) for c in CONTINENT_ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.000), ncol=7, frameon=False, columnspacing=0.9, handletextpad=0.3, fontsize=6.2)
    fig.subplots_adjust(left=0.068, right=0.992, top=0.965, bottom=0.075)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {})]:
        out = args.out_dir / f"{args.stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor="white", **kwargs)
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
