#!/usr/bin/env python3
"""Draw the final 15-city transferability figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from analyze_transfer_associations import fit_factor


RELEASE = Path(__file__).resolve().parents[2]
EXPANDED = RELEASE / "results/transfer"
QUANTITY15 = RELEASE / "results/transfer"
PREDICTORS = RELEASE / "results/transfer"
ILLUSTRATION = RELEASE / "figures/source/Figure3_panel_a_illustration.png"
OUTPUT = RELEASE / "outputs/figures/Figure3"

sys.path.insert(
    0, str(RELEASE / "code/figures")
)
import plot_transferability_base as base  # noqa: E402


GROUPS = ((0, 4), (4, 1), (5, 3), (8, 3), (11, 2), (13, 2))
BOUNDARIES = (3.5, 4.5, 7.5, 10.5, 12.5)
CONTINENT_COLORS = {
    "Africa": "#0072B2",
    "Asia": "#E69F00",
    "Europe": "#009E73",
    "North America": "#CC79A7",
    "Latin America & Caribbean": "#D55E00",
    "Oceania": "#56B4E9",
}
CONTINENT_GROUPS = (
    ("Asia", 0, 4),
    ("Oceania", 4, 1),
    ("Africa", 5, 3),
    ("Europe", 8, 3),
    ("North America", 11, 2),
    ("Latin America & Caribbean", 13, 2),
)


def load_matrix() -> pd.DataFrame:
    table = pd.read_csv(EXPANDED / "country_transfer_15x15_delta_matrix.csv")
    if table.shape != (15, 16) or table.columns[0] != "source_country":
        raise RuntimeError(f"Expected a 15 x 15 transfer matrix, found {table.shape}")
    if not np.isfinite(table.iloc[:, 1:].to_numpy(float)).all():
        raise RuntimeError("Transfer matrix contains non-finite values")
    return table


def draw_heatmap(
    ax: plt.Axes, table: pd.DataFrame, colorbar_ax: plt.Axes | None = None,
) -> None:
    source_labels = table.iloc[:, 0].tolist()
    target_labels = table.columns[1:].tolist()
    values = table.iloc[:, 1:].to_numpy(float) * 100.0
    cmap = LinearSegmentedColormap.from_list(
        "transfer",
        [
            (0.00, "#B7473D"),
            (0.45, "#E7B0AA"),
            (0.48, "#E3E3E3"),
            (0.52, "#E3E3E3"),
            (0.55, "#C9DFE9"),
            (1.00, "#2878A5"),
        ],
        N=256,
    )
    image = ax.imshow(
        np.clip(values, -5, 5), cmap=cmap,
        norm=TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5), aspect="equal",
    )
    ax.set_anchor((0.72, 0.5))
    ax.set_xticks(np.arange(15), target_labels, rotation=42, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(np.arange(15), source_labels)
    ax.set_xlabel("Target city", labelpad=1.5)
    ax.set_ylabel("Source country")
    ax.tick_params(length=0, labelsize=5.2, pad=1.7, labelcolor="#202020")
    for boundary in BOUNDARIES:
        ax.axhline(boundary, color="white", lw=0.65)
        ax.axvline(boundary, color="white", lw=0.65)
    for continent, start, size in CONTINENT_GROUPS:
        ax.add_patch(Rectangle(
            (start - 0.5, start - 0.5), size, size, fill=False,
            edgecolor="white", linewidth=2.2, zorder=4, clip_on=False,
        ))
        ax.add_patch(Rectangle(
            (start - 0.5, start - 0.5), size, size, fill=False,
            edgecolor=CONTINENT_COLORS[continent], linewidth=1.45,
            zorder=5, clip_on=False,
        ))
    for row in range(15):
        for column in range(15):
            value = values[row, column]
            label = "0.0" if abs(value) < 0.05 else f"{value:.1f}"
            ax.text(
                column, row, label, ha="center", va="center", fontsize=5.0,
                color="#252525", zorder=3,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    if colorbar_ax is None:
        colorbar_ax = ax.inset_axes((1.035, 0.10, 0.025, 0.80))
    colorbar = ax.figure.colorbar(
        image, cax=colorbar_ax, extend="both", extendfrac=0.025,
    )
    colorbar.set_ticks((-5, -2.5, 0, 2.5, 5))
    colorbar.ax.tick_params(labelsize=5.2, length=2.2, width=0.55)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.set_title(r"Gain ($\Delta R^2$)" + "\n" + r"($\times10^{-2}$)",
                          fontsize=5.2, pad=2)


def draw_updated_fit(
    ax: plt.Axes, frame: pd.DataFrame, predictor: str, x_column: str,
    xlabel: str, color: str, label: str, log_axis: bool,
) -> None:
    model, result = fit_factor(frame, predictor)
    x = frame[x_column].to_numpy(float)
    raw_predictor = frame[predictor].to_numpy(float)
    if log_axis:
        line_x = np.geomspace(x.min(), x.max(), 300)
        line_predictor = np.log(line_x)
        ax.set_xscale("log")
        ax.set_xticks([500, 1_000, 2_000, 5_000, 10_000, 20_000])
        ax.set_xticklabels(["500", "1,000", "2,000", "5,000", "10,000", "20,000"])
        ax.xaxis.set_minor_formatter(base.NullFormatter())
    else:
        line_x = np.linspace(x.min(), x.max(), 300)
        line_predictor = line_x
    line_z = (line_predictor - raw_predictor.mean()) / raw_predictor.std(ddof=0)
    design = np.column_stack([np.ones(len(line_z)), line_z])
    names = ["Intercept", "z_predictor"]
    fitted_z = design @ model.fe_params.loc[names].to_numpy()
    covariance = model.cov_params().loc[names, names].to_numpy()
    standard_error_z = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))
    y_mean = frame["delta_r2"].mean()
    y_scale = frame["delta_r2"].std(ddof=0)
    mean = y_mean + y_scale * fitted_z
    lower = y_mean + y_scale * (fitted_z - 1.96 * standard_error_z)
    upper = y_mean + y_scale * (fitted_z + 1.96 * standard_error_z)
    ax.scatter(x, frame["delta_r2"], s=9, color=base.POINT_COLOR,
               alpha=0.34, edgecolors="none", zorder=2)
    ax.fill_between(line_x, lower, upper, color=color, alpha=0.20, linewidth=0)
    ax.plot(line_x, mean, color="white", linewidth=3.0, zorder=4)
    ax.plot(line_x, mean, color=color, linewidth=1.65, zorder=5)
    ax.axhline(0, color="#929292", linewidth=0.55, linestyle=(0, (3, 3)))
    p_value = result["wald_p"]
    p_text = r"$P < 0.001$" if p_value < 0.001 else rf"$P = {p_value:.3f}$"
    ax.text(0.05, 0.94, rf"$\beta = {result['standardized_beta']:.3f}$" + "\n" + p_text,
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3)
    ax.set_xlabel(xlabel)
    ax.set_ylim(bottom=-0.04)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.7, width=0.7)
    base.panel_label(ax, label)


def main() -> None:
    quantity, baseline = base.load_quantity(QUANTITY15 / "poi_fraction_three_scopes_mean.csv")
    matrix = load_matrix()
    pairs = pd.read_csv(PREDICTORS / "country_transfer_pairs_with_predictors.csv")
    pairs = pairs.loc[~pairs["matched"].astype(bool)].copy()
    if len(pairs) != 210:
        raise RuntimeError("Expected 210 off-diagonal transfer pairs")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.linewidth": 0.65,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.2, 9.25), facecolor="white")
    outer = figure.add_gridspec(
        5, 1, height_ratios=[2.55, 0.22, 2.45, 0.78, 1.80],
        left=0.075, right=0.94, bottom=0.065, top=0.975, hspace=0,
    )
    illustration_ax = figure.add_subplot(outer[0])
    base.draw_illustration(illustration_ax, ILLUSTRATION)
    figure.text(0.043, 0.978, "a", ha="left", va="top", fontsize=10,
                fontweight="bold")

    middle = outer[2].subgridspec(
        1, 2, width_ratios=[0.98, 1.32], wspace=0.04,
    )
    line_ax = figure.add_subplot(middle[0])
    base.draw_line(line_ax, quantity, baseline, "b")
    line_ax.set_xticks([0, 5, 20, 40, 60, 80, 100])
    line_ax.tick_params(labelsize=6.0, length=2.1)
    line_ax.xaxis.label.set_size(6.5)
    line_ax.yaxis.label.set_size(6.5)
    line_ax.legend(
        fontsize=6.2, handlelength=1.5, labelspacing=0.17,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
        borderpad=0.20, loc="lower right", bbox_to_anchor=(0.99, 0.23),
    )

    heatmap_slot = middle[1].subgridspec(
        1, 2, width_ratios=[1.0, 0.026], wspace=0.01,
    )
    heatmap_ax = figure.add_subplot(heatmap_slot[0])
    colorbar_ax = figure.add_subplot(heatmap_slot[1])
    colorbar_position = colorbar_ax.get_position()
    colorbar_ax.set_position([
        colorbar_position.x0,
        colorbar_position.y0 + colorbar_position.height * 0.10,
        colorbar_position.width,
        colorbar_position.height * 0.80,
    ])
    draw_heatmap(heatmap_ax, matrix, colorbar_ax)
    figure.text(
        heatmap_ax.get_position().x0 - 0.026,
        heatmap_ax.get_position().y1 + 0.010, "c",
        ha="left", va="bottom", fontsize=10, fontweight="bold",
    )
    legend_order = (
        "Africa", "Asia", "Europe", "North America",
        "Latin America & Caribbean", "Oceania",
    )
    figure.legend(
        handles=[Patch(facecolor=CONTINENT_COLORS[name], edgecolor="none", label=name)
                 for name in legend_order],
        loc="lower center",
        bbox_to_anchor=(0.72, heatmap_ax.get_position().y1 + 0.006),
        ncol=6, frameon=False, fontsize=5.0, handlelength=1.0,
        columnspacing=0.8, handletextpad=0.3, borderaxespad=0,
    )

    lower = outer[4].subgridspec(1, 2, wspace=0.22)
    similarity_ax = figure.add_subplot(lower[0])
    distance_ax = figure.add_subplot(lower[1], sharey=similarity_ax)
    draw_updated_fit(similarity_ax, pairs, "aef_similarity", "aef_similarity",
                  "AEF representation similarity", "#276B8E", "d", False)
    draw_updated_fit(distance_ax, pairs, "log_distance_km", "distance_km",
                  "Geographic distance (km)", "#C84C2F", "e", True)
    similarity_ax.set_ylabel(r"Mean transfer gain ($\Delta R^2$)")
    similarity_ax.set_xlabel("AEF similarity")
    distance_ax.set_xlabel("Geographic distance (km)")
    distance_ax.tick_params(labelleft=False)
    for ax in (similarity_ax, distance_ax):
        ax.tick_params(labelsize=5.6, length=2.0)
        ax.xaxis.label.set_size(6.1)
        ax.yaxis.label.set_size(6.1)
        for text in ax.texts:
            if "beta" in text.get_text() or "\\beta" in text.get_text():
                text.set_fontsize(6.2)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    plt.close(figure)
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
