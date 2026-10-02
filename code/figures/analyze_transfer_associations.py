#!/usr/bin/env python3
"""Build geographic predictors and fit the Figure 3 association models."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf


RELEASE = Path(__file__).resolve().parents[2]
INPUT = RELEASE / "results/transfer/country_transfer_pairs_with_predictors.csv"
CAPITALS = RELEASE / "results/transfer/source_capital_coordinates.csv"
TARGET_CENTROIDS = RELEASE / "results/transfer/target_city_settlement_centroids.csv"
CSV_OUTPUT = RELEASE / "results/transfer/country_transfer_mixed_effects.csv"
JSON_OUTPUT = RELEASE / "results/transfer/country_transfer_mixed_effects.json"


def great_circle(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = np.radians([*left, *right])
    value = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1))))


def fit_factor(frame: pd.DataFrame, predictor: str):
    data = frame.copy()
    for destination, column in (("z_gain", "delta_r2"), ("z_predictor", predictor)):
        values = data[column].to_numpy(float)
        scale = values.std(ddof=0)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Cannot standardize {column}")
        data[destination] = (values - values.mean()) / scale
    data["all_observations"] = "all"
    specification = smf.mixedlm(
        "z_gain ~ z_predictor",
        data,
        groups=data["all_observations"],
        re_formula="0",
        vc_formula={
            "source": "0 + C(source_country)",
            "target": "0 + C(target_city)",
        },
    )
    with warnings.catch_warnings(record=True) as captured:
        model = specification.fit(reml=False, method=["lbfgs", "powell"], disp=False)
    if not model.converged:
        raise RuntimeError(f"Model failed to converge: {predictor}")
    beta = float(model.fe_params["z_predictor"])
    se = float(model.bse_fe["z_predictor"])
    if not np.isfinite(se) or se <= 0:
        raise RuntimeError(f"Invalid standard error: {predictor}")
    interval = model.conf_int().loc["z_predictor"].to_numpy(float)
    result = {
        "predictor": predictor,
        "standardized_beta": beta,
        "standard_error": se,
        "wald_p": float(2 * norm.sf(abs(beta / se))),
        "ci95_lower": float(interval[0]),
        "ci95_upper": float(interval[1]),
        "converged": bool(model.converged),
        "warnings": [str(item.message) for item in captured],
    }
    return model, result


def main() -> None:
    pairs = pd.read_csv(INPUT)
    required = {
        "source_country", "target_city", "matched", "delta_r2",
        "aef_similarity",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise RuntimeError(f"Missing transfer columns: {sorted(missing)}")
    capitals = pd.read_csv(CAPITALS)
    centroids = pd.read_csv(TARGET_CENTROIDS)
    if capitals["source_country"].duplicated().any():
        raise RuntimeError("Source-country capital table contains duplicates")
    if centroids["target"].duplicated().any():
        raise RuntimeError("Target-city centroid table contains duplicates")
    pairs = pairs.drop(columns=[
        "distance_km", "log_distance_km", "source_capital",
        "source_capital_latitude", "source_capital_longitude",
        "target_city_centroid_latitude", "target_city_centroid_longitude",
    ], errors="ignore")
    pairs = pairs.merge(
        capitals[["source_country", "source_capital", "source_capital_latitude",
                  "source_capital_longitude"]],
        on="source_country", how="left", validate="many_to_one",
    ).merge(
        centroids[["target", "target_city_centroid_latitude",
                   "target_city_centroid_longitude"]],
        on="target", how="left", validate="many_to_one",
    )
    coordinate_columns = [
        "source_capital_latitude", "source_capital_longitude",
        "target_city_centroid_latitude", "target_city_centroid_longitude",
    ]
    if pairs[coordinate_columns].isna().any().any():
        raise RuntimeError("Missing source-capital or target-city coordinates")
    pairs["distance_km"] = [
        great_circle((slat, slon), (tlat, tlon))
        for slat, slon, tlat, tlon in pairs[coordinate_columns].itertuples(index=False, name=None)
    ]
    pairs["log_distance_km"] = np.log(pairs["distance_km"])
    columns = [
        "source", "source_country", "target", "target_city", "matched",
        "r2", "aef_r2", "delta_r2", "aef_similarity", "distance_km",
        "log_distance_km", "source_capital", "source_capital_latitude",
        "source_capital_longitude", "target_city_centroid_latitude",
        "target_city_centroid_longitude",
    ]
    pairs[columns].to_csv(INPUT, index=False)
    frame = pairs.loc[~pairs["matched"].astype(bool)].copy()
    if len(frame) != 210:
        raise RuntimeError(f"Expected 210 off-diagonal pairs, found {len(frame)}")
    results = [
        fit_factor(frame, predictor)[1]
        for predictor in ("aef_similarity", "log_distance_km")
    ]
    pd.DataFrame(results).to_csv(CSV_OUTPUT, index=False)
    JSON_OUTPUT.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response": "mean transfer delta R2 across five tasks and five downstream seeds",
        "geographic_distance": "great-circle distance from source-country capital to target-city settlement-support centroid; natural log before standardization",
        "source_capitals": "World Bank country API; Pretoria used for South Africa",
        "target_city_centroids": "area centroids of the predefined settlement supports in EPSG:6933, transformed to geographic coordinates",
        "model": "separate mixed-effects models with crossed source-country and target-city random intercepts; two-sided Wald tests",
        "pairs": len(frame),
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
