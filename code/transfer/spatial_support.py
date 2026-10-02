"""Spatial support utilities for the transfer experiments."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np


ORIGIN_LEFT = -180.0
ORIGIN_BOTTOM = -84.0
RESOLUTION_DEG = 0.0008983111749910168


def union_geometries(frame: gpd.GeoDataFrame):
    if hasattr(frame.geometry, "union_all"):
        return frame.geometry.union_all()
    return frame.geometry.unary_union


def contains_xy(geometry, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    try:
        from shapely import contains_xy as shapely_contains_xy

        return np.asarray(shapely_contains_xy(geometry, lon, lat), dtype=bool)
    except ImportError:
        from shapely import vectorized

        return np.asarray(vectorized.contains(geometry, lon, lat), dtype=bool)


def grid_bounds(bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    west, south, east, north = bounds
    return (
        math.floor((west - ORIGIN_LEFT) / RESOLUTION_DEG),
        math.ceil((east - ORIGIN_LEFT) / RESOLUTION_DEG),
        math.floor((south - ORIGIN_BOTTOM) / RESOLUTION_DEG),
        math.ceil((north - ORIGIN_BOTTOM) / RESOLUTION_DEG),
    )


def split_region_rows(
    region_rows: np.ndarray,
    sp_x: np.ndarray,
    sp_y: np.ndarray,
    support_path: Path,
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    support = gpd.read_file(support_path, layer="settlement_support").to_crs("EPSG:4326")
    geometry = union_geometries(support)
    bounds = tuple(float(value) for value in support.total_bounds)
    x0, x1, y0, y1 = grid_bounds(bounds)
    inside_parts: list[np.ndarray] = []
    outside_parts: list[np.ndarray] = []

    for start in range(0, int(region_rows.size), int(chunk_rows)):
        rows = np.asarray(region_rows[start : start + chunk_rows], dtype=np.int64)
        x = np.asarray(sp_x[rows], dtype=np.int32)
        y = np.asarray(sp_y[rows], dtype=np.int32)
        candidate = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        inside = np.zeros(rows.size, dtype=bool)
        if candidate.any():
            lon = ORIGIN_LEFT + (x[candidate].astype(np.float64) + 0.5) * RESOLUTION_DEG
            lat = ORIGIN_BOTTOM + (y[candidate].astype(np.float64) + 0.5) * RESOLUTION_DEG
            inside[candidate] = contains_xy(geometry, lon, lat)
        inside_parts.append(rows[inside])
        outside_parts.append(rows[~inside])

    city_rows = np.concatenate(inside_parts).astype(np.int64, copy=False)
    region_ex_city_rows = np.concatenate(outside_parts).astype(np.int64, copy=False)
    if city_rows.size == 0:
        raise RuntimeError(f"No POI rows intersect {support_path}")
    if city_rows.size + region_ex_city_rows.size != region_rows.size:
        raise RuntimeError("City/region-ex-city filters do not partition the region filter")
    if np.intersect1d(city_rows, region_ex_city_rows, assume_unique=True).size:
        raise RuntimeError("City and region-ex-city filters overlap")
    return city_rows, region_ex_city_rows, list(bounds)
