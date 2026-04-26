from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def to_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def find_one(root: Path, pattern: str) -> Path | None:
    return next(root.rglob(pattern), None)


def find_site_paths(base: Path, site_id: str) -> dict[str, Path]:
    roots = {
        "events": base / "events",
        "stage": base / "stage_parquet",
        "rain": base / "rain_zarr",
        "meta": base / "site_meta",
    }
    paths = {
        "events_fp": find_one(roots["events"], f"{site_id}_rain_windows.csv"),
        "stage_fp": find_one(roots["stage"], f"{site_id}.parquet"),
        "zarr_fp": find_one(roots["rain"], f"{site_id}.zarr"),
        "meta_fp": find_one(roots["meta"], f"{site_id}_monitoring_location.json"),
    }
    missing = [k for k, v in paths.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing files for {site_id}: {missing}")
    return {k: Path(v) for k, v in paths.items()}


def build_window_indices(time_index: pd.DatetimeIndex, starts, ends) -> tuple[np.ndarray, np.ndarray]:
    arr = time_index.to_numpy(dtype="datetime64[ns]")
    starts_arr = pd.to_datetime(starts).to_numpy(dtype="datetime64[ns]")
    ends_arr = pd.to_datetime(ends).to_numpy(dtype="datetime64[ns]")
    i0 = arr.searchsorted(starts_arr, side="left")
    i1 = arr.searchsorted(ends_arr, side="right")
    return i0.astype(np.int64), i1.astype(np.int64)


def hours_between(t0, t1) -> np.ndarray:
    return (t1 - t0).astype("timedelta64[s]").astype(np.float64) / 3600.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def load_meta_gauge_latlon(meta_fp: Path) -> tuple[float, float]:
    meta = json.loads(meta_fp.read_text(encoding="utf-8"))
    lon, lat = meta["geometry"]["coordinates"][:2]
    return float(lat), float(lon)
