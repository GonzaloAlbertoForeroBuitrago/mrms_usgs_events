from __future__ import annotations

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import zarr

from .common import (
    build_window_indices,
    find_site_paths,
    hours_between,
    load_meta_gauge_latlon,
    to_naive_timestamp,
)

STRONG_RAIN_MM_H = 3.0


def load_events(events_fp: Path) -> pd.DataFrame:
    events = pd.read_csv(events_fp, parse_dates=["date_peak", "start_rain", "end_rain"])

    if events.empty:
        raise ValueError(f"Events file is empty: {events_fp}")

    for col in ["date_peak", "start_rain", "end_rain"]:
        events[col] = pd.to_datetime(events[col], errors="coerce").map(to_naive_timestamp)

    events["flow_peak"] = pd.to_numeric(events.get("flow_peak", np.nan), errors="coerce")
    events = events.dropna(subset=["date_peak", "start_rain"]).sort_values("date_peak").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype=np.int64)

    return events


def load_stage(stage_fp: Path) -> pd.DataFrame:
    stage = pd.read_parquet(stage_fp, columns=["datetime", "Stage_ft"])
    stage["datetime"] = pd.to_datetime(stage["datetime"], errors="coerce").map(to_naive_timestamp)
    stage["Stage_ft"] = pd.to_numeric(stage["Stage_ft"], errors="coerce")
    stage = stage.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if stage.empty:
        raise ValueError(f"Stage parquet has no valid rows: {stage_fp}")

    return stage


def load_rain_zarr(zarr_fp: Path) -> dict:
    root = zarr.open_group(str(zarr_fp), mode="r")

    time_raw = root["time"][:]
    if np.issubdtype(time_raw.dtype, np.datetime64):
        time = pd.to_datetime(time_raw)
    elif np.issubdtype(time_raw.dtype, np.integer):
        time = pd.to_datetime(time_raw.astype("int64"), unit="ns")
    else:
        time = pd.to_datetime(time_raw.astype("U"), errors="coerce")

    time = pd.DatetimeIndex(time).map(to_naive_timestamp)

    return {
        "root": root,
        "time": pd.DatetimeIndex(time),
        "lat": np.asarray(root["lat"][:], dtype=np.float64),
        "lon": np.asarray(root["lon"][:], dtype=np.float64),
        "rain": root["rain"],
    }


def build_matched_events(
    events: pd.DataFrame,
    stage: pd.DataFrame,
    rain_time: pd.DatetimeIndex,
) -> pd.DataFrame:
    matched = events.copy()

    matched["prev_stage_peak_time"] = matched["date_peak"].shift(1)
    matched["effective_start_rain"] = matched[["start_rain", "prev_stage_peak_time"]].max(axis=1)
    matched["effective_start_rain"] = matched["effective_start_rain"].fillna(matched["start_rain"])
    matched["overlap_trimmed"] = matched["effective_start_rain"] > matched["start_rain"]

    rain_start = pd.to_datetime(matched["effective_start_rain"]).dt.floor("h")
    rain_end = pd.to_datetime(matched["date_peak"]).dt.ceil("h")


    r0, r1 = build_window_indices(
        rain_time,
        rain_start,
        rain_end,
    )

    matched["rain_window_start_idx"] = r0.astype(np.int64)
    matched["rain_window_end_idx"] = r1.astype(np.int64)
    matched["rain_window_n_steps"] = np.maximum(r1 - r0, 0).astype(np.int32)

    stage_time = pd.DatetimeIndex(stage["datetime"])
    stage_vals = stage["Stage_ft"].to_numpy(dtype=np.float64)

    s0, _ = build_window_indices(
        stage_time,
        matched["effective_start_rain"],
        matched["date_peak"],
    )

    flow_start = np.full(len(matched), np.nan, dtype=np.float64)
    valid = s0 < len(stage_vals)

    if len(stage_vals):
        clipped = np.clip(s0, 0, len(stage_vals) - 1)
        flow_start[valid] = stage_vals[clipped[valid]]

    matched["flow_start"] = flow_start
    matched["delta_water_stage"] = matched["flow_peak"].to_numpy(dtype=np.float64) - flow_start

    matched["event_duration_hr"] = hours_between(
        matched["effective_start_rain"].to_numpy(dtype="datetime64[ns]"),
        matched["date_peak"].to_numpy(dtype="datetime64[ns]"),
    )

    p50 = matched["delta_water_stage"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.50)
    matched["delta_water_stage_p50"] = float(p50) if np.isfinite(p50) else np.nan
    matched["is_stage_response_p50"] = matched["delta_water_stage"] >= matched["delta_water_stage_p50"]

    return matched


def compute_site_historical_tables(
    *,
    state: str,
    site_id: str,
    matched: pd.DataFrame,
    rain_time: pd.DatetimeIndex,
    rain_array,
    pixel_lat: np.ndarray,
    pixel_lon: np.ndarray,
    gauge_lat: float,
    gauge_lon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rain_time_arr = rain_time.to_numpy(dtype="datetime64[ns]")

    basin_rows = []
    pixel_chunks = []

    for _, ev in matched.iterrows():
        a = int(ev["rain_window_start_idx"])
        b = int(ev["rain_window_end_idx"])

        if b <= a:
            continue

        block = np.asarray(rain_array[a:b, :], dtype=np.float32)
        block = np.where(np.isfinite(block) & (block > 0), block, 0.0).astype(np.float32, copy=False)

        if not np.any(block):
            continue

        t = rain_time_arr[a:b]
        peak_time = np.datetime64(pd.Timestamp(ev["date_peak"]).to_datetime64())

        basin_hourly_sum = block.sum(axis=1, dtype=np.float64)
        basin_accumulation = float(basin_hourly_sum.sum())

        pixel_accumulation = block.sum(axis=0, dtype=np.float64)
        pixel_value = block.max(axis=0)

        pixel_peak_idx = np.argmax(block, axis=0)
        pixel_peak_time = t[pixel_peak_idx]

        time_to_stage_peak = hours_between(pixel_peak_time, np.full(len(pixel_peak_time), peak_time))

        cumulative_basin = np.cumsum(basin_hourly_sum)
        basin_acc_peak_idx = int(np.argmax(cumulative_basin))
        basin_acc_peak_time = t[basin_acc_peak_idx]

        time_to_rain_peak_accumulation = float(
            hours_between(
                np.array([basin_acc_peak_time], dtype="datetime64[ns]"),
                np.array([peak_time], dtype="datetime64[ns]"),
            )[0]
        )

        positive = pixel_accumulation > 0
        strong = pixel_value >= STRONG_RAIN_MM_H

        n_pixels = int(block.shape[1])
        n_positive_pixels = int(np.count_nonzero(positive))
        n_strong_pixels = int(np.count_nonzero(strong))

        basin_rows.append(
            {
                "state": state,
                "site_id": str(site_id),
                "event_id": int(ev["event_id"]),
                "date_peak": pd.Timestamp(ev["date_peak"]),
                "event_start": pd.Timestamp(ev["effective_start_rain"]),
                "event_end": pd.Timestamp(ev["date_peak"]),
                "event_duration_hr": float(ev["event_duration_hr"]),
                "flow_start": float(ev["flow_start"]) if np.isfinite(ev["flow_start"]) else np.nan,
                "flow_peak": float(ev["flow_peak"]) if np.isfinite(ev["flow_peak"]) else np.nan,
                "delta_water_stage": float(ev["delta_water_stage"]) if np.isfinite(ev["delta_water_stage"]) else np.nan,
                "delta_water_stage_p50": float(ev["delta_water_stage_p50"]) if np.isfinite(ev["delta_water_stage_p50"]) else np.nan,
                "is_stage_response_p50": bool(ev["is_stage_response_p50"]),
                "basin_accumulation": basin_accumulation,
                "basin_max_hourly_accumulation": float(basin_hourly_sum.max()),
                "time_to_rain_peak_accumulation_hr": time_to_rain_peak_accumulation,
                "max_pixel_value": float(pixel_value.max()),
                "max_pixel_accumulation": float(pixel_accumulation.max()),
                "n_pixels": n_pixels,
                "n_positive_pixels": n_positive_pixels,
                "n_strong_pixels": n_strong_pixels,
                "strong_rain_threshold_mm_h": STRONG_RAIN_MM_H,
                "gauge_lat": float(gauge_lat),
                "gauge_lon": float(gauge_lon),
            }
        )

        keep = strong # to keep > 0 change for positive

        if not np.any(keep):
            continue

        pixel_id_basin = np.flatnonzero(keep)

        pixel_df = pd.DataFrame(
            {
                "state": state,
                "site_id": str(site_id),
                "event_id": int(ev["event_id"]),
                "date_peak": pd.Timestamp(ev["date_peak"]),
                "event_start": pd.Timestamp(ev["effective_start_rain"]),
                "event_end": pd.Timestamp(ev["date_peak"]),
                "pixel_id_basin": pixel_id_basin.astype(np.int32),
                "lat": pixel_lat[pixel_id_basin].astype(np.float64),
                "lon": pixel_lon[pixel_id_basin].astype(np.float64),
                "pixel_value": pixel_value[pixel_id_basin].astype(np.float32),
                "pixel_accumulation": pixel_accumulation[pixel_id_basin].astype(np.float32),
                "basin_accumulation": np.float32(basin_accumulation),
                "delta_water_stage": np.float32(ev["delta_water_stage"]) if np.isfinite(ev["delta_water_stage"]) else np.nan,
                "delta_water_stage_p50": np.float32(ev["delta_water_stage_p50"]) if np.isfinite(ev["delta_water_stage_p50"]) else np.nan,
                "is_stage_response_p50": bool(ev["is_stage_response_p50"]),
                "time_to_rain_peak_accumulation_hr": np.float32(time_to_rain_peak_accumulation),
                "time_to_stage_peak_hr": time_to_stage_peak[pixel_id_basin].astype(np.float32),
                "is_strong_pixel": strong[pixel_id_basin],
                "strong_rain_threshold_mm_h": np.float32(STRONG_RAIN_MM_H),
            }
        )

        pixel_chunks.append(pixel_df)

        del block, pixel_df
        gc.collect()

    basin_df = pd.DataFrame(basin_rows)
    pixel_df = pd.concat(pixel_chunks, ignore_index=True) if pixel_chunks else pd.DataFrame()

    return basin_df, pixel_df


def build_site_historical_summary(
    *,
    base_dir: Path,
    site_id: str,
    out_dir: Path,
    state: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path] | None:
    out_dir = Path(out_dir)
    basin_out_dir = out_dir / "basin_event_history"
    pixel_out_dir = out_dir / "pixel_event_history"

    basin_out_dir.mkdir(parents=True, exist_ok=True)
    pixel_out_dir.mkdir(parents=True, exist_ok=True)

    basin_fp = basin_out_dir / f"{site_id}_basin_event_history.parquet"
    pixel_fp = pixel_out_dir / f"{site_id}_pixel_event_history.parquet"

    if basin_fp.exists() and pixel_fp.exists() and not overwrite:
        return {"basin": basin_fp, "pixel": pixel_fp}

    paths = find_site_paths(base_dir, site_id)

    if state is None:
        try:
            state = paths["zarr_fp"].parts[-4]
        except Exception:
            state = "UNKNOWN"

    state = str(state).upper()

    events = load_events(paths["events_fp"])
    stage = load_stage(paths["stage_fp"])
    gauge_lat, gauge_lon = load_meta_gauge_latlon(paths["meta_fp"])

    rain_data = load_rain_zarr(paths["zarr_fp"])
    matched = build_matched_events(events, stage, rain_data["time"])

    basin_df, pixel_df = compute_site_historical_tables(
        state=state,
        site_id=site_id,
        matched=matched,
        rain_time=rain_data["time"],
        rain_array=rain_data["rain"],
        pixel_lat=rain_data["lat"],
        pixel_lon=rain_data["lon"],
        gauge_lat=gauge_lat,
        gauge_lon=gauge_lon,
    )

    if basin_df.empty:
        return None

    basin_df.to_parquet(basin_fp, index=False)

    if not pixel_df.empty:
        pixel_df.to_parquet(pixel_fp, index=False)

    return {"basin": basin_fp, "pixel": pixel_fp if pixel_fp.exists() else None}


def build_many_historical_summaries(
    *,
    base_dir: Path,
    mask_input: Path,
    out_dir: Path,
    overwrite: bool = False,
    state: str | None = None,
) -> tuple[int, int]:
    m = pd.read_csv(mask_input, sep="\t", dtype={"site_id": str})

    if state is not None and "state" in m.columns:
        m = m[m["state"].astype(str).str.upper() == state.upper()].copy()

    ok = 0
    fail = 0

    for i, row in enumerate(m.itertuples(index=False), start=1):
        site_id = str(getattr(row, "site_id"))
        site_state = str(getattr(row, "state", state or "UNKNOWN")).upper()

        try:
            out = build_site_historical_summary(
                base_dir=base_dir,
                site_id=site_id,
                state=site_state,
                out_dir=out_dir,
                overwrite=overwrite,
            )
            ok += int(out is not None)
            print(f"[{i}/{len(m)}] {site_state} {site_id} OK")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(m)}] {site_state} {site_id} ERROR {type(e).__name__}: {e}")

    return ok, fail