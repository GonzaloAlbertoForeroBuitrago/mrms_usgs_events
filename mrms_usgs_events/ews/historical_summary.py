from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .common import (
    build_window_indices,
    find_site_paths,
    haversine_km,
    hours_between,
    load_meta_gauge_latlon,
    to_naive_timestamp,
)


def load_events(events_fp: Path) -> pd.DataFrame:
    events = pd.read_csv(events_fp, parse_dates=["date_peak", "start_rain", "end_rain"])
    if events.empty:
        raise ValueError(f"Events file is empty: {events_fp}")

    for col in ["date_peak", "start_rain", "end_rain"]:
        events[col] = pd.to_datetime(events[col], errors="coerce").map(to_naive_timestamp)

    events["flow_peak"] = pd.to_numeric(events.get("flow_peak", np.nan), errors="coerce")
    events = events.sort_values("date_peak").reset_index(drop=True)
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


def load_rain_zarr(zarr_fp: Path) -> dict[str, np.ndarray | pd.DatetimeIndex]:
    root = zarr.open_group(str(zarr_fp), mode="r")
    time_raw = root["time"][:]
    lat = np.asarray(root["lat"][:], dtype=np.float64)
    lon = np.asarray(root["lon"][:], dtype=np.float64)
    rain = np.asarray(root["rain"][:], dtype=np.float32)

    if np.issubdtype(time_raw.dtype, np.datetime64):
        time = pd.to_datetime(time_raw)
    elif np.issubdtype(time_raw.dtype, np.integer):
        time = pd.to_datetime(time_raw.astype("int64"), unit="ns")
    else:
        time = pd.to_datetime(time_raw.astype("U"), errors="coerce")

    time = pd.DatetimeIndex(time).map(to_naive_timestamp)
    rain = np.where(np.isfinite(rain), rain, 0.0).astype(np.float32, copy=False)
    return {"time": time, "lat": lat, "lon": lon, "rain": rain}


def build_match(events: pd.DataFrame, rain_time: pd.DatetimeIndex, rain: np.ndarray) -> pd.DataFrame:
    matched = events.copy()
    matched["prev_stage_peak_time"] = matched["date_peak"].shift(1)
    matched["effective_start_rain"] = matched[["start_rain", "prev_stage_peak_time"]].max(axis=1)
    matched["effective_start_rain"] = matched["effective_start_rain"].fillna(matched["start_rain"])
    matched["overlap_trimmed"] = matched["effective_start_rain"] > matched["start_rain"]

    r0, r1 = build_window_indices(rain_time, matched["effective_start_rain"], matched["date_peak"])
    matched["rain_window_start_idx"] = r0
    matched["rain_window_end_idx"] = r1
    matched["rain_window_n_steps"] = np.maximum(r1 - r0, 0).astype(np.int32)
    matched["window_has_positive_rain"] = [(b > a) and np.any(rain[a:b] > 0.0) for a, b in zip(r0, r1)]
    return matched


def compute_event_summary(
    *,
    site_id: str,
    matched: pd.DataFrame,
    stage_df: pd.DataFrame,
    rain_time: pd.DatetimeIndex,
    rain: np.ndarray,
    pixel_lat: np.ndarray,
    pixel_lon: np.ndarray,
    gauge_lat: float,
    gauge_lon: float,
) -> pd.DataFrame:
    stage_time = pd.DatetimeIndex(stage_df["datetime"])
    stage_vals = stage_df["Stage_ft"].to_numpy(dtype=np.float64)
    rain_time_arr = rain_time.to_numpy(dtype="datetime64[ns]")

    r0 = matched["rain_window_start_idx"].to_numpy(dtype=np.int64)
    r1 = matched["rain_window_end_idx"].to_numpy(dtype=np.int64)
    peak_times = matched["date_peak"].to_numpy(dtype="datetime64[ns]")
    start_times = matched["effective_start_rain"].to_numpy(dtype="datetime64[ns]")

    s0, _ = build_window_indices(stage_time, matched["effective_start_rain"], matched["date_peak"])
    flow_start = np.full(len(matched), np.nan, dtype=np.float64)
    valid_stage = s0 < len(stage_vals)
    clipped = np.clip(s0, 0, max(len(stage_vals) - 1, 0))
    flow_start[valid_stage] = stage_vals[clipped[valid_stage]]

    flow_peak = matched["flow_peak"].to_numpy(dtype=np.float64)
    delta_stage = flow_peak - flow_start
    event_duration_hr = hours_between(start_times, peak_times)

    rows = []
    n_pixels = rain.shape[1]
    dist_to_gauge = haversine_km(gauge_lat, gauge_lon, pixel_lat, pixel_lon)

    for i, (a, b) in enumerate(zip(r0, r1)):
        if b <= a:
            continue

        block = np.where(rain[a:b, :] > 0.0, rain[a:b, :], 0.0).astype(np.float32, copy=False)
        if not np.any(block):
            continue

        t = rain_time_arr[a:b]
        basin_hourly_sum = block.sum(axis=1, dtype=np.float64)
        event_cum = np.cumsum(basin_hourly_sum)
        event_total_acc = float(event_cum[-1])
        idx_event_acc_peak = int(np.argmax(event_cum))
        time_event_acc_to_stage_peak_hr = float(
            hours_between(
                np.array([t[idx_event_acc_peak]], dtype="datetime64[ns]"),
                np.array([peak_times[i]], dtype="datetime64[ns]"),
            )[0]
        )

        pixel_acc_total = block.sum(axis=0, dtype=np.float64)
        max_pixel_acc_idx = int(np.argmax(pixel_acc_total))
        max_pixel_acc = float(pixel_acc_total[max_pixel_acc_idx])

        flat_idx = int(np.argmax(block))
        idx_time_max_pixel, idx_pixel_inst = np.unravel_index(flat_idx, block.shape)
        max_pixel_rain = float(block[idx_time_max_pixel, idx_pixel_inst])
        time_max_pixel_to_stage_peak_hr = float(
            hours_between(
                np.array([t[idx_time_max_pixel]], dtype="datetime64[ns]"),
                np.array([peak_times[i]], dtype="datetime64[ns]"),
            )[0]
        )

        pixel_cum = np.cumsum(block[:, max_pixel_acc_idx], dtype=np.float64)
        idx_pixel_acc_peak_time = int(np.argmax(pixel_cum))
        time_max_pixel_acc_to_stage_peak_hr = float(
            hours_between(
                np.array([t[idx_pixel_acc_peak_time]], dtype="datetime64[ns]"),
                np.array([peak_times[i]], dtype="datetime64[ns]"),
            )[0]
        )

        pixel_acc_contribution_pct = 100.0 * max_pixel_acc / event_total_acc if event_total_acc > 0 else np.nan
        stage_contribution_proxy_pct = pixel_acc_contribution_pct if np.isfinite(delta_stage[i]) and delta_stage[i] > 0 else np.nan

        rows.append({
            "site_id": site_id,
            "event_id": int(matched.loc[i, "event_id"]),
            "date_peak": pd.Timestamp(peak_times[i]),
            "effective_start_rain": pd.Timestamp(start_times[i]),
            "event_duration_hr": float(event_duration_hr[i]),
            "flow_start": float(flow_start[i]) if np.isfinite(flow_start[i]) else np.nan,
            "flow_peak": float(flow_peak[i]) if np.isfinite(flow_peak[i]) else np.nan,
            "delta_stage": float(delta_stage[i]) if np.isfinite(delta_stage[i]) else np.nan,
            "event_total_acc": event_total_acc,
            "event_max_hourly_basin_sum": float(np.max(basin_hourly_sum)),
            "time_event_acc_to_stage_peak_hr": time_event_acc_to_stage_peak_hr,
            "max_pixel_acc": max_pixel_acc,
            "max_pixel_acc_pixel_id": int(max_pixel_acc_idx),
            "max_pixel_acc_lat": float(pixel_lat[max_pixel_acc_idx]),
            "max_pixel_acc_lon": float(pixel_lon[max_pixel_acc_idx]),
            "max_pixel_acc_distance_to_gauge_km": float(dist_to_gauge[max_pixel_acc_idx]),
            "time_max_pixel_acc_to_stage_peak_hr": time_max_pixel_acc_to_stage_peak_hr,
            "max_pixel_rain": max_pixel_rain,
            "max_pixel_rain_pixel_id": int(idx_pixel_inst),
            "max_pixel_rain_lat": float(pixel_lat[idx_pixel_inst]),
            "max_pixel_rain_lon": float(pixel_lon[idx_pixel_inst]),
            "max_pixel_rain_distance_to_gauge_km": float(dist_to_gauge[idx_pixel_inst]),
            "time_max_pixel_rain_to_stage_peak_hr": time_max_pixel_to_stage_peak_hr,
            "time_diff_pixel_rain_vs_event_acc_hr": time_max_pixel_to_stage_peak_hr - time_event_acc_to_stage_peak_hr,
            "time_diff_pixel_acc_vs_event_acc_hr": time_max_pixel_acc_to_stage_peak_hr - time_event_acc_to_stage_peak_hr,
            "pixel_acc_contribution_pct": float(pixel_acc_contribution_pct),
            "stage_contribution_proxy_pct": float(stage_contribution_proxy_pct),
            "n_pixels": int(n_pixels),
            "n_positive_pixels": int(np.sum(pixel_acc_total > 0.0)),
        })

    return pd.DataFrame(rows)


def build_site_historical_summary(*, base_dir: Path, site_id: str, out_dir: Path, overwrite: bool = False) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{site_id}_historical_event_summary.parquet"
    if out_fp.exists() and not overwrite:
        return out_fp

    paths = find_site_paths(base_dir, site_id)
    events = load_events(paths["events_fp"])
    stage = load_stage(paths["stage_fp"])
    gauge_lat, gauge_lon = load_meta_gauge_latlon(paths["meta_fp"])
    rain_data = load_rain_zarr(paths["zarr_fp"])
    matched = build_match(events, rain_data["time"], rain_data["rain"])

    df = compute_event_summary(
        site_id=site_id,
        matched=matched,
        stage_df=stage,
        rain_time=rain_data["time"],
        rain=rain_data["rain"],
        pixel_lat=rain_data["lat"],
        pixel_lon=rain_data["lon"],
        gauge_lat=gauge_lat,
        gauge_lon=gauge_lon,
    )

    if df.empty:
        return None

    df.to_parquet(out_fp, index=False)
    return out_fp


def build_many_historical_summaries(*, base_dir: Path, mask_input: Path, out_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    m = pd.read_csv(mask_input, sep="\t", dtype={"site_id": str})
    sites = m["site_id"].astype(str).tolist()

    ok = 0
    fail = 0
    for i, site_id in enumerate(sites, start=1):
        try:
            out = build_site_historical_summary(base_dir=base_dir, site_id=site_id, out_dir=out_dir, overwrite=overwrite)
            ok += int(out is not None)
            print(f"[{i}/{len(sites)}] {site_id} OK")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(sites)}] {site_id} ERROR {type(e).__name__}: {e}")

    return ok, fail
