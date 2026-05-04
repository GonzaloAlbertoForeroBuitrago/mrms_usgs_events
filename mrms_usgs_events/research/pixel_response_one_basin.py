from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


BASE_DIR = Path(os.environ.get("BASE_DIR", "/data/repository_code/unified_data"))
SITE_ID = str(os.environ["SITE_ID"]).zfill(8)

OUT_DIR = Path(
    os.environ.get(
        "PIXEL_RESPONSE_OUT_DIR",
        BASE_DIR / "research" / "hydrologic_drivers" / "pixel_response" / "outputs",
    )
)

EVENT_TIMING_DIR = BASE_DIR / "research" / "hydrologic_drivers" / "pixel_response" / "event_timing"

MIN_RAIN_MMHR = float(os.environ.get("MIN_RAIN_MMHR", "7.5"))
MIN_STAGE_RISE_FT = float(os.environ.get("MIN_STAGE_RISE_FT", "0.0"))

OUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_TIMING_DIR.mkdir(parents=True, exist_ok=True)


def find_site_file(base: Path, folder: str, site_id: str, suffix: str) -> Path:
    hits = list((base / folder).rglob(f"{site_id}{suffix}"))
    if not hits:
        raise FileNotFoundError(f"Missing {folder} file for {site_id} suffix={suffix}")
    return hits[0]


def read_zarr_time(root) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(root["time"][:]))


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    if x.size == 0 or np.all(~np.isfinite(x)):
        return np.zeros_like(x)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def classify_response(score, lag, freq):
    if not np.isfinite(score) or score <= 0 or freq <= 0:
        return "NO_SIGNAL"
    if lag <= 2.0 and score >= 0.60:
        return "HIGH_FAST_RESPONSE"
    if lag <= 3.0 and score >= 0.30:
        return "MEDIUM_FAST_RESPONSE"
    if lag > 3.0 and score >= 0.20:
        return "SLOW_RESPONSE"
    if score > 0:
        return "NO_RESPONSE"
    return "NO_SIGNAL"


def intensity_class(v: float) -> str:
    if not np.isfinite(v):
        return "NO_SIGNAL"
    if v < MIN_RAIN_MMHR:
        return "NO_SIGNAL"
    if v < 15:
        return "LOW"
    if v < 30:
        return "MEDIUM"
    if v < 50:
        return "HIGH"
    return "EXTREME"


def points_geojson(df: pd.DataFrame) -> dict:
    features = []
    for _, r in df.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]):
            continue
        props = {}
        for k, v in r.drop(["lat", "lon"]).items():
            if pd.isna(v):
                props[k] = None
            elif hasattr(v, "item"):
                props[k] = v.item()
            else:
                props[k] = v
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    history_fp = BASE_DIR / "ews_history" / f"{SITE_ID}_historical_event_summary.parquet"
    zarr_fp = find_site_file(BASE_DIR, "rain_zarr", SITE_ID, ".zarr")
    stage_fp = find_site_file(BASE_DIR, "stage_parquet", SITE_ID, ".parquet")

    print("=" * 100)
    print("PIXEL RESPONSE ANALYSIS WITH DYNAMIC TIMING")
    print("=" * 100)
    print(f"site_id          : {SITE_ID}")
    print(f"history          : {history_fp}")
    print(f"zarr             : {zarr_fp}")
    print(f"stage            : {stage_fp}")
    print(f"min_rain_mmhr    : {MIN_RAIN_MMHR}")
    print(f"min_stage_rise_ft: {MIN_STAGE_RISE_FT}")

    if not history_fp.exists():
        raise FileNotFoundError(history_fp)

    hist = pd.read_parquet(history_fp)
    hist["date_peak"] = pd.to_datetime(hist["date_peak"])
    hist["effective_start_rain"] = pd.to_datetime(hist["effective_start_rain"])

    root = zarr.open_group(str(zarr_fp), mode="r")
    rain = np.asarray(root["rain"][:], dtype="float32")
    times = read_zarr_time(root)
    lat = np.asarray(root["lat"][:], dtype="float32")
    lon = np.asarray(root["lon"][:], dtype="float32")

    n_time, n_pixels = rain.shape

    hist = hist.copy()
    hist = hist[np.isfinite(hist["delta_stage"])]
    hist = hist[hist["delta_stage"] > MIN_STAGE_RISE_FT]

    event_rows = []
    valid_events = 0

    for k, ev in hist.iterrows():
        event_id = int(ev["event_id"]) if "event_id" in ev and pd.notna(ev["event_id"]) else int(k + 1)
        start = pd.to_datetime(ev["effective_start_rain"])
        stage_peak_time = pd.to_datetime(ev["date_peak"])
        delta_stage = float(ev["delta_stage"])

        mask = (times >= start) & (times <= stage_peak_time)
        idx = np.where(mask)[0]

        if idx.size < 2:
            continue

        rain_win = rain[idx, :]
        time_win = times[idx]

        rain_win = np.where(np.isfinite(rain_win), rain_win, 0.0)

        basin_hourly_sum = rain_win.sum(axis=1)
        if np.nanmax(basin_hourly_sum) <= 0:
            continue

        basin_peak_i = int(np.nanargmax(basin_hourly_sum))
        basin_acc_peak_time = pd.Timestamp(time_win[basin_peak_i])
        basin_acc_peak_mm = float(basin_hourly_sum[basin_peak_i])
        basin_total_acc_mm = float(rain_win.sum())

        basin_to_stage_lag_hr = (stage_peak_time - basin_acc_peak_time).total_seconds() / 3600.0

        pixel_peak = np.nanmax(rain_win, axis=0)
        pixel_acc = np.nansum(rain_win, axis=0)
        pixel_peak_i = np.nanargmax(rain_win, axis=0)

        active = (pixel_peak >= MIN_RAIN_MMHR) & (pixel_acc > 0)

        if active.sum() == 0:
            continue

        valid_events += 1

        event_total_acc = basin_total_acc_mm if basin_total_acc_mm > 0 else np.nan

        for p in np.where(active)[0]:
            peak_time = pd.Timestamp(time_win[int(pixel_peak_i[p])])

            pixel_to_basin_lag_hr = (basin_acc_peak_time - peak_time).total_seconds() / 3600.0
            pixel_to_stage_lag_hr = (stage_peak_time - peak_time).total_seconds() / 3600.0

            contribution_pct = (float(pixel_acc[p]) / event_total_acc * 100.0) if event_total_acc and np.isfinite(event_total_acc) else np.nan
            attributed_stage_ft = delta_stage * (contribution_pct / 100.0) if np.isfinite(contribution_pct) else np.nan

            event_rows.append({
                "site_id": SITE_ID,
                "event_id": event_id,
                "date_peak": stage_peak_time,
                "effective_start_rain": start,
                "pixel_id": int(p),
                "lat": float(lat[p]),
                "lon": float(lon[p]),
                "pixel_peak_time": peak_time,
                "basin_acc_peak_time": basin_acc_peak_time,
                "stage_peak_time": stage_peak_time,
                "pixel_peak_mm_h": float(pixel_peak[p]),
                "pixel_acc_mm": float(pixel_acc[p]),
                "basin_acc_peak_mm": basin_acc_peak_mm,
                "basin_total_acc_mm": basin_total_acc_mm,
                "delta_stage_ft": delta_stage,
                "contribution_pct": contribution_pct,
                "attributed_stage_ft": attributed_stage_ft,
                "pixel_to_basin_lag_hr": pixel_to_basin_lag_hr,
                "basin_to_stage_lag_hr": basin_to_stage_lag_hr,
                "pixel_to_stage_lag_hr": pixel_to_stage_lag_hr,
                "intensity_class": intensity_class(float(pixel_peak[p])),
            })

        print(
            f"[EVENT] {stage_peak_time} active_pixels={active.sum()} "
            f"basin_peak={basin_acc_peak_mm:.2f} delta_stage={delta_stage:.2f}"
        )

    event_df = pd.DataFrame(event_rows)

    timing_fp = EVENT_TIMING_DIR / f"{SITE_ID}_pixel_event_timing.parquet"
    event_df.to_parquet(timing_fp, index=False)

    if event_df.empty:
        print("No valid pixel-event timing rows. Writing empty summaries.")
        summary = pd.DataFrame({
            "site_id": SITE_ID,
            "pixel_id": np.arange(n_pixels),
            "lat": lat,
            "lon": lon,
            "n_events_contributing": 0,
            "event_frequency": 0.0,
            "median_contribution_pct": 0.0,
            "p75_contribution_pct": 0.0,
            "sum_attributed_stage_ft": 0.0,
            "median_attributed_stage_ft": 0.0,
            "median_pixel_acc_mm": 0.0,
            "median_pixel_peak_mm_h": 0.0,
            "median_best_lag_hr": np.nan,
            "median_best_corr": np.nan,
            "median_pixel_to_basin_lag_hr": np.nan,
            "median_basin_to_stage_lag_hr": np.nan,
            "median_pixel_to_stage_lag_hr": np.nan,
            "hydrologic_influence_score": 0.0,
            "response_class": "NO_SIGNAL",
        })
    else:
        g = event_df.groupby("pixel_id")

        summary = g.agg(
            n_events_contributing=("event_id", "nunique"),
            median_contribution_pct=("contribution_pct", "median"),
            p75_contribution_pct=("contribution_pct", lambda s: np.nanpercentile(s, 75)),
            sum_attributed_stage_ft=("attributed_stage_ft", "sum"),
            median_attributed_stage_ft=("attributed_stage_ft", "median"),
            median_pixel_acc_mm=("pixel_acc_mm", "median"),
            median_pixel_peak_mm_h=("pixel_peak_mm_h", "median"),
            median_best_lag_hr=("pixel_to_stage_lag_hr", "median"),
            median_pixel_to_basin_lag_hr=("pixel_to_basin_lag_hr", "median"),
            median_basin_to_stage_lag_hr=("basin_to_stage_lag_hr", "median"),
            median_pixel_to_stage_lag_hr=("pixel_to_stage_lag_hr", "median"),
            median_delta_stage_ft=("delta_stage_ft", "median"),
        ).reset_index()

        # correlation per pixel: pixel_acc vs delta_stage
        corr_rows = []
        for p, sub in event_df.groupby("pixel_id"):
            if len(sub) >= 3 and sub["pixel_acc_mm"].std() > 0 and sub["delta_stage_ft"].std() > 0:
                corr = float(np.corrcoef(sub["pixel_acc_mm"], sub["delta_stage_ft"])[0, 1])
            else:
                corr = np.nan
            corr_rows.append({"pixel_id": p, "median_best_corr": corr})

        corr_df = pd.DataFrame(corr_rows)
        summary = summary.merge(corr_df, on="pixel_id", how="left")

        all_pixels = pd.DataFrame({
            "pixel_id": np.arange(n_pixels, dtype=int),
            "lat": lat,
            "lon": lon,
        })

        summary = all_pixels.merge(summary, on="pixel_id", how="left")
        summary["site_id"] = SITE_ID

        for c in [
            "n_events_contributing",
            "median_contribution_pct",
            "p75_contribution_pct",
            "sum_attributed_stage_ft",
            "median_attributed_stage_ft",
            "median_pixel_acc_mm",
            "median_pixel_peak_mm_h",
        ]:
            summary[c] = summary[c].fillna(0)

        summary["event_frequency"] = summary["n_events_contributing"] / max(valid_events, 1)

        score = (
            0.25 * norm01(summary["median_contribution_pct"].to_numpy())
            + 0.20 * norm01(summary["p75_contribution_pct"].to_numpy())
            + 0.20 * norm01(summary["sum_attributed_stage_ft"].to_numpy())
            + 0.15 * norm01(summary["event_frequency"].to_numpy())
            + 0.10 * norm01(summary["median_pixel_peak_mm_h"].to_numpy())
            + 0.10 * norm01(np.clip(summary["median_best_corr"].fillna(0).to_numpy(), 0, None))
        )

        summary["hydrologic_influence_score"] = score
        summary["response_class"] = [
            classify_response(s, lag, f)
            for s, lag, f in zip(
                summary["hydrologic_influence_score"],
                summary["median_pixel_to_stage_lag_hr"],
                summary["event_frequency"],
            )
        ]

        ordered = [
            "site_id",
            "pixel_id",
            "lat",
            "lon",
            "n_events_contributing",
            "event_frequency",
            "median_contribution_pct",
            "p75_contribution_pct",
            "sum_attributed_stage_ft",
            "median_attributed_stage_ft",
            "median_pixel_acc_mm",
            "median_pixel_peak_mm_h",
            "median_best_lag_hr",
            "median_best_corr",
            "median_pixel_to_basin_lag_hr",
            "median_basin_to_stage_lag_hr",
            "median_pixel_to_stage_lag_hr",
            "hydrologic_influence_score",
            "response_class",
        ]

        summary = summary[ordered]

    out_parquet = OUT_DIR / f"{SITE_ID}_pixel_response_summary.parquet"
    out_csv = OUT_DIR / f"{SITE_ID}_pixel_response_summary.csv"
    out_geojson = OUT_DIR / f"{SITE_ID}_pixel_response_summary.geojson"

    summary.to_parquet(out_parquet, index=False)
    summary.to_csv(out_csv, index=False)
    out_geojson.write_text(json.dumps(points_geojson(summary), indent=2), encoding="utf-8")

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"valid_events : {valid_events}")
    print(f"event rows   : {len(event_df)}")
    print(f"summary rows : {len(summary)}")
    print(f"timing       : {timing_fp}")
    print(f"parquet      : {out_parquet}")
    print(f"csv          : {out_csv}")
    print(f"geojson      : {out_geojson}")
    print()
    if "response_class" in summary:
        print(summary["response_class"].value_counts(dropna=False).to_string())
        print()
        print("Top pixels:")
        print(
            summary.sort_values("hydrologic_influence_score", ascending=False)
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()