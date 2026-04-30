from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


STRONG_RAIN_MM_H = 7.5
MIN_EVENTS = 3


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _q(s: pd.Series, p: float) -> float:
    x = _num(s).replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.quantile(p)) if len(x) else np.nan


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _num(a)
    b = _num(b)
    return pd.Series(
        np.where((b > 0) & np.isfinite(b), a / b, np.nan),
        index=a.index,
    )


def _read_zarr_time(root) -> pd.DatetimeIndex:
    time_raw = root["time"][:]

    if np.issubdtype(time_raw.dtype, np.datetime64):
        time = pd.to_datetime(time_raw)
    elif np.issubdtype(time_raw.dtype, np.integer):
        time = pd.to_datetime(time_raw.astype("int64"), unit="ns")
    else:
        time = pd.to_datetime(time_raw.astype(str), errors="coerce")

    return pd.DatetimeIndex(time)


def _find_zarr_files(base_dir: Path) -> dict[str, Path]:
    out = {}

    for fp in sorted((base_dir / "rain_zarr").rglob("*.zarr")):
        site_id = fp.stem
        if site_id not in out:
            out[site_id] = fp

    return out


def _event_strong_rain_features_from_zarr(
    *,
    rain_array,
    rain_time: pd.DatetimeIndex,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> dict:
    mask = (rain_time >= start_time) & (rain_time <= end_time)
    idx = np.flatnonzero(mask)

    if idx.size == 0:
        return {
            "strong_max_pixel_rain": np.nan,
            "strong_event_total_acc": np.nan,
            "strong_max_pixel_acc": np.nan,
            "strong_event_max_hourly_basin_sum": np.nan,
            "strong_n_positive_pixels": 0,
        }

    # Read only the event window from Zarr, not the full station Zarr.
    start_i = int(idx[0])
    end_i = int(idx[-1]) + 1

    vals = np.asarray(rain_array[start_i:end_i, :], dtype=np.float32)
    vals = np.where(np.isfinite(vals), vals, 0.0)

    strong = np.where(vals >= STRONG_RAIN_MM_H, vals, 0.0)

    acc_by_pixel = strong.sum(axis=0, dtype=np.float64)
    hourly_basin_sum = strong.sum(axis=1, dtype=np.float64)

    out = {
        "strong_max_pixel_rain": float(strong.max()) if strong.size else np.nan,
        "strong_event_total_acc": float(strong.sum(dtype=np.float64)),
        "strong_max_pixel_acc": float(acc_by_pixel.max()) if acc_by_pixel.size else np.nan,
        "strong_event_max_hourly_basin_sum": float(hourly_basin_sum.max()) if hourly_basin_sum.size else np.nan,
        "strong_n_positive_pixels": int(np.count_nonzero(acc_by_pixel > 0)),
    }

    del vals, strong, acc_by_pixel, hourly_basin_sum
    return out


def fit_one_summary(
    fp: Path,
    *,
    base_dir: Path,
    zarr_index: dict[str, Path],
) -> dict:
    df = pd.read_parquet(fp).copy()

    if df.empty:
        raise ValueError(f"Empty summary: {fp}")

    site_id = str(df["site_id"].iloc[0])

    required = [
        "date_peak",
        "effective_start_rain",
        "delta_stage",
        "time_max_pixel_rain_to_stage_peak_hr",
        "time_max_pixel_acc_to_stage_peak_hr",
        "max_pixel_rain_distance_to_gauge_km",
        "max_pixel_acc_distance_to_gauge_km",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{site_id}: missing columns: {missing}")

    df["date_peak"] = pd.to_datetime(df["date_peak"])
    df["effective_start_rain"] = pd.to_datetime(df["effective_start_rain"])
    df["delta_stage"] = _num(df["delta_stage"])

    delta_stage_target_p50 = _q(df["delta_stage"], 0.50)
    df["target_stage_response_p50"] = df["delta_stage"] >= delta_stage_target_p50

    zarr_fp = zarr_index.get(site_id)
    if zarr_fp is None:
        raise FileNotFoundError(f"No zarr found for site_id={site_id}")

    root = zarr.open_group(str(zarr_fp), mode="r")
    rain_array = root["rain"]
    rain_time = _read_zarr_time(root)

    strong_rows = []

    severe_df = df[df["target_stage_response_p50"]].copy()

    for _, row in severe_df.iterrows():
        features = _event_strong_rain_features_from_zarr(
            rain_array=rain_array,
            rain_time=rain_time,
            start_time=pd.Timestamp(row["effective_start_rain"]),
            end_time=pd.Timestamp(row["date_peak"]),
        )

        strong_rows.append(
            {
                "site_id": site_id,
                "date_peak": row["date_peak"],
                "delta_stage": row["delta_stage"],
                "time_max_pixel_rain_to_stage_peak_hr": row[
                    "time_max_pixel_rain_to_stage_peak_hr"
                ],
                "time_max_pixel_acc_to_stage_peak_hr": row[
                    "time_max_pixel_acc_to_stage_peak_hr"
                ],
                "max_pixel_rain_distance_to_gauge_km": row[
                    "max_pixel_rain_distance_to_gauge_km"
                ],
                "max_pixel_acc_distance_to_gauge_km": row[
                    "max_pixel_acc_distance_to_gauge_km"
                ],
                **features,
            }
        )

    del root, rain_array, rain_time

    severe = pd.DataFrame(strong_rows)

    if severe.empty:
        return {
            "site_id": site_id,
            "alert_ready": False,
            "n_events": int(len(df)),
            "n_stage_response_events_p50": 0,
            "n_stage_response_events_with_strong_rain": 0,
            "delta_stage_target_p50": delta_stage_target_p50,
            "strong_rain_filter_mm_h": STRONG_RAIN_MM_H,
        }

    severe = severe.replace([np.inf, -np.inf], np.nan)

    severe_rain = severe[
        severe["strong_max_pixel_rain"].fillna(0.0) >= STRONG_RAIN_MM_H
    ].copy()

    alert_ready = len(severe_rain) >= MIN_EVENTS
    calib = severe_rain.copy()

    calib["velocity_from_stage1_km_h"] = _safe_ratio(
        calib["max_pixel_rain_distance_to_gauge_km"],
        calib["time_max_pixel_rain_to_stage_peak_hr"],
    )

    calib["velocity_from_stage2_km_h"] = _safe_ratio(
        calib["max_pixel_acc_distance_to_gauge_km"],
        calib["time_max_pixel_acc_to_stage_peak_hr"],
    )

    calib["stage1_vs_stage2_lead_advantage_hr"] = (
        _num(calib["time_max_pixel_rain_to_stage_peak_hr"])
        - _num(calib["time_max_pixel_acc_to_stage_peak_hr"])
    )

    out = {
        "site_id": site_id,
        "alert_ready": bool(alert_ready),
        "n_events": int(len(df)),
        "n_stage_response_events_p50": int(len(severe)),
        "n_stage_response_events_with_strong_rain": int(len(severe_rain)),
        "delta_stage_target_p50": delta_stage_target_p50,
        "strong_rain_filter_mm_h": STRONG_RAIN_MM_H,

        "stage1_signal": "current_strong_max_pixel_rain",
        "stage1_strong_max_pixel_rain_threshold": max(
            STRONG_RAIN_MM_H,
            _q(calib["strong_max_pixel_rain"], 0.50),
        ),

        "stage2_signal_primary": "current_strong_event_total_acc",
        "stage2_strong_event_total_acc_threshold": _q(
            calib["strong_event_total_acc"], 0.50
        ),

        "stage2_signal_secondary": "current_strong_max_pixel_acc",
        "stage2_strong_max_pixel_acc_threshold": _q(
            calib["strong_max_pixel_acc"], 0.50
        ),

        "stage2_signal_hourly_basin": "current_strong_event_max_hourly_basin_sum",
        "stage2_strong_event_max_hourly_basin_sum_threshold": _q(
            calib["strong_event_max_hourly_basin_sum"], 0.50
        ),

        "stage1_expected_lead_time_median_hr": _q(
            calib["time_max_pixel_rain_to_stage_peak_hr"], 0.50
        ),
        "stage1_expected_lead_time_p25_hr": _q(
            calib["time_max_pixel_rain_to_stage_peak_hr"], 0.25
        ),
        "stage1_expected_lead_time_p75_hr": _q(
            calib["time_max_pixel_rain_to_stage_peak_hr"], 0.75
        ),

        "stage2_expected_lead_time_median_hr": _q(
            calib["time_max_pixel_acc_to_stage_peak_hr"], 0.50
        ),
        "stage2_expected_lead_time_p25_hr": _q(
            calib["time_max_pixel_acc_to_stage_peak_hr"], 0.25
        ),
        "stage2_expected_lead_time_p75_hr": _q(
            calib["time_max_pixel_acc_to_stage_peak_hr"], 0.75
        ),

        "expected_velocity_from_stage1_median_km_h": _q(
            calib["velocity_from_stage1_km_h"], 0.50
        ),
        "expected_velocity_from_stage1_p25_km_h": _q(
            calib["velocity_from_stage1_km_h"], 0.25
        ),
        "expected_velocity_from_stage1_p75_km_h": _q(
            calib["velocity_from_stage1_km_h"], 0.75
        ),

        "expected_velocity_from_stage2_median_km_h": _q(
            calib["velocity_from_stage2_km_h"], 0.50
        ),

        "stage1_vs_stage2_lead_advantage_median_hr": _q(
            calib["stage1_vs_stage2_lead_advantage_hr"], 0.50
        ),
        "stage1_vs_stage2_lead_advantage_p75_hr": _q(
            calib["stage1_vs_stage2_lead_advantage_hr"], 0.75
        ),
    }

    del df, severe_df, severe, severe_rain, calib
    gc.collect()

    return out


def fit_basin_predictors(
    *,
    summary_dir: Path,
    out_dir: Path,
) -> Path:
    summary_dir = Path(summary_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = summary_dir.parent
    files = sorted(summary_dir.glob("*_historical_event_summary.parquet"))

    print("=" * 100)
    print("FITTING STRONG-RAIN OPERATIONAL PREDICTORS")
    print("=" * 100)
    print(f"summary_dir : {summary_dir}")
    print(f"base_dir    : {base_dir}")
    print(f"out_dir     : {out_dir}")
    print(f"strong rain : >= {STRONG_RAIN_MM_H} mm/h")
    print(f"files       : {len(files)}")

    zarr_index = _find_zarr_files(base_dir)
    print(f"zarr files  : {len(zarr_index)}")

    rows = []

    for i, fp in enumerate(files, start=1):
        try:
            rows.append(
                fit_one_summary(
                    fp,
                    base_dir=base_dir,
                    zarr_index=zarr_index,
                )
            )
        except Exception as e:
            print(f"[ERROR] {fp.name}: {type(e).__name__}: {e}")

        if i % 50 == 0:
            gc.collect()
            print(f"[{i}/{len(files)}] processed")

    predictors = pd.DataFrame(rows)

    preferred = [
        "site_id",
        "alert_ready",
        "n_events",
        "n_stage_response_events_p50",
        "n_stage_response_events_with_strong_rain",
        "delta_stage_target_p50",
        "strong_rain_filter_mm_h",
        "stage1_signal",
        "stage1_strong_max_pixel_rain_threshold",
        "stage1_expected_lead_time_median_hr",
        "stage1_expected_lead_time_p25_hr",
        "stage1_expected_lead_time_p75_hr",
        "expected_velocity_from_stage1_median_km_h",
        "expected_velocity_from_stage1_p25_km_h",
        "expected_velocity_from_stage1_p75_km_h",
        "stage2_signal_primary",
        "stage2_strong_event_total_acc_threshold",
        "stage2_signal_secondary",
        "stage2_strong_max_pixel_acc_threshold",
        "stage2_signal_hourly_basin",
        "stage2_strong_event_max_hourly_basin_sum_threshold",
        "stage2_expected_lead_time_median_hr",
        "stage2_expected_lead_time_p25_hr",
        "stage2_expected_lead_time_p75_hr",
        "expected_velocity_from_stage2_median_km_h",
        "stage1_vs_stage2_lead_advantage_median_hr",
        "stage1_vs_stage2_lead_advantage_p75_hr",
    ]

    cols = [c for c in preferred if c in predictors.columns]
    rest = [c for c in predictors.columns if c not in cols]
    predictors = predictors[cols + rest]

    parquet_fp = out_dir / "basin_operational_alert_predictors.parquet"
    csv_fp = out_dir / "basin_operational_alert_predictors.csv"

    predictors.to_parquet(parquet_fp, index=False)
    predictors.to_csv(csv_fp, index=False)

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"parquet: {parquet_fp}")
    print(f"csv    : {csv_fp}")
    print(f"rows   : {len(predictors)}")

    return parquet_fp