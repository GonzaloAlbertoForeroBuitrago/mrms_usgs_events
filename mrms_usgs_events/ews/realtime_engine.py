# realtime_engine.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STRONG_RAIN_MM_H = 7.5
HEAVY_CONVECTIVE_MM_HR = 25.4
EXTREME_CONVECTIVE_MM_HR = 50.0


def _finite_float(x, default=np.nan) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _flag_ge(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(threshold) and value >= threshold)


def load_recent_rain_npz(fp: Path):
    z = np.load(fp, allow_pickle=True)

    rain = np.asarray(z["rain"], dtype=np.float32)
    time = pd.to_datetime(z["time"])

    rain = np.where(np.isfinite(rain), rain, 0.0)

    return rain, pd.DatetimeIndex(time)


def load_operational_predictors(predictor_dir: Path) -> pd.DataFrame:
    predictor_fp = predictor_dir / "basin_operational_alert_predictors.parquet"

    if not predictor_fp.exists():
        raise FileNotFoundError(
            f"Missing predictor file: {predictor_fp}"
        )

    df = pd.read_parquet(predictor_fp)

    if df.empty:
        raise ValueError("Predictor file is empty")

    df["site_id"] = df["site_id"].astype(str)

    return df


def build_current_event_features(
    *,
    site_id: str,
    vals: np.ndarray,
    time: pd.DatetimeIndex,
    pixpos: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict:

    vals = np.where(np.isfinite(vals), vals, 0.0).astype(
        np.float32,
        copy=False,
    )

    strong = np.where(
        vals >= STRONG_RAIN_MM_H,
        vals,
        0.0,
    ).astype(np.float32, copy=False)

    acc_by_pixel = vals.sum(axis=0, dtype=np.float64)
    strong_acc_by_pixel = strong.sum(axis=0, dtype=np.float64)

    basin_hourly_sum = vals.sum(axis=1, dtype=np.float64)
    strong_basin_hourly_sum = strong.sum(axis=1, dtype=np.float64)

    raw_flat_idx = int(np.argmax(vals))
    raw_t_idx, raw_pix_idx = np.unravel_index(
        raw_flat_idx,
        vals.shape,
    )

    raw_state_pos = int(pixpos[raw_pix_idx])

    strong_flat_idx = int(np.argmax(strong))
    strong_t_idx, strong_pix_idx = np.unravel_index(
        strong_flat_idx,
        strong.shape,
    )

    strong_state_pos = int(pixpos[strong_pix_idx])

    strong_acc_pix_idx = int(np.argmax(strong_acc_by_pixel))
    strong_acc_state_pos = int(pixpos[strong_acc_pix_idx])

    start_time = pd.Timestamp(time[0])
    end_time = pd.Timestamp(time[-1])

    strong_peak_time = pd.Timestamp(time[strong_t_idx])

    duration_hr = (
        end_time - start_time
    ).total_seconds() / 3600.0

    strong_peak_lag_hr = (
        end_time - strong_peak_time
    ).total_seconds() / 3600.0

    return {

        "site_id": str(site_id),

        "event_start_time": str(start_time),
        "event_end_time": str(end_time),
        "event_duration_hr": float(duration_hr),

        "strong_rain_filter_mm_h": STRONG_RAIN_MM_H,

        # Raw
        "raw_event_1h_max_mm": float(vals.max()),
        "raw_event_total_acc": float(vals.sum(dtype=np.float64)),
        "raw_event_max_pixel_acc": float(acc_by_pixel.max()),
        "raw_event_max_hourly_basin_sum": float(
            basin_hourly_sum.max()
        ),

        "raw_event_max_pixel_lat": float(
            lat[raw_state_pos]
        ),
        "raw_event_max_pixel_lon": float(
            lon[raw_state_pos]
        ),

        # Strong filtered
        "current_strong_max_pixel_rain": float(
            strong.max()
        ),

        "current_strong_event_total_acc": float(
            strong.sum(dtype=np.float64)
        ),

        "current_strong_max_pixel_acc": float(
            strong_acc_by_pixel.max()
        ),

        "current_strong_event_max_hourly_basin_sum": float(
            strong_basin_hourly_sum.max()
        ),

        "current_strong_max_pixel_rain_time": str(
            strong_peak_time
        ),

        "current_strong_max_pixel_rain_lag_hr": float(
            strong_peak_lag_hr
        ),

        "current_strong_max_pixel_rain_lat": float(
            lat[strong_state_pos]
        ),

        "current_strong_max_pixel_rain_lon": float(
            lon[strong_state_pos]
        ),

        "current_strong_max_pixel_acc_lat": float(
            lat[strong_acc_state_pos]
        ),

        "current_strong_max_pixel_acc_lon": float(
            lon[strong_acc_state_pos]
        ),

        "n_pixels": int(len(pixpos)),

        "n_positive_pixels": int(
            np.count_nonzero(acc_by_pixel > 0)
        ),

        "n_strong_positive_pixels": int(
            np.count_nonzero(strong_acc_by_pixel > 0)
        ),
    }


def classify_operational_alert(
    current: dict,
    predictor: pd.Series | None,
) -> dict:

    if predictor is None:
        return {
            "predictor_available": False,
            "alert_level": "NO_PREDICTOR",
            "alert_level_numeric": -1,
            "risk_score": np.nan,
        }

    stage1_thr = max(
        STRONG_RAIN_MM_H,
        _finite_float(
            predictor.get(
                "stage1_strong_max_pixel_rain_threshold"
            )
        ),
    )

    stage2_total_thr = _finite_float(
        predictor.get(
            "stage2_strong_event_total_acc_threshold"
        )
    )

    stage2_pixel_thr = _finite_float(
        predictor.get(
            "stage2_strong_max_pixel_acc_threshold"
        )
    )

    current_stage1 = _finite_float(
        current["current_strong_max_pixel_rain"]
    )

    current_stage2_total = _finite_float(
        current["current_strong_event_total_acc"]
    )

    current_stage2_pixel = _finite_float(
        current["current_strong_max_pixel_acc"]
    )

    stage1_alert = _flag_ge(
        current_stage1,
        stage1_thr,
    )

    stage2_total_alert = _flag_ge(
        current_stage2_total,
        stage2_total_thr,
    )

    stage2_pixel_alert = _flag_ge(
        current_stage2_pixel,
        stage2_pixel_thr,
    )

    if stage1_alert and stage2_total_alert:
        level = "WARNING"
        level_num = 3

    elif stage1_alert and stage2_pixel_alert:
        level = "ELEVATED"
        level_num = 2

    elif stage1_alert:
        level = "WATCH"
        level_num = 1

    else:
        level = "NORMAL"
        level_num = 0

    risk_score = float(
        np.clip(
            (
                (current_stage1 / stage1_thr)
                if stage1_thr > 0
                else 0.0
            ),
            0.0,
            2.0,
        ) / 2.0
    )

    return {

        "predictor_available": True,

        "alert_level": level,
        "alert_level_numeric": int(level_num),

        "risk_score": risk_score,

        "stage1_alert": bool(stage1_alert),
        "stage2_total_alert": bool(stage2_total_alert),
        "stage2_pixel_alert": bool(stage2_pixel_alert),

        "stage1_threshold_mm_h": stage1_thr,
        "stage2_total_threshold": stage2_total_thr,
        "stage2_pixel_threshold": stage2_pixel_thr,

        "estimated_time_to_stage_peak_hr": _finite_float(
            predictor.get(
                "stage1_expected_lead_time_median_hr"
            )
        ),

        "expected_velocity_km_h": _finite_float(
            predictor.get(
                "expected_velocity_from_stage1_median_km_h"
            )
        ),
    }


def _convective_class(v: float) -> str:

    if v >= EXTREME_CONVECTIVE_MM_HR:
        return "Extreme Convective Pixel"

    if v >= HEAVY_CONVECTIVE_MM_HR:
        return "Heavy Convective Pixel"

    if v >= STRONG_RAIN_MM_H:
        return "Strong Rain Pixel"

    return "None"


def run_state_alert_engine(
    *,
    state: str,
    recent_rain_npz: Path,
    state_basin_index: Path,
    predictor_dir: Path,
    out_dir: Path,
    historical_summary_dir: Path | None = None,
):

    state = state.upper()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rain, time = load_recent_rain_npz(
        recent_rain_npz
    )

    idx = np.load(
        state_basin_index,
        allow_pickle=True,
    )

    site_ids = idx["site_ids"].astype(str)

    basin_ptr = idx["basin_ptr"]
    basin_indices = idx["basin_indices"]

    lon = idx["lon"]
    lat = idx["lat"]

    predictors = load_operational_predictors(
        predictor_dir
    )

    predictor_by_site = predictors.set_index(
        "site_id",
        drop=False,
    )

    rows = []

    geojson_features = []

    for i, site_id in enumerate(site_ids):

        a = int(basin_ptr[i])
        b = int(basin_ptr[i + 1])

        pixpos = basin_indices[a:b]

        if len(pixpos) == 0:
            continue

        vals = rain[:, pixpos]

        current = build_current_event_features(
            site_id=str(site_id),
            vals=vals,
            time=time,
            pixpos=pixpos,
            lat=lat,
            lon=lon,
        )

        predictor = None

        if str(site_id) in predictor_by_site.index:

            predictor = predictor_by_site.loc[
                str(site_id)
            ]

            if isinstance(
                predictor,
                pd.DataFrame,
            ):
                predictor = predictor.iloc[0]

        alert = classify_operational_alert(
            current=current,
            predictor=predictor,
        )

        convective_class = _convective_class(
            current[
                "current_strong_max_pixel_rain"
            ]
        )

        row = {
            **current,
            **alert,
            "state": state,
            "valid_time": str(time[-1]),
            "convective_class": convective_class,
        }

        rows.append(row)

        geojson_features.append({

            "type": "Feature",

            "properties": {

                "site_id": str(site_id),

                "state": state,

                "alert_level": alert[
                    "alert_level"
                ],

                "alert_level_numeric": alert[
                    "alert_level_numeric"
                ],

                "risk_score": alert[
                    "risk_score"
                ],

                "convective_class": convective_class,

                "current_strong_max_pixel_rain":
                    current[
                        "current_strong_max_pixel_rain"
                    ],

                "current_strong_event_total_acc":
                    current[
                        "current_strong_event_total_acc"
                    ],

                "current_strong_max_pixel_acc":
                    current[
                        "current_strong_max_pixel_acc"
                    ],
            },

            "geometry": {

                "type": "Point",

                "coordinates": [

                    current[
                        "current_strong_max_pixel_rain_lon"
                    ],

                    current[
                        "current_strong_max_pixel_rain_lat"
                    ],
                ],
            },
        })

    out = pd.DataFrame(rows)

    parquet_fp = out_dir / "basin_alerts.parquet"
    csv_fp = out_dir / "basin_alerts.csv"
    json_fp = out_dir / "alerts.json"
    geojson_fp = out_dir / "max_pixels.geojson"

    out.to_parquet(
        parquet_fp,
        index=False,
    )

    out.to_csv(
        csv_fp,
        index=False,
    )

    json_fp.write_text(

        json.dumps({

            "state": state,

            "valid_time": str(
                time[-1]
            ),

            "n_basins": int(
                len(out)
            ),

            "alerts": out.replace({
                np.nan: None
            }).to_dict(
                orient="records"
            ),
        },

        indent=2),

        encoding="utf-8",
    )

    geojson_fp.write_text(

        json.dumps({

            "type": "FeatureCollection",

            "features": geojson_features,
        },

        indent=2),

        encoding="utf-8",
    )

    print("=" * 100)
    print("OPERATIONAL STATE ALERT ENGINE")
    print("=" * 100)

    print(f"state      : {state}")
    print(f"rain shape : {rain.shape}")
    print(f"basins     : {len(out)}")

    print()
    print(
        out[
            "alert_level"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    return {

        "basin_alerts_parquet":
            parquet_fp,

        "basin_alerts_csv":
            csv_fp,

        "alerts_json":
            json_fp,

        "max_pixels_geojson":
            geojson_fp,
    }