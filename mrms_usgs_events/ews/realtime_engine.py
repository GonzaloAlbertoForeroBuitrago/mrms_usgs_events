from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HEAVY_CONVECTIVE_MM_HR = 25.4
EXTREME_CONVECTIVE_MM_HR = 50.0


def load_recent_rain_npz(fp: Path):
    z = np.load(fp, allow_pickle=True)
    rain = np.asarray(z["rain"], dtype=np.float32)
    time = pd.to_datetime(z["time"])

    if rain.ndim != 2:
        raise ValueError("recent rain npz must contain rain with shape time x state_pixels")

    return rain, time


def finite_float(x, default=np.nan) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def flag_ge(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(threshold) and value >= threshold)


def percentile_rank(value: float, sample: pd.Series) -> float:
    vals = pd.to_numeric(sample, errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0 or not np.isfinite(value):
        return np.nan

    return float(100.0 * np.mean(vals <= value))


def load_history_for_site(summary_dir: Path, site_id: str) -> pd.DataFrame | None:
    fp = summary_dir / f"{site_id}_historical_event_summary.parquet"
    if not fp.exists():
        return None

    df = pd.read_parquet(fp)
    if df.empty:
        return None

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
    """
    Treat the recent 12h window as one operational rainfall event.

    vals shape:
        time x basin_pixels
    """
    vals = np.where(np.isfinite(vals), vals, 0.0).astype(np.float32, copy=False)

    last_hour = vals[-1]
    acc_by_pixel = vals.sum(axis=0)

    event_flat_idx = int(np.argmax(vals))
    event_max_time_idx, event_max_local_pixel = np.unravel_index(event_flat_idx, vals.shape)
    event_max_state_pos = int(pixpos[event_max_local_pixel])

    acc_max_local_pixel = int(np.argmax(acc_by_pixel))
    acc_max_state_pos = int(pixpos[acc_max_local_pixel])

    basin_hourly_sum = vals.sum(axis=1, dtype=np.float64)
    event_cum = np.cumsum(basin_hourly_sum)
    event_acc_peak_time_idx = int(np.argmax(event_cum))

    event_start_time = pd.Timestamp(time[0])
    event_end_time = pd.Timestamp(time[-1])
    event_1h_max_time = pd.Timestamp(time[event_max_time_idx])
    event_acc_peak_time = pd.Timestamp(time[event_acc_peak_time_idx])

    duration_hr = (event_end_time - event_start_time).total_seconds() / 3600.0
    time_event_1h_max_to_now_hr = (event_end_time - event_1h_max_time).total_seconds() / 3600.0
    time_event_acc_peak_to_now_hr = (event_end_time - event_acc_peak_time).total_seconds() / 3600.0

    return {
        "site_id": site_id,
        "event_start_time": str(event_start_time),
        "event_end_time": str(event_end_time),
        "event_duration_hr": float(duration_hr),

        # Last/current hour.
        "last_1h_max_mm": float(last_hour.max()),
        "last_1h_mean_mm": float(last_hour.mean()),
        "last_1h_basin_sum_mm": float(last_hour.sum()),

        # Operational event intensity.
        "event_1h_max_mm": float(vals.max()),
        "event_1h_mean_mm": float(vals.mean()),
        "event_1h_max_time": str(event_1h_max_time),
        "event_1h_max_pixel_lat": float(lat[event_max_state_pos]),
        "event_1h_max_pixel_lon": float(lon[event_max_state_pos]),

        # Operational event accumulation.
        "event_12h_total_basin_sum_mm": float(vals.sum()),
        "event_12h_max_pixel_acc_mm": float(acc_by_pixel.max()),
        "event_12h_mean_pixel_acc_mm": float(acc_by_pixel.mean()),
        "event_acc_peak_time": str(event_acc_peak_time),
        "event_12h_max_pixel_acc_lat": float(lat[acc_max_state_pos]),
        "event_12h_max_pixel_acc_lon": float(lon[acc_max_state_pos]),

        # Timing inside current event.
        "time_event_1h_max_to_now_hr": float(time_event_1h_max_to_now_hr),
        "time_event_acc_peak_to_now_hr": float(time_event_acc_peak_to_now_hr),

        "n_pixels": int(len(pixpos)),
    }


def match_current_event_to_history(
    *,
    current: dict,
    hist: pd.DataFrame | None,
) -> dict:
    """
    Compare current operational event against historical events for the same basin.

    This estimates:
    - expected stage rise from similar historical events
    - estimated time to peak from similar historical events
    - percentile ranks for current rainfall metrics
    """
    if hist is None or hist.empty:
        return {
            "history_available": False,
            "similar_event_count": 0,
            "expected_stage_rise_ft": np.nan,
            "expected_stage_rise_p75_ft": np.nan,
            "expected_stage_rise_p90_ft": np.nan,
            "estimated_time_to_peak_hr": np.nan,
            "estimated_time_to_peak_p75_hr": np.nan,
            "event_total_acc_percentile": np.nan,
            "max_pixel_acc_percentile": np.nan,
            "max_pixel_rain_percentile": np.nan,
            "delta_stage_p50": np.nan,
            "delta_stage_p75": np.nan,
            "delta_stage_p90": np.nan,
            "match_method": "no_history",
        }

    h = hist.copy()

    for col in [
        "event_total_acc",
        "max_pixel_acc",
        "max_pixel_rain",
        "delta_stage",
        "time_event_acc_to_stage_peak_hr",
        "time_max_pixel_acc_to_stage_peak_hr",
        "time_max_pixel_rain_to_stage_peak_hr",
    ]:
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")

    event_total_acc = current["event_12h_total_basin_sum_mm"]
    max_pixel_acc = current["event_12h_max_pixel_acc_mm"]
    max_pixel_rain = current["event_1h_max_mm"]

    event_total_pct = percentile_rank(event_total_acc, h["event_total_acc"])
    max_pixel_acc_pct = percentile_rank(max_pixel_acc, h["max_pixel_acc"])
    max_pixel_rain_pct = percentile_rank(max_pixel_rain, h["max_pixel_rain"])

    delta_stage_p50 = finite_float(h["delta_stage"].quantile(0.50))
    delta_stage_p75 = finite_float(h["delta_stage"].quantile(0.75))
    delta_stage_p90 = finite_float(h["delta_stage"].quantile(0.90))

    # Similarity matching:
    # First try events that are at least reasonably close to current rainfall severity.
    similar = h[
        (h["max_pixel_acc"] <= max_pixel_acc * 1.25)
        & (h["max_pixel_acc"] >= max_pixel_acc * 0.50)
    ].copy()

    # If too few, use top historical analogs by normalized distance.
    match_method = "range_match_max_pixel_acc"

    if len(similar) < 5:
        features = ["event_total_acc", "max_pixel_acc", "max_pixel_rain"]
        hh = h[features + ["delta_stage", "time_event_acc_to_stage_peak_hr"]].dropna().copy()

        if not hh.empty:
            med = hh[features].median()
            iqr = hh[features].quantile(0.75) - hh[features].quantile(0.25)
            iqr = iqr.replace(0, np.nan).fillna(hh[features].std()).replace(0, 1.0)

            cur = pd.Series(
                {
                    "event_total_acc": event_total_acc,
                    "max_pixel_acc": max_pixel_acc,
                    "max_pixel_rain": max_pixel_rain,
                }
            )

            dist = (((hh[features] - cur) / iqr) ** 2).sum(axis=1) ** 0.5
            similar = h.loc[dist.sort_values().head(min(10, len(dist))).index].copy()
            match_method = "nearest_historical_analogs"

    if similar.empty:
        similar = h.copy()
        match_method = "fallback_all_history"

    expected_stage_rise = finite_float(similar["delta_stage"].median())
    expected_stage_rise_p75 = finite_float(similar["delta_stage"].quantile(0.75))
    expected_stage_rise_p90 = finite_float(similar["delta_stage"].quantile(0.90))

    # Use event accumulation-to-stage timing as primary travel-time estimate.
    time_col = "time_event_acc_to_stage_peak_hr"
    if time_col not in similar.columns or similar[time_col].dropna().empty:
        time_col = "time_max_pixel_acc_to_stage_peak_hr"

    estimated_time = finite_float(similar[time_col].median())
    estimated_time_p75 = finite_float(similar[time_col].quantile(0.75))

    return {
        "history_available": True,
        "similar_event_count": int(len(similar)),
        "expected_stage_rise_ft": expected_stage_rise,
        "expected_stage_rise_p75_ft": expected_stage_rise_p75,
        "expected_stage_rise_p90_ft": expected_stage_rise_p90,
        "estimated_time_to_peak_hr": estimated_time,
        "estimated_time_to_peak_p75_hr": estimated_time_p75,
        "event_total_acc_percentile": event_total_pct,
        "max_pixel_acc_percentile": max_pixel_acc_pct,
        "max_pixel_rain_percentile": max_pixel_rain_pct,
        "delta_stage_p50": delta_stage_p50,
        "delta_stage_p75": delta_stage_p75,
        "delta_stage_p90": delta_stage_p90,
        "match_method": match_method,
    }


def classify_alert(
    *,
    event_1h_max_mm: float,
    max_pixel_acc_percentile: float,
    max_pixel_rain_percentile: float,
    expected_stage_rise_ft: float,
    delta_stage_p50: float,
    delta_stage_p75: float,
    delta_stage_p90: float,
) -> tuple[str, float, str]:
    """
    Percentile-based hydrologic alert score.

    Logic:
      - Rain intensity identifies convective forcing.
      - Rainfall percentiles compare current event against historical rainfall events.
      - Stage thresholds compare expected stage rise against basin-specific historical response.

    Final score:
      combines rainfall intensity, accumulation percentile, rain intensity percentile,
      and expected stage response.
    """

    convective_heavy = event_1h_max_mm >= HEAVY_CONVECTIVE_MM_HR
    convective_extreme = event_1h_max_mm >= EXTREME_CONVECTIVE_MM_HR

    # ------------------------------------------------------------------
    # Rainfall score from current event against historical event percentiles
    # ------------------------------------------------------------------
    acc_pct = max_pixel_acc_percentile if np.isfinite(max_pixel_acc_percentile) else 0.0
    rain_pct = max_pixel_rain_percentile if np.isfinite(max_pixel_rain_percentile) else 0.0

    acc_score = np.clip(acc_pct / 100.0, 0.0, 1.0)
    rain_score = np.clip(rain_pct / 100.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Convective score from absolute rain-rate intensity
    # ------------------------------------------------------------------
    if convective_extreme:
        convective_score = 1.0
    elif convective_heavy:
        convective_score = 0.7
    elif event_1h_max_mm >= 10.0:
        convective_score = 0.4
    elif event_1h_max_mm >= 2.5:
        convective_score = 0.2
    else:
        convective_score = 0.0

    # ------------------------------------------------------------------
    # Stage score using basin-specific historical delta_stage percentiles
    # ------------------------------------------------------------------
    stage_p50_threshold = max(2.0, delta_stage_p50) if np.isfinite(delta_stage_p50) else 2.0
    stage_p75_threshold = max(4.0, delta_stage_p75) if np.isfinite(delta_stage_p75) else 4.0
    stage_p90_threshold = max(6.0, delta_stage_p90) if np.isfinite(delta_stage_p90) else 6.0

    if not np.isfinite(expected_stage_rise_ft):
        stage_score = 0.0
        stage_signal = "no_expected_stage"
    elif expected_stage_rise_ft >= stage_p90_threshold:
        stage_score = 1.0
        stage_signal = "stage_ge_p90"
    elif expected_stage_rise_ft >= stage_p75_threshold:
        stage_score = 0.75
        stage_signal = "stage_ge_p75"
    elif expected_stage_rise_ft >= stage_p50_threshold:
        stage_score = 0.50
        stage_signal = "stage_ge_p50"
    elif expected_stage_rise_ft > 0:
        stage_score = 0.25
        stage_signal = "stage_positive"
    else:
        stage_score = 0.0
        stage_signal = "stage_none"

    # ------------------------------------------------------------------
    # Final hydrologic score
    # Stage has the largest weight because alert severity should represent
    # probable water-level response, not rainfall alone.
    # ------------------------------------------------------------------
    risk_score = (
        0.40 * stage_score
        + 0.25 * acc_score
        + 0.20 * rain_score
        + 0.15 * convective_score
    )

    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Avoid false alerts when there is no meaningful rainfall forcing.
    # ------------------------------------------------------------------
    if event_1h_max_mm < 2.5 and acc_pct < 20 and stage_score == 0.0:
        return (
            "NORMAL",
            0.0,
            "No meaningful rainfall forcing or expected stage response",
        )

    # ------------------------------------------------------------------
    # Alert levels requested by user
    # ------------------------------------------------------------------
    if stage_score < 0.5:
        # no hay respuesta hidrológica fuerte → limitar nivel
        if risk_score > 0.60:
            level = "HIGH"
        elif risk_score > 0.50:
            level = "MODERATE"
        elif risk_score > 0.40:
            level = "WATCH"
        elif risk_score > 0.30:
            level = "LOW"
        else:
            level = "NORMAL"
    else:
        # comportamiento normal
        if risk_score > 0.60:
            level = "EXTREME"
        elif risk_score > 0.50:
            level = "HIGH"
        elif risk_score > 0.40:
            level = "MODERATE"
        elif risk_score > 0.30:
            level = "WATCH"
        elif risk_score > 0.20:
            level = "LOW"
        else:
            level = "NORMAL"

    reason = (
        f"score={risk_score:.2f}; "
        f"convective_score={convective_score:.2f}; "
        f"acc_pct={acc_pct:.1f}; "
        f"rain_pct={rain_pct:.1f}; "
        f"stage_score={stage_score:.2f}; "
        f"{stage_signal}"
    )

    return level, risk_score, reason


def run_state_alert_engine(
    *,
    state: str,
    recent_rain_npz: Path,
    state_basin_index: Path,
    predictor_dir: Path,
    out_dir: Path,
    historical_summary_dir: Path | None = None,
) -> dict[str, Path]:
    state = state.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    rain, time = load_recent_rain_npz(recent_rain_npz)
    idx = np.load(state_basin_index, allow_pickle=True)

    site_ids = idx["site_ids"].astype(str)
    basin_ptr = idx["basin_ptr"]
    basin_indices = idx["basin_indices"]
    lon = idx["lon"]
    lat = idx["lat"]

    rows = []
    max_pixel_features = []
    convective_pixel_features = []

    for i, site_id in enumerate(site_ids):
        a = int(basin_ptr[i])
        b = int(basin_ptr[i + 1])
        pixpos = basin_indices[a:b]

        if len(pixpos) == 0:
            continue

        vals = rain[:, pixpos]

        current = build_current_event_features(
            site_id=site_id,
            vals=vals,
            time=time,
            pixpos=pixpos,
            lat=lat,
            lon=lon,
        )

        hist = None
        if historical_summary_dir is not None:
            hist = load_history_for_site(Path(historical_summary_dir), site_id)

        analog = match_current_event_to_history(current=current, hist=hist)

        alert, risk_score, alert_reason = classify_alert(
            event_1h_max_mm=current["event_1h_max_mm"],
            max_pixel_acc_percentile=analog["max_pixel_acc_percentile"],
            max_pixel_rain_percentile=analog["max_pixel_rain_percentile"],
            expected_stage_rise_ft=analog["expected_stage_rise_ft"],
            delta_stage_p50=analog["delta_stage_p50"],
            delta_stage_p75=analog["delta_stage_p75"],
            delta_stage_p90=analog["delta_stage_p90"],
        )

        convective_class = "None"
        if current["event_1h_max_mm"] >= EXTREME_CONVECTIVE_MM_HR:
            convective_class = "Extreme Convective Pixel"
        elif current["event_1h_max_mm"] >= HEAVY_CONVECTIVE_MM_HR:
            convective_class = "Heavy Convective Pixel"

        row = {
            **current,
            **analog,
            "state": state,
            "valid_time": str(time[-1]),
            "convective_class": convective_class,
            "risk_score": float(risk_score),
            "alert_level": alert,
            "alert_reason": alert_reason,
        }

        rows.append(row)

        max_pixel_features.append(
            {
                "type": "Feature",
                "properties": {
                    "site_id": site_id,
                    "state": state,
                    "alert_level": alert,
                    "risk_score": float(risk_score),
                    "alert_reason": alert_reason,
                    "convective_class": convective_class,
                    "event_1h_max_mm": current["event_1h_max_mm"],
                    "event_12h_max_pixel_acc_mm": current["event_12h_max_pixel_acc_mm"],
                    "expected_stage_rise_ft": analog["expected_stage_rise_ft"],
                    "estimated_time_to_peak_hr": analog["estimated_time_to_peak_hr"],
                    "similar_event_count": analog["similar_event_count"],
                    "match_method": analog["match_method"],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        current["event_1h_max_pixel_lon"],
                        current["event_1h_max_pixel_lat"],
                    ],
                },
            }
        )

        if current["event_1h_max_mm"] >= HEAVY_CONVECTIVE_MM_HR:
            convective_pixel_features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "site_id": site_id,
                        "state": state,
                        "convective_class": convective_class,
                        "event_1h_max_mm": current["event_1h_max_mm"],
                        "event_1h_max_time": current["event_1h_max_time"],
                        "alert_level": alert,
                        "risk_score": float(risk_score),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            current["event_1h_max_pixel_lon"],
                            current["event_1h_max_pixel_lat"],
                        ],
                    },
                }
            )

    out = pd.DataFrame(rows)

    parquet_fp = out_dir / "basin_alerts.parquet"
    csv_fp = out_dir / "basin_alerts.csv"
    json_fp = out_dir / "alerts.json"
    geojson_fp = out_dir / "max_pixels.geojson"
    convective_geojson_fp = out_dir / "convective_pixels.geojson"
    slider_fp = out_dir / "state_rain_slider.npz"

    out.to_parquet(parquet_fp, index=False)
    out.to_csv(csv_fp, index=False)

    json_fp.write_text(
        json.dumps(
            {
                "state": state,
                "valid_time": str(time[-1]),
                "n_basins": int(len(out)),
                "alert_counts": out["alert_level"].value_counts(dropna=False).to_dict()
                if len(out)
                else {},
                "alerts": out.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    geojson_fp.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": max_pixel_features},
            indent=2,
        ),
        encoding="utf-8",
    )

    convective_geojson_fp.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": convective_pixel_features},
            indent=2,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        slider_fp,
        state=np.array(state),
        time=time.astype(str).to_numpy(),
        rain=rain.astype(np.float32),
        rows=idx["rows"],
        cols=idx["cols"],
        lon=idx["lon"],
        lat=idx["lat"],
    )

    print("=" * 100)
    print("OPERATIONAL STATE ALERT ENGINE")
    print("=" * 100)
    print(f"state: {state}")
    print(f"rain shape: {rain.shape}")
    print(f"basins: {len(out)}")
    print(f"saved: {parquet_fp}")
    print(f"saved: {json_fp}")
    print(f"saved: {geojson_fp}")
    print(f"saved: {convective_geojson_fp}")
    print(f"saved: {slider_fp}")

    if len(out):
        print()
        print("Alert counts:")
        print(out["alert_level"].value_counts(dropna=False).to_string())

        print()
        cols = [
            "site_id",
            "alert_level",
            "risk_score",
            "event_1h_max_mm",
            "convective_class",
            "event_12h_max_pixel_acc_mm",
            "max_pixel_acc_percentile",
            "max_pixel_rain_percentile",
            "expected_stage_rise_ft",
            "estimated_time_to_peak_hr",
            "similar_event_count",
            "match_method",
            "alert_reason",
        ]
        print(
            out.sort_values("risk_score", ascending=False)[cols]
            .head(30)
            .to_string(index=False)
        )

    return {
        "basin_alerts_parquet": parquet_fp,
        "basin_alerts_csv": csv_fp,
        "alerts_json": json_fp,
        "max_pixels_geojson": geojson_fp,
        "convective_pixels_geojson": convective_geojson_fp,
        "state_rain_slider_npz": slider_fp,
    }