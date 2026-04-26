from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_recent_rain_npz(fp: Path):
    z = np.load(fp, allow_pickle=True)
    rain = np.asarray(z["rain"], dtype=np.float32)
    time = pd.to_datetime(z["time"])
    if rain.ndim != 2:
        raise ValueError("recent rain npz must contain rain with shape time x state_pixels")
    return rain, time


def alert_level(score: float) -> str:
    if score >= 0.90:
        return "EXTREME"
    if score >= 0.75:
        return "HIGH"
    if score >= 0.50:
        return "MODERATE"
    if score >= 0.25:
        return "LOW"
    return "NORMAL"


def score_from_thresholds(value, p50, p75, p90, p95):
    if not np.isfinite(value):
        return 0.0
    if np.isfinite(p95) and value >= p95:
        return 1.0
    if np.isfinite(p90) and value >= p90:
        return 0.85
    if np.isfinite(p75) and value >= p75:
        return 0.65
    if np.isfinite(p50) and value >= p50:
        return 0.40
    return 0.10 if value > 0 else 0.0


def run_state_alert_engine(*, state: str, recent_rain_npz: Path, state_basin_index: Path, predictor_dir: Path, out_dir: Path) -> dict[str, Path]:
    state = state.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    rain, time = load_recent_rain_npz(recent_rain_npz)
    idx = np.load(state_basin_index, allow_pickle=True)

    site_ids = idx["site_ids"].astype(str)
    basin_ptr = idx["basin_ptr"]
    basin_indices = idx["basin_indices"]
    lon = idx["lon"]
    lat = idx["lat"]

    predictors = pd.read_parquet(predictor_dir / "basin_predictors.parquet")
    predictors["site_id"] = predictors["site_id"].astype(str)
    predictors = predictors.set_index("site_id")

    rows = []
    max_pixel_features = []

    for i, site_id in enumerate(site_ids):
        a = int(basin_ptr[i])
        b = int(basin_ptr[i + 1])
        pixpos = basin_indices[a:b]
        if len(pixpos) == 0:
            continue

        vals = rain[:, pixpos]
        vals = np.where(np.isfinite(vals), vals, 0.0)

        current_1h_max = float(vals[-1].max())
        current_1h_mean = float(vals[-1].mean())
        current_12h_total_basin_sum = float(vals.sum())
        current_12h_max_pixel_acc = float(vals.sum(axis=0).max())
        current_12h_mean_pixel_acc = float(vals.sum(axis=0).mean())

        last_max_local = int(np.argmax(vals[-1]))
        last_max_state_pos = int(pixpos[last_max_local])
        acc_max_local = int(np.argmax(vals.sum(axis=0)))
        acc_max_state_pos = int(pixpos[acc_max_local])

        if site_id in predictors.index:
            p = predictors.loc[site_id]
            score_event_total = score_from_thresholds(current_12h_total_basin_sum, p.get("event_total_acc_p50", np.nan), p.get("event_total_acc_p75", np.nan), p.get("event_total_acc_p90", np.nan), p.get("event_total_acc_p95", np.nan))
            score_pixel_acc = score_from_thresholds(current_12h_max_pixel_acc, p.get("max_pixel_acc_p50", np.nan), p.get("max_pixel_acc_p75", np.nan), p.get("max_pixel_acc_p90", np.nan), p.get("max_pixel_acc_p95", np.nan))
            score_pixel_rain = score_from_thresholds(current_1h_max, p.get("max_pixel_rain_p50", np.nan), p.get("max_pixel_rain_p75", np.nan), p.get("max_pixel_rain_p90", np.nan), p.get("max_pixel_rain_p95", np.nan))

            risk_score = float(np.nanmax([score_event_total, score_pixel_acc, score_pixel_rain]))
            best_pred = str(p.get("best_stage_predictor", ""))
            slope = float(p.get("best_stage_predictor_slope", np.nan))
            intercept = float(p.get("best_stage_predictor_intercept", np.nan))

            predictor_value = {
                "event_total_acc": current_12h_total_basin_sum,
                "max_pixel_acc": current_12h_max_pixel_acc,
                "max_pixel_rain": current_1h_max,
                "event_max_hourly_basin_sum": float(vals[-1].sum()),
            }.get(best_pred, np.nan)

            expected_stage_rise_ft = max(0.0, slope * predictor_value + intercept) if np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(predictor_value) else np.nan
            time_to_peak_hr = float(p.get("best_travel_time_median_hr", np.nan))
            best_time_label = str(p.get("best_travel_time_label", ""))
        else:
            risk_score = 0.0
            expected_stage_rise_ft = np.nan
            time_to_peak_hr = np.nan
            best_pred = ""
            best_time_label = ""

        level = alert_level(risk_score)

        row = {
            "site_id": site_id,
            "state": state,
            "valid_time": str(time[-1]),
            "current_1h_max_mm": current_1h_max,
            "current_1h_mean_mm": current_1h_mean,
            "current_12h_total_basin_sum_mm": current_12h_total_basin_sum,
            "current_12h_max_pixel_acc_mm": current_12h_max_pixel_acc,
            "current_12h_mean_pixel_acc_mm": current_12h_mean_pixel_acc,
            "risk_score": risk_score,
            "alert_level": level,
            "expected_stage_rise_ft": expected_stage_rise_ft,
            "estimated_time_to_peak_hr": time_to_peak_hr,
            "best_stage_predictor": best_pred,
            "best_time_predictor": best_time_label,
            "max_1h_pixel_lat": float(lat[last_max_state_pos]),
            "max_1h_pixel_lon": float(lon[last_max_state_pos]),
            "max_12h_acc_pixel_lat": float(lat[acc_max_state_pos]),
            "max_12h_acc_pixel_lon": float(lon[acc_max_state_pos]),
            "n_pixels": int(len(pixpos)),
        }
        rows.append(row)

        max_pixel_features.append({
            "type": "Feature",
            "properties": {
                "site_id": site_id,
                "state": state,
                "alert_level": level,
                "risk_score": risk_score,
                "current_1h_max_mm": current_1h_max,
                "current_12h_max_pixel_acc_mm": current_12h_max_pixel_acc,
                "expected_stage_rise_ft": expected_stage_rise_ft,
                "estimated_time_to_peak_hr": time_to_peak_hr,
            },
            "geometry": {"type": "Point", "coordinates": [float(lon[last_max_state_pos]), float(lat[last_max_state_pos])]},
        })

    out = pd.DataFrame(rows)

    parquet_fp = out_dir / "basin_alerts.parquet"
    csv_fp = out_dir / "basin_alerts.csv"
    json_fp = out_dir / "alerts.json"
    geojson_fp = out_dir / "max_pixels.geojson"
    slider_fp = out_dir / "state_rain_slider.npz"

    out.to_parquet(parquet_fp, index=False)
    out.to_csv(csv_fp, index=False)

    json_fp.write_text(json.dumps({"state": state, "valid_time": str(time[-1]), "n_basins": int(len(out)), "alerts": out.to_dict(orient="records")}, indent=2), encoding="utf-8")
    geojson_fp.write_text(json.dumps({"type": "FeatureCollection", "features": max_pixel_features}, indent=2), encoding="utf-8")

    np.savez_compressed(slider_fp, state=np.array(state), time=time.astype(str).to_numpy(), rain=rain.astype(np.float32), rows=idx["rows"], cols=idx["cols"], lon=idx["lon"], lat=idx["lat"])

    return {
        "basin_alerts_parquet": parquet_fp,
        "basin_alerts_csv": csv_fp,
        "alerts_json": json_fp,
        "max_pixels_geojson": geojson_fp,
        "state_rain_slider_npz": slider_fp,
    }
