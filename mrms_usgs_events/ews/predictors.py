from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def safe_corr(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def slope_intercept(x, y):
    x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan
    b, a = np.polyfit(x[m], y[m], 1)
    return float(b), float(a)


def fit_one_summary(fp: Path) -> dict:
    df = pd.read_parquet(fp)
    site_id = str(df["site_id"].iloc[0])

    time_targets = {
        "time_event_acc_to_stage_peak_hr": "event_acc",
        "time_max_pixel_acc_to_stage_peak_hr": "pixel_acc",
        "time_max_pixel_rain_to_stage_peak_hr": "pixel_rain",
    }

    candidates = []
    for col, label in time_targets.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 5:
            candidates.append({
                "label": label,
                "column": col,
                "n": int(len(vals)),
                "median": float(vals.median()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
            })

    if candidates:
        candidates = sorted(candidates, key=lambda d: (d["iqr"], -d["n"]))
        best_time = candidates[0]
    else:
        best_time = {"label": "none", "column": "", "n": 0, "median": np.nan, "mean": np.nan, "std": np.nan, "iqr": np.nan}

    rain_predictors = [
        "event_total_acc",
        "event_max_hourly_basin_sum",
        "max_pixel_acc",
        "max_pixel_rain",
        "pixel_acc_contribution_pct",
        "max_pixel_acc_distance_to_gauge_km",
        "max_pixel_rain_distance_to_gauge_km",
    ]

    stage_scores = []
    for col in rain_predictors:
        corr = safe_corr(df[col], df["delta_stage"])
        slope, intercept = slope_intercept(df[col], df["delta_stage"])
        stage_scores.append({
            "column": col,
            "corr_delta_stage": corr,
            "abs_corr_delta_stage": abs(corr) if np.isfinite(corr) else np.nan,
            "slope_delta_stage": slope,
            "intercept_delta_stage": intercept,
        })

    stage_scores_df = pd.DataFrame(stage_scores).sort_values("abs_corr_delta_stage", ascending=False)
    best_stage = stage_scores_df.iloc[0].to_dict() if len(stage_scores_df) else {}

    qcols = ["event_total_acc", "event_max_hourly_basin_sum", "max_pixel_acc", "max_pixel_rain", "delta_stage"]
    q = {}
    for col in qcols:
        vals = pd.to_numeric(df[col], errors="coerce")
        for p in [0.50, 0.75, 0.90, 0.95]:
            q[f"{col}_p{int(p*100)}"] = float(vals.quantile(p)) if vals.notna().any() else np.nan

    out = {
        "site_id": site_id,
        "n_events": int(len(df)),
        "best_travel_time_label": best_time["label"],
        "best_travel_time_column": best_time["column"],
        "best_travel_time_median_hr": best_time["median"],
        "best_travel_time_mean_hr": best_time["mean"],
        "best_travel_time_std_hr": best_time["std"],
        "best_travel_time_iqr_hr": best_time["iqr"],
        "best_stage_predictor": best_stage.get("column", ""),
        "best_stage_predictor_corr": best_stage.get("corr_delta_stage", np.nan),
        "best_stage_predictor_slope": best_stage.get("slope_delta_stage", np.nan),
        "best_stage_predictor_intercept": best_stage.get("intercept_delta_stage", np.nan),
        "corr_event_total_acc_delta_stage": safe_corr(df["event_total_acc"], df["delta_stage"]),
        "corr_max_pixel_acc_delta_stage": safe_corr(df["max_pixel_acc"], df["delta_stage"]),
        "corr_max_pixel_rain_delta_stage": safe_corr(df["max_pixel_rain"], df["delta_stage"]),
        "corr_distance_time_pixel_acc": safe_corr(df["max_pixel_acc_distance_to_gauge_km"], df["time_max_pixel_acc_to_stage_peak_hr"]),
        "corr_intensity_time_pixel_rain": safe_corr(df["max_pixel_rain"], df["time_max_pixel_rain_to_stage_peak_hr"]),
    }
    out.update(q)
    return out


def fit_basin_predictors(*, summary_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(summary_dir.glob("*_historical_event_summary.parquet"))
    rows = []

    for i, fp in enumerate(files, start=1):
        try:
            rows.append(fit_one_summary(fp))
            if i % 100 == 0:
                print(f"[{i}/{len(files)}] fitted")
        except Exception as e:
            print(f"[ERROR] {fp.name}: {type(e).__name__}: {e}")

    models = pd.DataFrame(rows)
    out_fp = out_dir / "basin_predictors.parquet"
    models.to_parquet(out_fp, index=False)
    models.to_csv(out_dir / "basin_predictors.csv", index=False)
    return out_fp
