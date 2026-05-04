from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path("/data/repository_code/unified_data")


def _norm01(x):
    x = np.asarray(x, dtype="float64")
    if x.size == 0 or np.all(~np.isfinite(x)):
        return np.zeros_like(x)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _points_geojson(df: pd.DataFrame) -> dict:
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


def build_pixel_signature_one(
    *,
    site_id: str,
    state: str | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
    pixel_response_fp: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:

    site_id = str(site_id).zfill(8)
    base_dir = Path(base_dir)

    if pixel_response_fp is None:
        pixel_response_fp = (
            base_dir / "research" / "hydrologic_drivers" / "pixel_response"
            / "outputs" / f"{site_id}_pixel_response_summary.parquet"
        )

    pixel_response_fp = Path(pixel_response_fp)
    out_dir = Path(out_dir) if out_dir else base_dir / "pixel_signatures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / f"{site_id}_pixel_signature.parquet"
    out_geojson = out_dir / f"{site_id}_pixel_signature.geojson"

    if out_parquet.exists() and out_geojson.exists() and not overwrite:
        return {"parquet": out_parquet, "geojson": out_geojson}

    df = pd.read_parquet(pixel_response_fp)

    rename = {
        "hydrologic_influence_score": "response_score",
        "median_best_lag_hr": "best_lag_hr",
        "median_best_corr": "best_corr",
        "median_contribution_pct": "contribution_pct",
        "p75_contribution_pct": "contribution_pct_p75",
        "median_pixel_acc_mm": "pixel_acc_mm",
        "median_pixel_peak_mm_h": "pixel_peak_mm_h",
        "median_pixel_to_basin_lag_hr": "pixel_to_basin_lag_hr",
        "median_basin_to_stage_lag_hr": "basin_to_stage_lag_hr",
        "median_pixel_to_stage_lag_hr": "pixel_to_stage_lag_hr",
    }

    out = df.rename(columns=rename).copy()
    out["site_id"] = out["site_id"].astype(str).str.zfill(8)
    out["state"] = state.upper() if state else None
    out["response_score_norm"] = _norm01(out["response_score"].to_numpy())
    out["signature_version"] = "v2_dynamic_timing"

    keep = [
        "site_id", "state", "pixel_id", "lat", "lon",
        "response_score", "response_score_norm", "response_class",
        "best_lag_hr", "best_corr",
        "pixel_to_basin_lag_hr", "basin_to_stage_lag_hr", "pixel_to_stage_lag_hr",
        "event_frequency", "n_events_contributing",
        "contribution_pct", "contribution_pct_p75",
        "sum_attributed_stage_ft", "median_attributed_stage_ft",
        "pixel_acc_mm", "pixel_peak_mm_h",
        "signature_version",
    ]

    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    out.to_parquet(out_parquet, index=False)
    out_geojson.write_text(json.dumps(_points_geojson(out), indent=2), encoding="utf-8")

    # Optional timing signature from event-level rows
    timing_fp = (
        base_dir / "research" / "hydrologic_drivers" / "pixel_response"
        / "event_timing" / f"{site_id}_pixel_event_timing.parquet"
    )

    timing_out = out_dir / f"{site_id}_pixel_timing_signature.parquet"

    if timing_fp.exists():
        ev = pd.read_parquet(timing_fp)

        if not ev.empty:
            timing = (
                ev.groupby(["site_id", "pixel_id", "intensity_class"], as_index=False)
                .agg(
                    lat=("lat", "mean"),
                    lon=("lon", "mean"),
                    n_events=("event_id", "nunique"),
                    pixel_peak_mm_h_p50=("pixel_peak_mm_h", "median"),
                    pixel_acc_mm_p50=("pixel_acc_mm", "median"),
                    stage_rise_ft_p50=("delta_stage_ft", "median"),
                    pixel_to_basin_lag_p25_hr=("pixel_to_basin_lag_hr", lambda s: np.nanpercentile(s, 25)),
                    pixel_to_basin_lag_p50_hr=("pixel_to_basin_lag_hr", "median"),
                    pixel_to_basin_lag_p75_hr=("pixel_to_basin_lag_hr", lambda s: np.nanpercentile(s, 75)),
                    basin_to_stage_lag_p25_hr=("basin_to_stage_lag_hr", lambda s: np.nanpercentile(s, 25)),
                    basin_to_stage_lag_p50_hr=("basin_to_stage_lag_hr", "median"),
                    basin_to_stage_lag_p75_hr=("basin_to_stage_lag_hr", lambda s: np.nanpercentile(s, 75)),
                    pixel_to_stage_lag_p25_hr=("pixel_to_stage_lag_hr", lambda s: np.nanpercentile(s, 25)),
                    pixel_to_stage_lag_p50_hr=("pixel_to_stage_lag_hr", "median"),
                    pixel_to_stage_lag_p75_hr=("pixel_to_stage_lag_hr", lambda s: np.nanpercentile(s, 75)),
                    contribution_pct_p50=("contribution_pct", "median"),
                    attributed_stage_ft_p50=("attributed_stage_ft", "median"),
                )
            )
            timing["state"] = state.upper() if state else None
            timing["timing_signature_version"] = "v2_by_intensity"
            timing.to_parquet(timing_out, index=False)

    return {"parquet": out_parquet, "geojson": out_geojson, "timing": timing_out}


def build_pixel_signatures_many(
    *,
    base_dir: Path = DEFAULT_BASE_DIR,
    state: str | None = None,
    pixel_response_dir: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:

    base_dir = Path(base_dir)

    if pixel_response_dir is None:
        pixel_response_dir = (
            base_dir / "research" / "hydrologic_drivers"
            / "pixel_response" / "outputs"
        )

    pixel_response_dir = Path(pixel_response_dir)
    out_dir = Path(out_dir) if out_dir else base_dir / "pixel_signatures"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pixel_response_dir.glob("*_pixel_response_summary.parquet"))
    if limit:
        files = files[:limit]

    ok = []
    failed = []

    for i, fp in enumerate(files, 1):
        site_id = fp.name.replace("_pixel_response_summary.parquet", "").zfill(8)

        try:
            paths = build_pixel_signature_one(
                site_id=site_id,
                state=state,
                base_dir=base_dir,
                pixel_response_fp=fp,
                out_dir=out_dir,
                overwrite=overwrite,
            )
            ok.append({"site_id": site_id, **{k: str(v) for k, v in paths.items()}})
            print(f"[OK {i}/{len(files)}] {site_id}")
        except Exception as e:
            failed.append({"site_id": site_id, "file": str(fp), "error": f"{type(e).__name__}: {e}"})
            print(f"[FAIL {i}/{len(files)}] {site_id}: {e}")

    summary = {
        "base_dir": str(base_dir),
        "state": state,
        "pixel_response_dir": str(pixel_response_dir),
        "n_files": len(files),
        "ok": len(ok),
        "failed": failed,
        "out_dir": str(out_dir),
    }

    summary_fp = out_dir / "pixel_signatures_build_summary.json"
    summary_fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_file"] = str(summary_fp)

    return summary
