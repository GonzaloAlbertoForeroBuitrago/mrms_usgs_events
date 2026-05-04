from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path("/data/repository_code/unified_data")


def _safe_value(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return str(v) if isinstance(v, pd.Timestamp) else v


def _points_geojson(df: pd.DataFrame, lat_col: str, lon_col: str) -> dict:
    features = []
    for _, r in df.iterrows():
        lat = r.get(lat_col)
        lon = r.get(lon_col)
        if pd.isna(lat) or pd.isna(lon):
            continue

        props = {
            k: _safe_value(v)
            for k, v in r.items()
            if k not in [lat_col, lon_col]
        }

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


def export_state_alerts_for_tethys(
    *,
    state: str,
    base_dir: Path = DEFAULT_BASE_DIR,
    alerts_parquet: Path | None = None,
    out_dir: Path | None = None,
    public_dir: Path | None = None,
) -> dict[str, str]:
    state = state.upper()
    base_dir = Path(base_dir)

    if alerts_parquet is None:
        alerts_parquet = base_dir / "ews_operational" / state / "basin_alerts.parquet"
    alerts_parquet = Path(alerts_parquet)

    out_dir = Path(out_dir) if out_dir else base_dir / "tethys_outputs" / state
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(alerts_parquet)

    lat_candidates = [
        "event_12h_max_pixel_acc_lat",
        "event_6h_max_pixel_acc_lat",
        "event_3h_max_pixel_acc_lat",
        "event_1h_max_pixel_lat",
        "lat",
        "gauge_lat",
    ]
    lon_candidates = [
        "event_12h_max_pixel_acc_lon",
        "event_6h_max_pixel_acc_lon",
        "event_3h_max_pixel_acc_lon",
        "event_1h_max_pixel_lon",
        "lon",
        "gauge_lon",
    ]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError(f"No usable lat/lon columns found in {alerts_parquet}")

    geojson_fp = out_dir / f"{state}_state_alerts.geojson"
    summary_fp = out_dir / f"{state}_state_alerts_summary.json"

    geojson_fp.write_text(
        json.dumps(_points_geojson(df, lat_col=lat_col, lon_col=lon_col), indent=2),
        encoding="utf-8",
    )

    score_col = "risk_score" if "risk_score" in df.columns else None
    level_col = "alert_level" if "alert_level" in df.columns else None

    top = df.copy()
    if score_col:
        top = top.sort_values(score_col, ascending=False)

    summary = {
        "state": state,
        "n_basins": int(len(df)),
        "alerts_parquet": str(alerts_parquet),
        "geojson": str(geojson_fp),
        "counts_by_alert_level": df[level_col].value_counts(dropna=False).to_dict() if level_col else {},
        "top_25": top.head(25).to_dict(orient="records"),
    }

    summary_fp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    copied = []
    if public_dir:
        dst_dir = Path(public_dir) / "ews" / state
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fp in [geojson_fp, summary_fp]:
            dst = dst_dir / fp.name
            shutil.copy2(fp, dst)
            copied.append(str(dst))

    return {
        "geojson": str(geojson_fp),
        "summary": str(summary_fp),
        "copied": copied,
    }


def export_pixel_signature_for_tethys(
    *,
    site_id: str,
    base_dir: Path = DEFAULT_BASE_DIR,
    signature_parquet: Path | None = None,
    out_dir: Path | None = None,
    public_dir: Path | None = None,
) -> dict[str, str]:
    site_id = str(site_id).zfill(8)
    base_dir = Path(base_dir)

    if signature_parquet is None:
        signature_parquet = base_dir / "pixel_signatures" / f"{site_id}_pixel_signature.parquet"
    signature_parquet = Path(signature_parquet)

    out_dir = Path(out_dir) if out_dir else base_dir / "tethys_outputs" / "pixel_signatures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(signature_parquet)

    geojson_fp = out_dir / f"{site_id}_pixel_signature.geojson"
    geojson_fp.write_text(
        json.dumps(_points_geojson(df, lat_col="lat", lon_col="lon"), indent=2),
        encoding="utf-8",
    )

    copied = []
    if public_dir:
        dst_dir = Path(public_dir) / "ews" / "pixel_signatures"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / geojson_fp.name
        shutil.copy2(geojson_fp, dst)
        copied.append(str(dst))

    return {"geojson": str(geojson_fp), "copied": copied}