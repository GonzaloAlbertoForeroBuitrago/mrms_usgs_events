from __future__ import annotations

from pathlib import Path
import json
import math
import shutil

import numpy as np
import pandas as pd
import orjson


ALERT_COLORS = {
    "NORMAL": "#3BA55D",
    "WATCH": "#F1C40F",
    "WARNING": "#E67E22",
    "SEVERE": "#E74C3C",
}

ALERT_RANK = {
    "NORMAL": 0,
    "WATCH": 1,
    "WARNING": 2,
    "SEVERE": 3,
}


def _json_safe_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
        return None if not math.isfinite(v) else v
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def _json_safe_properties(row: pd.Series, keep_cols: list[str] | None = None) -> dict:
    if keep_cols is None:
        keep_cols = list(row.index)
    return {c: _json_safe_value(row[c]) for c in keep_cols if c in row.index}


def _load_geojson(fp: Path) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_geojson(obj: dict, fp: Path) -> Path:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    return fp


def _find_basin_geojson(base_dir: Path, state: str, site_id: str) -> Path | None:
    state = state.upper()
    site_id = str(site_id)

    candidates = [
        base_dir / "usgs_basins_json" / state / f"{site_id}.json",
        base_dir / "basins_json" / state / f"{site_id}.json",
        base_dir / "created_basins_json" / state / f"{site_id}.json",
    ]

    for fp in candidates:
        if fp.exists():
            return fp

    # Fallback recursive search. Slower, but useful while organizing folders.
    roots = [
        base_dir / "usgs_basins_json",
        base_dir / "basins_json",
        base_dir / "created_basins_json",
    ]

    for root in roots:
        if root.exists():
            matches = list(root.rglob(f"{site_id}.json"))
            if matches:
                return matches[0]

    return None


def _extract_geometry_from_basin_json(fp: Path) -> dict | None:
    obj = _load_geojson(fp)

    if obj.get("type") == "FeatureCollection":
        feats = obj.get("features", [])
        if not feats:
            return None
        return feats[0].get("geometry")

    if obj.get("type") == "Feature":
        return obj.get("geometry")

    if obj.get("type") in {"Polygon", "MultiPolygon"}:
        return obj

    return None


def _pixel_polygon_from_center(lon: float, lat: float, dx: float = 0.01, dy: float = 0.01) -> dict:
    half_dx = dx / 2.0
    half_dy = dy / 2.0

    x0 = lon - half_dx
    x1 = lon + half_dx
    y0 = lat - half_dy
    y1 = lat + half_dy

    return {
        "type": "Polygon",
        "coordinates": [[
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0],
        ]],
    }


def export_basin_alerts_geojson(
    *,
    state: str,
    base_dir: Path,
    basin_alerts_parquet: Path,
    out_geojson: Path,
    relevant_levels: list[str] | None = None,
    max_features: int = 300,
) -> Path:
    import json
    from time import perf_counter

    import pandas as pd

    t_total = perf_counter()

    state = state.upper()
    base_dir = Path(base_dir)
    basin_alerts_parquet = Path(basin_alerts_parquet)
    out_geojson = Path(out_geojson)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)

    if relevant_levels is None:
        relevant_levels = ["SEVERE", "WARNING"]

    print("=" * 100, flush=True)
    print("EXPORT BASIN ALERTS GEOJSON", flush=True)
    print("=" * 100, flush=True)
    print(f"state            : {state}", flush=True)
    print(f"base_dir         : {base_dir}", flush=True)
    print(f"alerts parquet   : {basin_alerts_parquet}", flush=True)
    print(f"out_geojson      : {out_geojson}", flush=True)
    print(f"relevant_levels  : {relevant_levels}", flush=True)
    print(f"max_features     : {max_features}", flush=True)

    t_read_alerts = perf_counter()
    basin_df = pd.read_parquet(basin_alerts_parquet)
    print(f"[GEOJSON TIMING] read basin alerts parquet: {perf_counter() - t_read_alerts:.2f} sec", flush=True)
    print(f"[GEOJSON CHECK] basin alerts rows: {len(basin_df):,}", flush=True)

    t_filter = perf_counter()
    basin_df = basin_df[basin_df["alert_level"].isin(relevant_levels)].copy()

    if "alert_rank" in basin_df.columns:
        basin_df = basin_df.sort_values(
            ["alert_rank", "efficient_weighted_percentile", "current_max_pixel_value"],
            ascending=[False, False, False],
        )
    else:
        basin_order = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "SEVERE": 3}
        basin_df["alert_rank"] = basin_df["alert_level"].map(basin_order).fillna(0).astype(int)
        basin_df = basin_df.sort_values(
            ["alert_rank", "estimated_delta_water_stage", "current_max_pixel_value"],
            ascending=[False, False, False],
        )

    basin_df = basin_df.head(max_features).reset_index(drop=True)

    print(f"[GEOJSON TIMING] filter/sort alerts: {perf_counter() - t_filter:.2f} sec", flush=True)
    print(f"[GEOJSON CHECK] filtered basins: {len(basin_df):,}", flush=True)

    t_find_files = perf_counter()

    basins_dir_candidates = [
        base_dir / "basins_json" / state,
        base_dir / "basins_json",
        base_dir / "hydro_history" / "basins_json" / state,
        base_dir / "hydro_history" / "basins_json",
    ]

    existing_dirs = [p for p in basins_dir_candidates if p.exists()]
    print(f"[GEOJSON CHECK] basin json candidate dirs: {[str(p) for p in existing_dirs]}", flush=True)

    basin_json_lookup: dict[str, Path] = {}

    for root_dir in existing_dirs:
        for fp in root_dir.rglob("*.json"):
            site_id = fp.stem
            basin_json_lookup.setdefault(site_id, fp)

    print(f"[GEOJSON TIMING] build basin json lookup: {perf_counter() - t_find_files:.2f} sec", flush=True)
    print(f"[GEOJSON CHECK] basin json files indexed: {len(basin_json_lookup):,}", flush=True)

    t_features = perf_counter()

    features = []
    missing_basins = []

    for row in basin_df.itertuples(index=False):
        site_id = str(row.site_id)
        basin_fp = basin_json_lookup.get(site_id)

        if basin_fp is None:
            missing_basins.append(site_id)
            continue

        with open(basin_fp, "r", encoding="utf-8") as f:
            basin_geojson = json.load(f)

        if basin_geojson.get("type") == "FeatureCollection":
            if not basin_geojson.get("features"):
                missing_basins.append(site_id)
                continue
            geometry = basin_geojson["features"][0].get("geometry")
        elif basin_geojson.get("type") == "Feature":
            geometry = basin_geojson.get("geometry")
        else:
            geometry = basin_geojson.get("geometry")

        if geometry is None:
            missing_basins.append(site_id)
            continue

        properties = {}

        for col in basin_df.columns:
            value = getattr(row, col)

            if pd.isna(value):
                properties[col] = None
            elif hasattr(value, "item"):
                properties[col] = value.item()
            else:
                properties[col] = value

        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        )

    print(f"[GEOJSON TIMING] build features from basin json: {perf_counter() - t_features:.2f} sec", flush=True)
    print(f"[GEOJSON CHECK] features: {len(features):,}", flush=True)
    print(f"[GEOJSON CHECK] missing basins: {len(missing_basins):,}", flush=True)

    t_write = perf_counter()

    out = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(out_geojson, "wb") as f:
        f.write(
            orjson.dumps(
                out,
                option=orjson.OPT_NON_STR_KEYS,
            )
        )

    print(f"[GEOJSON TIMING] write geojson: {perf_counter() - t_write:.2f} sec", flush=True)

    print("=" * 100, flush=True)
    print("BASIN ALERTS GEOJSON DONE", flush=True)
    print("=" * 100, flush=True)
    print(f"output          : {out_geojson}", flush=True)
    print(f"features        : {len(features)}", flush=True)
    print(f"missing basins  : {len(missing_basins)}", flush=True)
    print(f"[GEOJSON TIMING] total export: {perf_counter() - t_total:.2f} sec", flush=True)

    return out_geojson


def export_pixel_alerts_geojson(
    *,
    pixel_alerts_parquet: Path,
    out_geojson: Path,
    max_pixels: int | None = None,
    pixel_size_deg: float = 0.01,
) -> Path:
    pixel_alerts_parquet = Path(pixel_alerts_parquet)
    out_geojson = Path(out_geojson)

    df = pd.read_parquet(pixel_alerts_parquet)

    if df.empty:
        obj = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "n_features": 0,
                "note": "No pixel alerts found.",
            },
        }
        return _write_geojson(obj, out_geojson)

    if max_pixels is not None and len(df) > max_pixels:
        sort_cols = [c for c in ["estimated_delta_water_stage", "current_pixel_value"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False).head(max_pixels).copy()
        else:
            df = df.head(max_pixels).copy()

    required = {"lon", "lat"}
    if not required.issubset(df.columns):
        raise ValueError("pixel_alerts parquet must contain lon and lat columns")

    property_cols = [
        "state",
        "site_id",
        "pixel_id_state",
        "pixel_id_basin",
        "row",
        "col",
        "current_pixel_value",
        "current_pixel_accumulation",
        "current_basin_accumulation",
        "historical_basin_accumulation_threshold",
        "estimated_delta_water_stage",
        "estimated_time_to_rain_peak_accumulation_hr",
        "estimated_time_to_stage_peak_hr",
        "matched_hist_pixel_value",
        "matched_hist_basin_accumulation",
        "matched_hist_delta_water_stage",
        "match_score",
        "matched_n",
    ]

    features = []

    for _, row in df.iterrows():
        lon = float(row["lon"])
        lat = float(row["lat"])
        geom = _pixel_polygon_from_center(lon, lat, dx=pixel_size_deg, dy=pixel_size_deg)

        props = _json_safe_properties(row, property_cols)

        delta = props.get("estimated_delta_water_stage")
        if delta is None:
            alert_level = "WATCH"
        elif delta >= 10.0:
            alert_level = "SEVERE"
        elif delta >= 2.0:
            alert_level = "WARNING"
        else:
            alert_level = "WATCH"

        props["alert_level"] = alert_level
        props["fill_color"] = ALERT_COLORS.get(alert_level, "#808080")
        props["stroke_color"] = "#222222"

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        })

    obj = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "n_features": len(features),
            "pixel_size_deg": pixel_size_deg,
        },
    }

    _write_geojson(obj, out_geojson)

    print("=" * 100)
    print("PIXEL ALERTS GEOJSON")
    print("=" * 100)
    print(f"output   : {out_geojson}")
    print(f"features : {len(features):,}")

    return out_geojson


def export_state_alerts_for_tethys(
    *,
    state: str,
    base_dir: Path = Path("/data/repository_code/unified_data"),
    alerts_dir: Path | None = None,
    out_dir: Path | None = None,
    public_dir: Path | None = None,
    max_pixels: int | None = 100_000,
    pixel_size_deg: float = 0.01,
) -> dict[str, Path]:
    state = state.upper()
    base_dir = Path(base_dir)

    if alerts_dir is None:
        alerts_dir = base_dir / "ews_alerts" / state
    else:
        alerts_dir = Path(alerts_dir)

    if out_dir is None:
        out_dir = base_dir / "ews_tethys" / state
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    basin_alerts_parquet = alerts_dir / "basin_alerts.parquet"
    pixel_alerts_parquet = alerts_dir / "pixel_alerts.parquet"

    if not basin_alerts_parquet.exists():
        raise FileNotFoundError(f"Missing basin alerts parquet: {basin_alerts_parquet}")

    if not pixel_alerts_parquet.exists():
        raise FileNotFoundError(f"Missing pixel alerts parquet: {pixel_alerts_parquet}")

    basin_geojson = out_dir / "basin_alerts.geojson"
    pixel_geojson = out_dir / "pixel_alerts.geojson"

    export_basin_alerts_geojson(
        state=state,
        base_dir=base_dir,
        basin_alerts_parquet=basin_alerts_parquet,
        out_geojson=basin_geojson,
    )

    export_pixel_alerts_geojson(
        pixel_alerts_parquet=pixel_alerts_parquet,
        out_geojson=pixel_geojson,
        max_pixels=max_pixels,
        pixel_size_deg=pixel_size_deg,
    )

    basin_csv = out_dir / "basin_alerts.csv"
    pixel_csv = out_dir / "pixel_alerts.csv"

    pd.read_parquet(basin_alerts_parquet).to_csv(basin_csv, index=False)
    pd.read_parquet(pixel_alerts_parquet).to_csv(pixel_csv, index=False)

    result = {
        "basin_geojson": basin_geojson,
        "pixel_geojson": pixel_geojson,
        "basin_csv": basin_csv,
        "pixel_csv": pixel_csv,
        "out_dir": out_dir,
    }

    if public_dir is not None:
        public_dir = Path(public_dir)
        public_state_dir = public_dir / state
        public_state_dir.mkdir(parents=True, exist_ok=True)

        copied = {}
        for name, fp in result.items():
            if name == "out_dir":
                continue
            dst = public_state_dir / Path(fp).name
            shutil.copy2(fp, dst)
            copied[f"public_{name}"] = dst

        result.update(copied)

    print("=" * 100)
    print("TETHYS EXPORT DONE")
    print("=" * 100)
    for k, v in result.items():
        print(f"{k:20s}: {v}")

    return result
