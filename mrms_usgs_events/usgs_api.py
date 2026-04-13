from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests
from shapely.geometry import MultiPolygon, mapping, shape

from .config import PipelineConfig
from .io import date_windows, now_utc_iso
from .paths import normalize_site_id

STATIONS_INVENTORY_PATH = "./data/stations_inventory.csv"
TIMEOUT = 60
MAX_RETRIES = 6
BACKOFF_SECONDS = 3
OGC_MONITORING_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items"
IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
HEADERS_OGC = {
    "User-Agent": "mrms-usgs-stage-backfill/1.0",
    "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.1",
    "Accept-Encoding": "gzip",
}
HEADERS_IV = {
    "User-Agent": "mrms-usgs-stage-backfill/1.0",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
}
START_DATE = "2019-04-01"
END_DATE = "2026-01-30"
PARAM_CODE = "00065"


def get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = 60,
    headers: Optional[dict] = None,
    max_retries: int = 5,
) -> dict:
    """
    Send a GET request and return JSON.
    If USGS responds with 429, wait and retry a few times.
    """
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, headers=headers, timeout=timeout)

        # Success
        if r.status_code == 200:
            return r.json()

        # Rate limit: wait and retry
        if r.status_code == 429:
            wait_s = min(60, 2 ** attempt)
            print(f"[USGS 429] attempt {attempt}/{max_retries}, waiting {wait_s}s")
            time.sleep(wait_s)
            continue

        # Any other HTTP error
        r.raise_for_status()

    raise RuntimeError(f"USGS request failed after {max_retries} retries: {url}")


def get_site_metadata(site_id: str) -> tuple[float, float, str]:
    meta_url = (
        f"https://api.waterdata.usgs.gov/ogcapi/v0/collections/"
        f"monitoring-locations/items/USGS-{site_id}?f=json"
    )

    meta = get_json(meta_url)
    geometry = meta.get("geometry")
    if not geometry or "coordinates" not in geometry:
        raise ValueError("Metadata is missing geometry.coordinates")
    
    coords = geometry["coordinates"]
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        raise ValueError("Invalid coordinates in metadata")
    
    lon = float(coords[0])
    lat = float(coords[1])

    return lon, lat

def get_hydrolocation_feature_id(lon: float, lat: float) -> str:
    lon_s = f"{lon:.6f}"
    lat_s = f"{lat:.6f}"

    hydro_url = (
        "https://api.water.usgs.gov/nldi/linked-data/"
        f"hydrolocation?coords=POINT({lon_s} {lat_s})"
    )

    hydro = get_json(hydro_url)
    features = hydro.get("features", [])

    if not features:
        raise ValueError("hydrolocation no devolvió resultados")

    props = features[0].get("properties", {})
    feature_id = (
        props.get("comid")
        or props.get("nhdplus_comid")
        or props.get("identifier")
        or props.get("id")
    )

    if feature_id is None:
        raise ValueError("No encontré un feature_id utilizable en hydrolocation")

    return str(feature_id)

def get_basin_geometry(feature_id: str) -> MultiPolygon:
    basin_url = f"https://api.water.usgs.gov/nldi/linked-data/comid/{feature_id}/basin"
    basin = get_json(basin_url)

    features = basin.get("features", [])
    if not features:
        raise ValueError("El endpoint basin no devolvió geometría")

    geometry = features[0].get("geometry")
    if not geometry:
        raise ValueError("Feature basin sin geometry")

    geom = shape(geometry)

    if geom.is_empty:
        raise ValueError("Geometría vacía")

    if geom.geom_type == "Polygon":
        geom = MultiPolygon([geom])
    elif geom.geom_type != "MultiPolygon":
        raise ValueError(f"Geometría no soportada: {geom.geom_type}")

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.is_empty:
        raise ValueError("Geometría inválida quedó vacía después de buffer(0)")

    return geom

def build_feature(site_id: str, geom: MultiPolygon) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "ogc_fid": 1,
            "area": float(geom.area),
            "perimeter": float(geom.length),
            "gage_id": site_id,
        },
        "id": 1,
        "geometry": mapping(geom),
        "prev": None,
        "next": None,
        "links": [],
    }

def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(path)

def fetch_monitoring_location(cfg: PipelineConfig, site_id: str) -> dict:
    sid = normalize_site_id(site_id)
    params = {"f": "json", "filter": f"agency_code='USGS' AND monitoring_location_number='{sid}'"}
    data = get_json(cfg.ogc_monitoring_locations, params=params, timeout=cfg.http_timeout_usgs, headers=cfg.http_headers_usgs)
    feats = data.get("features", [])
    if not feats:
        raise RuntimeError(f"Monitoring location not found for site_id={sid}")
    return feats[0]


def extract_inventory_row(feature: dict) -> dict:
    props = feature.get("properties", {}) or {}
    geom = feature.get("geometry", {}) or {}
    coords = geom.get("coordinates", [None, None])

    lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
    lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None

    return {
        "monitoring_location_number": str(props.get("monitoring_location_number") or ""),
        "monitoring_location_name": props.get("monitoring_location_name"),
        "state_name": props.get("state_name"),
        "county_name": props.get("county_name"),
        "altitude": props.get("altitude"),
        "contributing_drainage_area": props.get("contributing_drainage_area"),
        "time_zone_abbreviation": props.get("time_zone_abbreviation"),
        "id": props.get("id"),
        "hydrologic_unit_code": props.get("hydrologic_unit_code"),
        "lon": lon,
        "lat": lat,
        "uses_daylight_savings": props.get("uses_daylight_savings"),
    }


def download_basin_json(cfg: PipelineConfig, site_id: str, out_json: Path, done_marker: Path, *, overwrite: bool) -> None:
    if done_marker.exists() and out_json.exists() and not overwrite:
        return
    if out_json.exists() and not overwrite and not done_marker.exists():
        done_marker.write_text(now_utc_iso(), encoding="utf-8")
        return

    sid = normalize_site_id(site_id)
    url = f"{cfg.basin_gages_endpoint}/{sid}"
    params = {"f": "json"}

    try:
        data = get_json(url, params=params, timeout=cfg.http_timeout_usgs, headers=cfg.http_headers_usgs)
    except requests.exceptions.HTTPError as e:
        if getattr(e.response, "status_code", None) == 404:
            raise RuntimeError(f"Basin JSON not found for site_id={sid} (404)") from e
        raise

    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    done_marker.write_text(now_utc_iso(), encoding="utf-8")

def build_basin_json(cfg: PipelineConfig, site_id: str, out_json: Path, done_marker: Path, *, overwrite:bool) -> None:
    if done_marker.exists() and out_json.exists() and not overwrite:
        return
    if out_json.exists() and not overwrite and not done_marker.exists():
        done_marker.write_text(now_utc_iso(), encoding="utf-8")
        return
    try: 
        site_id = normalize_site_id(site_id)
        lon, lat = get_site_metadata(site_id)

        feature_id = get_hydrolocation_feature_id(lon, lat)
        geom = get_basin_geometry(feature_id)
        feature = build_feature(site_id, geom)

        atomic_write_json(out_json, feature)

        done_marker.write_text(now_utc_iso(), encoding="utf-8")

    except Exception as e:
        raise RuntimeError(f"Failed to build basin JSON for site_id={site_id}: {type(e).__name__}: {e}") from e
    
def discover_time_series_id(cfg: PipelineConfig, site_id: str) -> Optional[str]:
    sid = normalize_site_id(site_id)

    params = {
        "f": "json",
        "monitoring_location_id": f"USGS-{sid}",
        "parameter_code": cfg.param_stage,
        "limit": str(cfg.stage_ts_meta_limit),
    }

    data = get_json(
        cfg.ogc_ts_meta,
        params=params,
        timeout=cfg.http_timeout_usgs,
        headers=cfg.http_headers_usgs,
    )

    feats = data.get("features", [])
    if not feats:
        return None

    best_score = None
    best_id = None

    for f in feats:
        props = f.get("properties", {}) or {}

        ts_id = f.get("id") or props.get("id")
        if not ts_id:
            continue

        begin = pd.to_datetime(props.get("begin_utc"), utc=True, errors="coerce")
        end = pd.to_datetime(props.get("end_utc"), utc=True, errors="coerce")

        is_raw = (
            (props.get("computation_identifier") in (None, "", "None"))
            and (props.get("statistic_id") in (None, "", "None"))
        )

        # Score
        score = (
            1000 if is_raw else 0,  # 🔥 prioriza RAW
            end.timestamp() if pd.notna(end) else 0,
        )

        if best_score is None or score > best_score:
            best_score = score
            best_id = ts_id

    return best_id


def build_continuous_url(cfg: PipelineConfig, site_id: str, ts_id: Optional[str], start_dt: str, end_dt: str) -> str:
    sid = normalize_site_id(site_id)
    start_iso = f"{start_dt}T00:00:00Z"
    end_iso = f"{end_dt}T23:59:59Z"

    params: dict[str, str] = {
        "f": "json",
        "datetime": f"{start_iso}/{end_iso}",
        "properties": "time,value",
        "limit": "20000",
    }
    if ts_id:
        params["time_series_id"] = ts_id
    else:
        params["monitoring_location_id"] = f"USGS-{sid}"
        params["parameter_code"] = cfg.param_stage

    req = requests.Request("GET", cfg.ogc_continuous, params=params).prepare()
    assert req.url is not None
    return req.url


def paged_features(cfg: PipelineConfig, url: str) -> Iterable[dict]:
    next_url = url

    while next_url:
        payload = get_json(
            next_url,
            timeout=cfg.http_timeout_usgs,
            headers=cfg.http_headers_usgs,
        )

        for feat in payload.get("features", []):
            yield feat

        next_url = next(
            (lk.get("href") for lk in payload.get("links", []) if lk.get("rel") == "next" and lk.get("href")),
            None,
        )

        if next_url:
            time.sleep(0.2)


def fetch_stage_window(cfg: PipelineConfig, site_id: str, ts_id: Optional[str], start_dt: str, end_dt: str) -> Optional[pd.DataFrame]:
    url = build_continuous_url(cfg, site_id, ts_id, start_dt, end_dt)
    rows: list[tuple[str, Any]] = []
    for feat in paged_features(cfg, url):
        p = feat.get("properties", {}) or {}
        t = p.get("time")
        v = p.get("value")
        if t is not None and v is not None:
            rows.append((t, v))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["datetime", "Stage_ft"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
    df["Stage_ft"] = pd.to_numeric(df["Stage_ft"], errors="coerce")
    df = df.dropna(subset=["datetime", "Stage_ft"])
    if df.empty:
        return None
    return df.sort_values("datetime").drop_duplicates("datetime", keep="last")


def download_stage_parquet(
    cfg: PipelineConfig,
    site_id: str,
    out_parquet: Path,
    done_marker: Path,
    *,
    start_date: str,
    end_date: str,
    overwrite: bool,
) -> int:
    if done_marker.exists() and out_parquet.exists() and not overwrite:
        try:
            return int(len(pd.read_parquet(out_parquet, columns=["datetime"])))
        except Exception:
            return -1

    if out_parquet.exists() and not overwrite and not done_marker.exists():
        done_marker.write_text(now_utc_iso(), encoding="utf-8")
        try:
            return int(len(pd.read_parquet(out_parquet, columns=["datetime"])))
        except Exception:
            return -1

    ts_id = discover_time_series_id(cfg, site_id)

    parts: list[pd.DataFrame] = []
    for w_start, w_end in date_windows(start_date, end_date, cfg.stage_window_days):
        dfw = fetch_stage_window(cfg, site_id, ts_id, w_start, w_end)
        if dfw is not None and not dfw.empty:
            parts.append(dfw)
        time.sleep(0.35 + random.uniform(0.0, 0.2))
    if parts:
        print("Stage window data successfully retrieved")

    else:
        print("No stage window data found, falling back to IV data")
        df_all = fetch_iv(site_id)
        if df_all is None or df_all.empty:
            print("No IV data found either, returning 0")
            return 0
        print("IV data successfully retrieved")
        parts.append(df_all)

    df_all = pd.concat(parts, ignore_index=True).sort_values("datetime").drop_duplicates("datetime")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_parquet, index=False, engine="pyarrow")
    done_marker.write_text(now_utc_iso(), encoding="utf-8")
    return int(len(df_all))

def retry_get(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = TIMEOUT,
) -> requests.Response:
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} transient", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES:
                raise
            time.sleep(BACKOFF_SECONDS * attempt)

    raise last_err

def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["Stage_ft"] = pd.to_numeric(df["Stage_ft"], errors="coerce")

    df = df.dropna(subset=["datetime", "Stage_ft"])
    if df.empty:
        return df

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return df.reset_index(drop=True)

def normalize_iv_timeseries(timeseries_list: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for ts in timeseries_list:
        for block in ts.get("values", []) or []:
            for item in block.get("value", []) or []:
                qualifiers = item.get("qualifiers")
                if isinstance(qualifiers, list):
                    qualifiers = ",".join(qualifiers)

                rows.append(
                    {
                        "datetime": item.get("dateTime"),
                        "Stage_ft": item.get("value"),
                    }
                )

    return pd.DataFrame(rows)

def fetch_iv(site_id: str) -> pd.DataFrame:
    params = {
        "format": "json",
        "sites": site_id,
        "parameterCd": PARAM_CODE,
        "startDT": START_DATE,
        "endDT": END_DATE,
    }
    r = retry_get(IV_URL, params=params, headers=HEADERS_IV)
    data = r.json()
    timeseries_list = data.get("value", {}).get("timeSeries", []) or []
    return finalize_dataframe(normalize_iv_timeseries(timeseries_list))
