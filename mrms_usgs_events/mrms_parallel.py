from __future__ import annotations

import gzip
import os
import time
import json
from pathlib import Path
from multiprocessing import get_context
from typing import Any

import numpy as np
import pandas as pd
import requests
import zarr

from .config import PipelineConfig
from .mrms import (
    _require_gdal,
    as_utc,
    first_available_radaronly,
    get_or_download_radaronly,
    hours_from_windows,
    init_zarr,
    ensure_pixel_arrays,
)
from .geo import build_mask_and_lonlat_from_basin
from .io import now_utc_iso


def _worker_process_hour(args: dict[str, Any]) -> dict[str, Any]:
    """
    Process one MRMS hour in a worker:
    download/cache -> gzip decompress -> GDAL read -> basin pixel extraction.

    Important:
    This worker DOES NOT write to Zarr.
    """
    cfg = PipelineConfig(
        base_dir=Path(args["base_dir"]),
        mrms_cache_dir=Path(args["mrms_cache_dir"]),
    )

    i = int(args["i"])
    ts = pd.Timestamp(args["time_utc"]).tz_localize("UTC")
    rows = args["rows"]
    cols = args["cols"]
    dtype = args["dtype"]

    sess = requests.Session()
    sess.headers.update(cfg.http_headers_mrms)

    gdal = _require_gdal()
    source_ref = ""
    src = "missing"

    try:
        data, src, source_ref = get_or_download_radaronly(
            cfg,
            sess,
            ts,
            cache_dir=Path(cfg.mrms_cache_dir),
        )

        if data is None:
            return {
                "i": i,
                "time_utc": str(ts),
                "status": "missing",
                "src": src,
                "source_ref": source_ref,
                "rain": None,
                "error": "download_missing",
            }

        raw = gzip.decompress(data)

        vs = f"/vsimem/mrms_parallel_{os.getpid()}_{i}.grib2"
        ds = None

        try:
            gdal.FileFromMemBuffer(vs, raw)
            ds = gdal.Open(vs)

            if ds is None:
                return {
                    "i": i,
                    "time_utc": str(ts),
                    "status": "failed",
                    "src": src,
                    "source_ref": source_ref,
                    "rain": None,
                    "error": "gdal_open_failed",
                }

            arr2d = ds.ReadAsArray()

            if arr2d is None:
                return {
                    "i": i,
                    "time_utc": str(ts),
                    "status": "failed",
                    "src": src,
                    "source_ref": source_ref,
                    "rain": None,
                    "error": "gdal_read_array_failed",
                }

            rain_1d = arr2d[rows, cols].astype(dtype, copy=False)

            return {
                "i": i,
                "time_utc": str(ts),
                "status": "ok",
                "src": src,
                "source_ref": source_ref,
                "rain": rain_1d,
                "error": "",
            }

        finally:
            ds = None
            try:
                gdal.Unlink(vs)
            except Exception:
                pass

    except Exception as e:
        return {
            "i": i,
            "time_utc": str(ts),
            "status": "failed",
            "src": src,
            "source_ref": source_ref,
            "rain": None,
            "error": f"{type(e).__name__}: {e}",
        }

    finally:
        sess.close()


def resume_fill_rain_parallel(
    cfg: PipelineConfig,
    out_path: Path,
    mask: dict,
    missing_csv: Path,
    *,
    workers: int = 4,
) -> tuple[int, int, int]:
    """
    Parallel MRMS fill.

    Workers process hourly files in parallel.
    Only the main process writes to Zarr.
    """
    out_path = Path(out_path)
    missing_csv = Path(missing_csv)

    root = zarr.open_group(out_path.as_posix(), mode="r+", zarr_version=2)
    ensure_pixel_arrays(cfg, root, mask)

    rain = root["rain"]
    times = pd.to_datetime(root["time"][:])

    n = len(times)
    if n == 0:
        return 0, 0, 0

    done = np.zeros(n, dtype=bool)
    block = 256

    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        chunk = rain[i0:i1, :]
        done[i0:i1] = np.isfinite(chunk).any(axis=1)

    missing_idx = np.where(~done)[0]

    if missing_idx.size == 0:
        return 0, 0, 0

    rows = np.asarray(mask["rows"], dtype=np.int64)
    cols = np.asarray(mask["cols"], dtype=np.int64)

    tasks = [
        {
            "i": int(i),
            "time_utc": str(pd.Timestamp(times[i])),
            "rows": rows,
            "cols": cols,
            "dtype": cfg.dtype,
            "base_dir": cfg.base_dir.as_posix(),
            "mrms_cache_dir": Path(cfg.mrms_cache_dir).as_posix(),
        }
        for i in missing_idx
    ]

    workers = max(1, min(int(workers), int(len(tasks))))

    aws_ok = 0
    mt_ok = 0
    cache_hits = 0
    missing_rows: list[tuple[str, str, str]] = []

    ctx = get_context("spawn")

    print(f"[rain-parallel] hours_to_process={len(tasks)} workers={workers}")

    with ctx.Pool(processes=workers) as pool:
        for k, result in enumerate(pool.imap_unordered(_worker_process_hour, tasks), start=1):
            i = int(result["i"])
            ts = result["time_utc"]
            status = result["status"]
            src = result["src"]
            source_ref = result["source_ref"]
            error = result["error"]

            if status == "ok":
                rain[i, :] = result["rain"]

                if src == "aws":
                    aws_ok += 1
                elif src == "mt":
                    mt_ok += 1
                elif src == "cache":
                    cache_hits += 1

            else:
                rain[i, :] = np.nan
                missing_rows.append((ts, source_ref, error))

            if (k % max(1, cfg.debug_every_n) == 0) or (k == len(tasks)):
                print(
                    f"[rain-parallel] {k}/{len(tasks)} "
                    f"aws_ok={aws_ok} mt_ok={mt_ok} "
                    f"cache={cache_hits} missing_new={len(missing_rows)}"
                )

    if missing_rows:
        miss_df = pd.DataFrame(
            missing_rows,
            columns=["time_utc", "url", "reason"],
        )

        if missing_csv.exists():
            try:
                old = pd.read_csv(missing_csv)
                miss_df = (
                    pd.concat([old, miss_df], ignore_index=True)
                    .drop_duplicates(subset=["time_utc"], keep="last")
                )
            except Exception:
                pass

        missing_csv.parent.mkdir(parents=True, exist_ok=True)
        miss_df.to_csv(missing_csv, index=False)

    try:
        zarr.consolidate_metadata(out_path.as_posix())
    except Exception:
        pass

    return int(aws_ok), int(mt_ok), int(cache_hits)


def build_zarr_radaronly_from_timerange_parallel(
    cfg: PipelineConfig,
    start,
    end,
    basin_json: Path,
    out_zarr: Path,
    missing_csv: Path,
    *,
    workers: int = 4,
) -> tuple[int, int, int]:
    """
    Build MRMS RadarOnly Zarr for a manual time range using parallel hourly workers.
    """
    times = hours_from_windows(
        pd.DataFrame({"start_rain": [start], "end_rain": [end]}),
        "start_rain",
        "end_rain",
    )

    if len(times) == 0:
        return 0, 0, 0

    _, gz_bytes = first_available_radaronly(
        cfg,
        times,
        max_checks=80,
        widen=48,
    )

    mask = build_mask_and_lonlat_from_basin(
        basin_json,
        gz_bytes,
        dtype=cfg.dtype,
    )

    init_zarr(times, out_zarr)

    aws_ok, mt_ok, cache_hits = resume_fill_rain_parallel(
        cfg=cfg,
        out_path=out_zarr,
        mask=mask,
        missing_csv=missing_csv,
        workers=workers,
    )

    return int(len(times)), int(mask["rows"].size), int(aws_ok + mt_ok + cache_hits)


def write_current_manifest(
    *,
    site_id: str,
    state: str,
    start: str,
    end: str,
    workers: int,
    basin_json: Path,
    site_meta_json: Path,
    stage_parquet: Path,
    stage_local_parquet: Path,
    out_zarr: Path,
    missing_csv: Path,
    manifest_json: Path,
) -> None:
    manifest = {
        "site_id": site_id,
        "state": state,
        "mode": "current_manual_parallel",
        "start": start,
        "end": end,
        "workers": workers,
        "basin_json": basin_json.as_posix(),
        "site_meta_json": site_meta_json.as_posix(),
        "stage_utc_parquet": stage_parquet.as_posix(),
        "stage_local_parquet": stage_local_parquet.as_posix(),
        "rain_zarr": out_zarr.as_posix(),
        "missing_csv": missing_csv.as_posix(),
        "created_at_utc": now_utc_iso(),
    }

    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")