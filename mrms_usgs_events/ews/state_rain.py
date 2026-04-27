from __future__ import annotations

from pathlib import Path
from multiprocessing import get_context

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..mrms_parallel import _worker_process_hour


def build_current_state_rain_npz(
    *,
    state: str,
    state_mask_fp: Path,
    out_npz: Path,
    base_dir: Path,
    hours_back: int = 12,
    workers: int = 4,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """
    Build recent state rainfall array using a precomputed state MRMS mask.

    Output NPZ contains:
      rain: time x state_pixels
      time: datetime strings
      rows, cols, lon, lat
    """
    state = state.upper()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(base_dir=base_dir.resolve())

    z = np.load(state_mask_fp, allow_pickle=True)
    rows = z["rows"].astype(np.int32)
    cols = z["cols"].astype(np.int32)
    lon = z["lon"].astype(np.float32)
    lat = z["lat"].astype(np.float32)

    if start is not None and end is not None:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    else:
        end_ts = pd.Timestamp.utcnow().floor("h")
        start_ts = end_ts - pd.Timedelta(hours=hours_back - 1)

    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert(None)

    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert(None)

    times = pd.date_range(start=start_ts, end=end_ts, freq="h")     



    print("=" * 90)
    print("BUILD CURRENT STATE RAIN")
    print("=" * 90)
    print(f"state       : {state}")
    print(f"state mask  : {state_mask_fp}")
    print(f"out npz     : {out_npz}")
    print(f"base_dir    : {cfg.base_dir}")
    print(f"hours_back  : {hours_back}")
    print(f"time start  : {times[0]}")
    print(f"time end    : {times[-1]}")
    print(f"state pixels: {len(rows)}")
    print(f"workers     : {workers}")

    tasks = []

    for i, ts in enumerate(times):
        tasks.append(
            {
                "i": int(i),
                "time_utc": str(pd.Timestamp(ts).tz_localize(None)),
                "rows": rows,
                "cols": cols,
                "dtype": cfg.dtype,
                "base_dir": cfg.base_dir.as_posix(),
                "mrms_cache_dir": Path(cfg.mrms_cache_dir).as_posix(),
            }
        )

    workers = max(1, min(int(workers), len(tasks)))

    rain = np.full((len(times), len(rows)), np.nan, dtype=np.float32)
    ok_count = 0
    missing = []

    ctx = get_context("fork")

    with ctx.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_worker_process_hour, tasks):
            i = int(result["i"])
            status = result["status"]
            arr = result["rain"]

            if status == "ok" and arr is not None:
                rain[i, :] = arr.astype(np.float32, copy=False)
                ok_count += 1
                print(
                    f"[OK] {i+1}/{len(times)} {times[i]} "
                    f"src={result.get('src', '')}",
                    flush=True,
                )
            else:
                missing.append(str(times[i]))
                print(
                    f"[MISSING] {i+1}/{len(times)} {times[i]} "
                    f"reason={result.get('error', '')}",
                    flush=True,
                )

    rain = np.where(np.isfinite(rain), rain, 0.0).astype(np.float32)

    np.savez_compressed(
        out_npz,
        state=np.array(state),
        time=times.astype(str).to_numpy(),
        rain=rain,
        rows=rows,
        cols=cols,
        lon=lon,
        lat=lat,
        missing_times=np.array(missing, dtype=str),
    )

    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"saved       : {out_npz}")
    print(f"rain shape  : {rain.shape}")
    print(f"hours ok    : {ok_count}")
    print(f"hours missing: {len(missing)}")

    return out_npz