from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import gc

import numpy as np
import pandas as pd

from .historical_summary import build_site_historical_summary


def load_state_basin_index(fp: Path) -> dict:
    idx = np.load(fp, allow_pickle=True)

    return {
        "state": str(idx["state"]),
        "site_ids": idx["site_ids"].astype(str),
        "rows": idx["rows"].astype(np.int32),
        "cols": idx["cols"].astype(np.int32),
        "lon": idx["lon"].astype(np.float32),
        "lat": idx["lat"].astype(np.float32),
        "basin_ptr": idx["basin_ptr"].astype(np.int64),
        "basin_indices": idx["basin_indices"].astype(np.int32),
        "basin_n_pixels": idx["basin_n_pixels"].astype(np.int32),
        "n_state_pixels": int(idx["n_state_pixels"]),
    }


def attach_state_pixel_ids(
    pixel_fp: Path,
    *,
    site_id: str,
    idx: dict,
    overwrite: bool = True,
) -> None:
    pixel_fp = Path(pixel_fp)

    if not pixel_fp.exists():
        return

    pixel = pd.read_parquet(pixel_fp)

    if pixel.empty:
        return

    if {"pixel_id_state", "row", "col"}.issubset(pixel.columns):
        return

    matches = np.where(idx["site_ids"] == str(site_id))[0]

    if len(matches) == 0:
        print(f"[WARN] site_id not found in state_basin_index: {site_id}")
        return

    basin_i = int(matches[0])
    a = int(idx["basin_ptr"][basin_i])
    b = int(idx["basin_ptr"][basin_i + 1])

    state_pixel_ids = idx["basin_indices"][a:b]

    if len(state_pixel_ids) == 0:
        print(f"[WARN] empty basin index for site_id: {site_id}")
        return

    max_pixel_id_basin = int(pixel["pixel_id_basin"].max())

    if max_pixel_id_basin >= len(state_pixel_ids):
        raise ValueError(
            f"pixel_id_basin exceeds state_basin_index length for {site_id}. "
            f"max pixel_id_basin={max_pixel_id_basin}, basin pixels={len(state_pixel_ids)}"
        )

    pixel_id_basin = pixel["pixel_id_basin"].to_numpy(dtype=np.int64)
    pixel_id_state = state_pixel_ids[pixel_id_basin]

    pixel["pixel_id_state"] = pixel_id_state.astype(np.int32)
    pixel["row"] = idx["rows"][pixel_id_state].astype(np.int32)
    pixel["col"] = idx["cols"][pixel_id_state].astype(np.int32)

    ordered_cols = [
        "state",
        "site_id",
        "event_id",
        "date_peak",
        "event_start",
        "event_end",
        "pixel_id_state",
        "pixel_id_basin",
        "row",
        "col",
        "lat",
        "lon",
        "pixel_value",
        "pixel_accumulation",
        "basin_accumulation",
        "delta_water_stage",
        "delta_water_stage_p50",
        "is_stage_response_p50",
        "time_to_rain_peak_accumulation_hr",
        "time_to_stage_peak_hr",
        "is_strong_pixel",
        "strong_rain_threshold_mm_h",
    ]

    existing = [c for c in ordered_cols if c in pixel.columns]
    extra = [c for c in pixel.columns if c not in existing]
    pixel = pixel[existing + extra]

    if overwrite:
        pixel.to_parquet(pixel_fp, index=False)


def _build_one_site_worker(args: tuple) -> tuple[str, str, str | None, str | None]:
    base_dir, state, out_dir, site_id, overwrite_sites = args

    out_dir = Path(out_dir)
    site_id = str(site_id)

    site_basin_fp = out_dir / "basin_event_history" / f"{site_id}_basin_event_history.parquet"
    site_pixel_fp = out_dir / "pixel_event_history" / f"{site_id}_pixel_event_history.parquet"

    if site_basin_fp.exists() and site_pixel_fp.exists() and not overwrite_sites:
        return site_id, "SKIP", str(site_basin_fp), str(site_pixel_fp)

    try:
        result = build_site_historical_summary(
            base_dir=Path(base_dir),
            site_id=site_id,
            state=str(state),
            out_dir=out_dir,
            overwrite=bool(overwrite_sites),
        )

        if result is None:
            return site_id, "EMPTY", None, None

        basin_fp = str(result["basin"]) if result.get("basin") else None
        pixel_fp = str(result["pixel"]) if result.get("pixel") else None

        return site_id, "OK", basin_fp, pixel_fp

    except Exception as e:
        return site_id, f"ERROR {type(e).__name__}: {e}", None, None


def build_state_pixel_event_index_npz(
    *,
    state: str,
    idx: dict,
    pixel_files: list[Path],
    out_fp: Path,
    min_pixel_value: float = 7.5,
    only_stage_response_p50: bool = True,
    batch_size: int = 10,
) -> Path:
    state = state.upper()
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "site_id",
        "event_id",
        "pixel_id_state",
        "pixel_id_basin",
        "row",
        "col",
        "lat",
        "lon",
        "pixel_value",
        "pixel_accumulation",
        "basin_accumulation",
        "delta_water_stage",
        "time_to_rain_peak_accumulation_hr",
        "time_to_stage_peak_hr",
        "is_stage_response_p50",
    ]

    event_id_chunks = []
    site_index_chunks = []
    pixel_id_state_chunks = []
    pixel_id_basin_chunks = []
    pixel_value_chunks = []
    pixel_accumulation_chunks = []
    basin_accumulation_chunks = []
    delta_water_stage_chunks = []
    time_to_rain_peak_accumulation_chunks = []
    time_to_stage_peak_chunks = []

    site_to_index = {s: i for i, s in enumerate(idx["site_ids"].astype(str))}

    total_rows = 0
    kept_rows = 0

    print("=" * 100)
    print("BUILD STATE PIXEL EVENT NPZ INDEX")
    print("=" * 100)
    print(f"state                  : {state}")
    print(f"pixel files            : {len(pixel_files)}")
    print(f"min_pixel_value        : {min_pixel_value}")
    print(f"only_stage_response_p50: {only_stage_response_p50}")
    print(f"output                 : {out_fp}")
    print("=" * 100)

    for i in range(0, len(pixel_files), batch_size):
        batch_files = pixel_files[i : i + batch_size]
        dfs = []

        for fp in batch_files:
            try:
                df = pd.read_parquet(fp, columns=required_cols)

                if df.empty:
                    continue

                total_rows += len(df)

                if only_stage_response_p50:
                    df = df[df["is_stage_response_p50"] == True]

                df = df[df["pixel_value"] >= min_pixel_value]

                if df.empty:
                    continue

                dfs.append(df)

            except Exception as e:
                print(f"[WARN] could not read/index {fp}: {type(e).__name__}: {e}")

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        df["site_index"] = (
            df["site_id"]
            .astype(str)
            .map(site_to_index)
            .astype("int32")
        )

        df = df.dropna(subset=["site_index", "pixel_id_state"])

        df = df.sort_values(
            ["pixel_id_state", "site_index", "event_id"]
        ).reset_index(drop=True)

        event_id_chunks.append(df["event_id"].to_numpy(dtype=np.int32))
        site_index_chunks.append(df["site_index"].to_numpy(dtype=np.int32))
        pixel_id_state_chunks.append(df["pixel_id_state"].to_numpy(dtype=np.int32))
        pixel_id_basin_chunks.append(df["pixel_id_basin"].to_numpy(dtype=np.int32))

        pixel_value_chunks.append(df["pixel_value"].to_numpy(dtype=np.float32))
        pixel_accumulation_chunks.append(df["pixel_accumulation"].to_numpy(dtype=np.float32))
        basin_accumulation_chunks.append(df["basin_accumulation"].to_numpy(dtype=np.float32))
        delta_water_stage_chunks.append(df["delta_water_stage"].to_numpy(dtype=np.float32))

        time_to_rain_peak_accumulation_chunks.append(
            df["time_to_rain_peak_accumulation_hr"].to_numpy(dtype=np.float32)
        )

        time_to_stage_peak_chunks.append(
            df["time_to_stage_peak_hr"].to_numpy(dtype=np.float32)
        )

        kept_rows += len(df)

        print(
            f"[INDEX] {min(i + batch_size, len(pixel_files)):5d}/{len(pixel_files)} "
            f"batch_kept={len(df):,} total_kept={kept_rows:,}"
        )

        del dfs, df
        gc.collect()

    if not event_id_chunks:
        raise RuntimeError(f"No rows kept for NPZ index in {state}")

    event_id = np.concatenate(event_id_chunks).astype(np.int32)
    site_index = np.concatenate(site_index_chunks).astype(np.int32)
    pixel_id_state_event = np.concatenate(pixel_id_state_chunks).astype(np.int32)
    pixel_id_basin = np.concatenate(pixel_id_basin_chunks).astype(np.int32)

    pixel_value = np.concatenate(pixel_value_chunks).astype(np.float32)
    pixel_accumulation = np.concatenate(pixel_accumulation_chunks).astype(np.float32)
    basin_accumulation = np.concatenate(basin_accumulation_chunks).astype(np.float32)
    delta_water_stage = np.concatenate(delta_water_stage_chunks).astype(np.float32)

    time_to_rain_peak_accumulation_hr = np.concatenate(
        time_to_rain_peak_accumulation_chunks
    ).astype(np.float32)

    time_to_stage_peak_hr = np.concatenate(
        time_to_stage_peak_chunks
    ).astype(np.float32)

    order = np.lexsort((event_id, site_index, pixel_id_state_event))

    event_id = event_id[order]
    site_index = site_index[order]
    pixel_id_state_event = pixel_id_state_event[order]
    pixel_id_basin = pixel_id_basin[order]

    pixel_value = pixel_value[order]
    pixel_accumulation = pixel_accumulation[order]
    basin_accumulation = basin_accumulation[order]
    delta_water_stage = delta_water_stage[order]
    time_to_rain_peak_accumulation_hr = time_to_rain_peak_accumulation_hr[order]
    time_to_stage_peak_hr = time_to_stage_peak_hr[order]

    unique_pixel_id_state, first_idx = np.unique(
        pixel_id_state_event,
        return_index=True,
    )

    n_unique_pixels = len(unique_pixel_id_state)

    event_ptr = np.zeros(n_unique_pixels + 1, dtype=np.int64)
    event_ptr[:-1] = first_idx
    event_ptr[-1] = len(pixel_id_state_event)

    pixel_rows = idx["rows"][unique_pixel_id_state].astype(np.int32)
    pixel_cols = idx["cols"][unique_pixel_id_state].astype(np.int32)
    pixel_lat = idx["lat"][unique_pixel_id_state].astype(np.float32)
    pixel_lon = idx["lon"][unique_pixel_id_state].astype(np.float32)

    np.savez_compressed(
        out_fp,
        state=np.array(state),
        site_ids=idx["site_ids"].astype("U"),
        n_state_pixels=np.array(idx["n_state_pixels"], dtype=np.int32),
        strong_rain_threshold_mm_h=np.array(min_pixel_value, dtype=np.float32),
        only_stage_response_p50=np.array(only_stage_response_p50),
        pixel_id_state=unique_pixel_id_state.astype(np.int32),
        row=pixel_rows,
        col=pixel_cols,
        lat=pixel_lat,
        lon=pixel_lon,
        event_ptr=event_ptr,
        event_id=event_id,
        site_index=site_index,
        pixel_id_basin=pixel_id_basin,
        pixel_value=pixel_value,
        pixel_accumulation=pixel_accumulation,
        basin_accumulation=basin_accumulation,
        delta_water_stage=delta_water_stage,
        time_to_rain_peak_accumulation_hr=time_to_rain_peak_accumulation_hr,
        time_to_stage_peak_hr=time_to_stage_peak_hr,
    )

    print("=" * 100)
    print("NPZ INDEX DONE")
    print("=" * 100)
    print(f"output          : {out_fp}")
    print(f"raw rows read   : {total_rows:,}")
    print(f"rows kept       : {kept_rows:,}")
    print(f"unique pixels   : {n_unique_pixels:,}")
    print(f"events indexed  : {len(event_id):,}")

    return out_fp


def build_state_historical_summary_parallel(
    *,
    base_dir: Path,
    state: str,
    state_basin_index_fp: Path,
    out_dir: Path,
    workers: int = 4,
    overwrite_sites: bool = False,
    overwrite_index: bool = True,
    min_pixel_value: float = 7.5,
    only_stage_response_p50: bool = True,
    index_batch_size: int = 10,
) -> dict[str, Path | None]:
    state = state.upper()
    out_dir = Path(out_dir)

    site_basin_dir = out_dir / "basin_event_history"
    site_pixel_dir = out_dir / "pixel_event_history"
    index_dir = out_dir / "state_pixel_event_index"

    site_basin_dir.mkdir(parents=True, exist_ok=True)
    site_pixel_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_fp = index_dir / f"{state}_pixel_event_index.npz"

    if index_fp.exists() and not overwrite_index:
        return {"pixel_event_index": index_fp}

    idx = load_state_basin_index(state_basin_index_fp)
    site_ids = idx["site_ids"].astype(str)

    print("=" * 100)
    print("BUILD STATE HISTORICAL SUMMARY PARALLEL")
    print("=" * 100)
    print(f"state             : {state}")
    print(f"basins            : {len(site_ids)}")
    print(f"workers           : {workers}")
    print(f"state_basin_index : {state_basin_index_fp}")
    print(f"out_dir           : {out_dir}")
    print(f"index output      : {index_fp}")
    print("=" * 100)

    jobs = [
        (str(base_dir), state, str(out_dir), str(site_id), bool(overwrite_sites))
        for site_id in site_ids
    ]

    pixel_files: list[Path] = []
    basin_files: list[Path] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_build_one_site_worker, job) for job in jobs]

        for n, fut in enumerate(as_completed(futures), start=1):
            site_id, status, basin_fp, pixel_fp = fut.result()

            if status in {"OK", "SKIP"}:
                if basin_fp and Path(basin_fp).exists():
                    basin_files.append(Path(basin_fp))

                if pixel_fp and Path(pixel_fp).exists():
                    pixel_path = Path(pixel_fp)

                    try:
                        attach_state_pixel_ids(
                            pixel_path,
                            site_id=site_id,
                            idx=idx,
                            overwrite=True,
                        )

                        pixel_files.append(pixel_path)

                    except Exception as e:
                        print(
                            f"[WARN] attach_state_pixel_ids failed for {site_id}: "
                            f"{type(e).__name__}: {e}"
                        )

            print(f"[{n:5d}/{len(futures)}] {site_id} {status}")
            gc.collect()

    if not pixel_files:
        raise RuntimeError(f"No pixel historical files were created for {state}")

    index_fp = build_state_pixel_event_index_npz(
        state=state,
        idx=idx,
        pixel_files=sorted(pixel_files),
        out_fp=index_fp,
        min_pixel_value=min_pixel_value,
        only_stage_response_p50=only_stage_response_p50,
        batch_size=index_batch_size,
    )

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"basin files       : {len(basin_files)}")
    print(f"pixel files       : {len(pixel_files)}")
    print(f"pixel event index : {index_fp}")

    return {
        "pixel_event_index": index_fp,
        "basin_event_dir": site_basin_dir,
        "pixel_event_dir": site_pixel_dir,
    }


def build_state_historical_summary(
    *,
    base_dir: Path,
    state: str,
    state_basin_index_fp: Path,
    out_dir: Path,
    overwrite_sites: bool = False,
    overwrite_index: bool = True,
) -> dict[str, Path | None]:
    return build_state_historical_summary_parallel(
        base_dir=base_dir,
        state=state,
        state_basin_index_fp=state_basin_index_fp,
        out_dir=out_dir,
        workers=1,
        overwrite_sites=overwrite_sites,
        overwrite_index=overwrite_index,
        min_pixel_value=7.5,
        only_stage_response_p50=True,
        index_batch_size=10,
    )
