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
    min_pixel_value: float = 3.0,
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



def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if a.size < 5:
        return np.nan

    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _percentiles(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.nanquantile(values, 0.50)),
        float(np.nanquantile(values, 0.75)),
        float(np.nanquantile(values, 0.90)),
    )


def _empty_site_efficient_summary(site_id: str) -> dict:
    return {
        "site_id": str(site_id),
        "n_events_all": 0,
        "n_events_rain_response": 0,
        "n_events_good": 0,
        "corr_pixel_delta": np.nan,
        "corr_basin_delta": np.nan,
        "corr_combined_delta": np.nan,
        "pixel_weight": np.nan,
        "basin_weight": np.nan,
        "pixel_p50": np.nan,
        "pixel_p75": np.nan,
        "pixel_p90": np.nan,
        "basin_p50": np.nan,
        "basin_p75": np.nan,
        "basin_p90": np.nan,
        "delta_p50": np.nan,
        "delta_p75": np.nan,
        "delta_p90": np.nan,
        "p50_eff_pixel": np.nan,
        "p50_eff_basin": np.nan,
        "basin_peak_time": np.nan,
        "pixel_peak_time": np.nan,
        "max_delta_basin_peak_time": np.nan,
        "max_delta_pixel_peak_time": np.nan,
        "fastest_event_id": -1,
        "fastest_response_hr": np.nan,
        "fastest_delta_water_stage": np.nan,
        "fastest_basin_accumulation": np.nan,
        "fastest_pixel_accumulation": np.nan,
        "max_delta_event_id": -1,
        "max_delta_response_hr": np.nan,
        "max_delta_water_stage": np.nan,
        "max_delta_basin_accumulation": np.nan,
        "max_delta_pixel_accumulation": np.nan,
        "pixel_ref": np.array([], dtype=np.float32),
        "basin_ref": np.array([], dtype=np.float32),
        "delta_ref": np.array([], dtype=np.float32),
    }


def build_site_efficient_event_summary_from_basin_file(
    basin_fp: Path,
    *,
    pixel_fp: Path | None = None,
    min_events: int = 5,
) -> dict:
    """
    Build one compact efficient-event reference for a site.

    Basin-event history provides one event-level row with:
    - delta_water_stage
    - basin_accumulation
    - max_pixel_accumulation

    Pixel-event history provides pixel-level rows with:
    - time_to_stage_peak_hr
    - pixel_accumulation
    - pixel_value

    For the app popup, two positive-time diagnostics are cached:
    - basin_peak_time: minimum positive time_to_stage_peak_hr across all pixels
      in the event.
    - pixel_peak_time: time_to_stage_peak_hr for the strongest historical pixel
      in the event, using highest pixel_accumulation and then highest pixel_value.

    Both use values strictly greater than zero.
    """
    basin_fp = Path(basin_fp)
    site_id = basin_fp.name.replace("_basin_event_history.parquet", "")

    out = _empty_site_efficient_summary(site_id)

    if not basin_fp.exists():
        return out

    required = [
        "event_id",
        "delta_water_stage",
        "basin_accumulation",
        "max_pixel_accumulation",
    ]

    optional = []

    try:
        cols = pd.read_parquet(basin_fp).columns.tolist()
        read_cols = [c for c in required + optional if c in cols]

        if not set(required).issubset(read_cols):
            missing = sorted(set(required) - set(read_cols))
            print(f"[WARN] missing columns in {basin_fp}: {missing}")
            return out

        df = pd.read_parquet(basin_fp, columns=read_cols)

    except Exception as e:
        print(f"[WARN] could not read efficient reference {basin_fp}: {type(e).__name__}: {e}")
        return out

    if df.empty:
        return out

    # Add pixel-level timing diagnostics from pixel history.
    #
    # basin_peak_time:
    #   Minimum positive time_to_stage_peak_hr across all historical pixels
    #   for the event. This is the fastest positive basin response time.
    #
    # pixel_peak_time:
    #   time_to_stage_peak_hr for the strongest historical pixel in the event,
    #   where strongest means highest pixel_accumulation, then highest pixel_value.
    if pixel_fp is not None and Path(pixel_fp).exists():
        try:
            pix = pd.read_parquet(
                pixel_fp,
                columns=[
                    "event_id",
                    "time_to_stage_peak_hr",
                    "pixel_accumulation",
                    "pixel_value",
                ],
            )

            pix = pix[
                np.isfinite(pix["event_id"])
                & np.isfinite(pix["time_to_stage_peak_hr"])
                & np.isfinite(pix["pixel_accumulation"])
                & np.isfinite(pix["pixel_value"])
                & (pix["time_to_stage_peak_hr"] > 0)
            ].copy()

            if not pix.empty:
                basin_time = (
                    pix.groupby("event_id", as_index=False)
                    .agg(basin_peak_time=("time_to_stage_peak_hr", "min"))
                )

                strongest_pixel_time = (
                    pix.sort_values(
                        ["event_id", "pixel_accumulation", "pixel_value"],
                        ascending=[True, False, False],
                    )
                    .groupby("event_id", as_index=False)
                    .first()[["event_id", "time_to_stage_peak_hr"]]
                    .rename(columns={"time_to_stage_peak_hr": "pixel_peak_time"})
                )

                timing = basin_time.merge(
                    strongest_pixel_time,
                    on="event_id",
                    how="left",
                )

                df = df.merge(timing, on="event_id", how="left")
            else:
                df["basin_peak_time"] = np.nan
                df["pixel_peak_time"] = np.nan

        except Exception as e:
            print(
                f"[WARN] could not read pixel timing {pixel_fp}: "
                f"{type(e).__name__}: {e}"
            )
            df["basin_peak_time"] = np.nan
            df["pixel_peak_time"] = np.nan
    else:
        df["basin_peak_time"] = np.nan
        df["pixel_peak_time"] = np.nan

    if "basin_peak_time" not in df.columns:
        df["basin_peak_time"] = np.nan

    if "pixel_peak_time" not in df.columns:
        df["pixel_peak_time"] = np.nan

    df = df[
        np.isfinite(df["event_id"])
        & np.isfinite(df["delta_water_stage"])
        & np.isfinite(df["basin_accumulation"])
        & np.isfinite(df["max_pixel_accumulation"])
        & (df["basin_accumulation"] > 0)
        & (df["max_pixel_accumulation"] > 0)
    ].copy()

    if df.empty:
        return out

    # Keep only strictly positive timing values for both diagnostics.
    df["basin_peak_time_valid"] = df["basin_peak_time"].where(
        np.isfinite(df["basin_peak_time"])
        & (df["basin_peak_time"] > 0),
        np.nan,
    )

    df["pixel_peak_time_valid"] = df["pixel_peak_time"].where(
        np.isfinite(df["pixel_peak_time"])
        & (df["pixel_peak_time"] > 0),
        np.nan,
    )

    # Robust aggregation in case a basin file ever contains repeated event IDs.
    ev = (
        df.groupby("event_id", as_index=False)
        .agg(
            pixel=("max_pixel_accumulation", "max"),
            basin=("basin_accumulation", "max"),
            delta=("delta_water_stage", "max"),
            basin_peak_time=("basin_peak_time_valid", "min"),
            pixel_peak_time=("pixel_peak_time_valid", "min"),
        )
    )

    event_ids = ev["event_id"].to_numpy(dtype=np.int64)
    event_pixel = ev["pixel"].to_numpy(dtype=np.float32)
    event_basin = ev["basin"].to_numpy(dtype=np.float32)
    event_delta = ev["delta"].to_numpy(dtype=np.float32)
    event_basin_peak_time = ev["basin_peak_time"].to_numpy(dtype=np.float32)
    event_pixel_peak_time = ev["pixel_peak_time"].to_numpy(dtype=np.float32)

    valid = (
        np.isfinite(event_pixel)
        & np.isfinite(event_basin)
        & np.isfinite(event_delta)
        & (event_pixel > 0)
        & (event_basin > 0)
    )

    event_ids = event_ids[valid]
    event_pixel = event_pixel[valid]
    event_basin = event_basin[valid]
    event_delta = event_delta[valid]
    event_basin_peak_time = event_basin_peak_time[valid]
    event_pixel_peak_time = event_pixel_peak_time[valid]

    out["n_events_all"] = int(event_delta.size)

    if event_delta.size < min_events:
        return out

    basin_p25 = float(np.nanquantile(event_basin, 0.25))
    pixel_p25 = float(np.nanquantile(event_pixel, 0.25))

    rain_response = (
        (event_delta > 0)
        & (
            (event_basin >= basin_p25)
            | (event_pixel >= pixel_p25)
        )
    )

    rr_ids = event_ids[rain_response]
    rr_pixel = event_pixel[rain_response]
    rr_basin = event_basin[rain_response]
    rr_delta = event_delta[rain_response]
    rr_basin_peak_time = event_basin_peak_time[rain_response]
    rr_pixel_peak_time = event_pixel_peak_time[rain_response]

    out["n_events_rain_response"] = int(rr_delta.size)

    if rr_delta.size < min_events:
        return out

    eff_pixel = rr_delta / rr_pixel
    eff_basin = rr_delta / rr_basin

    p50_eff_pixel = float(np.nanquantile(eff_pixel, 0.50))
    p50_eff_basin = float(np.nanquantile(eff_basin, 0.50))

    good = (
        (eff_pixel >= p50_eff_pixel)
        | (eff_basin >= p50_eff_basin)
    )

    good_ids = rr_ids[good]
    good_pixel = rr_pixel[good].astype(np.float32)
    good_basin = rr_basin[good].astype(np.float32)
    good_delta = rr_delta[good].astype(np.float32)
    good_basin_peak_time = rr_basin_peak_time[good].astype(np.float32)
    good_pixel_peak_time = rr_pixel_peak_time[good].astype(np.float32)

    out["n_events_good"] = int(good_delta.size)
    out["p50_eff_pixel"] = p50_eff_pixel
    out["p50_eff_basin"] = p50_eff_basin

    if good_delta.size < min_events:
        return out

    corr_pixel = _safe_corr(good_pixel, good_delta)
    corr_basin = _safe_corr(good_basin, good_delta)

    pixel_std = np.nanstd(good_pixel)
    basin_std = np.nanstd(good_basin)

    z_pixel = (
        (good_pixel - np.nanmean(good_pixel)) / pixel_std
        if pixel_std > 0
        else np.full_like(good_pixel, np.nan)
    )
    z_basin = (
        (good_basin - np.nanmean(good_basin)) / basin_std
        if basin_std > 0
        else np.full_like(good_basin, np.nan)
    )

    corr_combined = _safe_corr(z_pixel + z_basin, good_delta)

    cp = max(corr_pixel, 0.0) if np.isfinite(corr_pixel) else 0.0
    cb = max(corr_basin, 0.0) if np.isfinite(corr_basin) else 0.0

    if (cp + cb) > 0:
        pixel_weight = cp / (cp + cb)
        basin_weight = cb / (cp + cb)
    else:
        pixel_weight = 0.5
        basin_weight = 0.5

    pixel_p50, pixel_p75, pixel_p90 = _percentiles(good_pixel)
    basin_p50, basin_p75, basin_p90 = _percentiles(good_basin)
    delta_p50, delta_p75, delta_p90 = _percentiles(good_delta)

    # Fastest historical severe event uses the pixel-stage response time.
    severe_mask = (
        np.isfinite(good_pixel_peak_time)
        & (good_pixel_peak_time > 0)
        & np.isfinite(good_delta)
        & (good_delta >= delta_p90)
    )

    if np.any(severe_mask):
        severe_pos = np.flatnonzero(severe_mask)
        fastest_local = severe_pos[int(np.nanargmin(good_pixel_peak_time[severe_mask]))]
    else:
        fastest_local = int(np.nanargmax(good_delta))

    max_delta_local = int(np.nanargmax(good_delta))

    out.update(
        {
            "corr_pixel_delta": corr_pixel,
            "corr_basin_delta": corr_basin,
            "corr_combined_delta": corr_combined,
            "pixel_weight": float(pixel_weight),
            "basin_weight": float(basin_weight),
            "pixel_p50": pixel_p50,
            "pixel_p75": pixel_p75,
            "pixel_p90": pixel_p90,
            "basin_p50": basin_p50,
            "basin_p75": basin_p75,
            "basin_p90": basin_p90,
            "delta_p50": delta_p50,
            "delta_p75": delta_p75,
            "delta_p90": delta_p90,
            "fastest_event_id": int(good_ids[fastest_local]),
            "fastest_response_hr": float(good_pixel_peak_time[fastest_local]),
            "fastest_delta_water_stage": float(good_delta[fastest_local]),
            "fastest_basin_accumulation": float(good_basin[fastest_local]),
            "fastest_pixel_accumulation": float(good_pixel[fastest_local]),
            "basin_peak_time": float(good_basin_peak_time[fastest_local]),
            "pixel_peak_time": float(good_pixel_peak_time[fastest_local]),
            "max_delta_event_id": int(good_ids[max_delta_local]),
            "max_delta_response_hr": float(good_pixel_peak_time[max_delta_local]),
            "max_delta_water_stage": float(good_delta[max_delta_local]),
            "max_delta_basin_accumulation": float(good_basin[max_delta_local]),
            "max_delta_pixel_accumulation": float(good_pixel[max_delta_local]),
            "max_delta_basin_peak_time": float(good_basin_peak_time[max_delta_local]),
            "max_delta_pixel_peak_time": float(good_pixel_peak_time[max_delta_local]),
            "pixel_ref": good_pixel,
            "basin_ref": good_basin,
            "delta_ref": good_delta,
        }
    )

    return out


def build_state_efficient_event_reference_npz(
    *,
    state: str,
    idx: dict,
    basin_files: list[Path],
    out_fp: Path,
    min_events: int = 5,
) -> Path:
    """
    Build a compact state-level cache for efficient-event percentile alerts.

    This does NOT replace the existing pixel_event_index. It adds a new
    companion NPZ that current_alerts.py can load quickly.
    """
    state = state.upper()
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    site_ids = idx["site_ids"].astype(str)
    site_to_file = {
        Path(fp).name.replace("_basin_event_history.parquet", ""): Path(fp)
        for fp in basin_files
    }

    pixel_dir = out_fp.parent.parent / "pixel_event_history"
    site_to_pixel_file = {
        fp.name.replace("_pixel_event_history.parquet", ""): fp
        for fp in pixel_dir.glob("*_pixel_event_history.parquet")
    }

    n_sites = len(site_ids)

    scalar_float_fields = [
        "corr_pixel_delta",
        "corr_basin_delta",
        "corr_combined_delta",
        "pixel_weight",
        "basin_weight",
        "pixel_p50",
        "pixel_p75",
        "pixel_p90",
        "basin_p50",
        "basin_p75",
        "basin_p90",
        "delta_p50",
        "delta_p75",
        "delta_p90",
        "p50_eff_pixel",
        "p50_eff_basin",
        "basin_peak_time",
        "pixel_peak_time",
        "max_delta_basin_peak_time",
        "max_delta_pixel_peak_time",
        "fastest_response_hr",
        "fastest_delta_water_stage",
        "fastest_basin_accumulation",
        "fastest_pixel_accumulation",
        "max_delta_response_hr",
        "max_delta_water_stage",
        "max_delta_basin_accumulation",
        "max_delta_pixel_accumulation",
    ]

    scalar_int_fields = [
        "n_events_all",
        "n_events_rain_response",
        "n_events_good",
        "fastest_event_id",
        "max_delta_event_id",
    ]

    scalars_float = {
        k: np.full(n_sites, np.nan, dtype=np.float32)
        for k in scalar_float_fields
    }
    scalars_int = {
        k: np.full(n_sites, -1, dtype=np.int32)
        for k in scalar_int_fields
    }

    pixel_ref_chunks = []
    basin_ref_chunks = []
    delta_ref_chunks = []
    ref_ptr = np.zeros(n_sites + 1, dtype=np.int64)

    print("=" * 100)
    print("BUILD STATE EFFICIENT EVENT REFERENCE NPZ")
    print("=" * 100)
    print(f"state       : {state}")
    print(f"basin files : {len(basin_files)}")
    print(f"sites       : {n_sites}")
    print(f"min_events  : {min_events}")
    print(f"output      : {out_fp}")
    print("=" * 100)

    total_ref = 0
    ok_sites = 0

    for i, site_id in enumerate(site_ids):
        fp = site_to_file.get(str(site_id))

        if fp is None:
            summary = _empty_site_efficient_summary(str(site_id))
        else:
            summary = build_site_efficient_event_summary_from_basin_file(
                fp,
                pixel_fp=site_to_pixel_file.get(str(site_id)),
                min_events=min_events,
            )

        for k in scalar_float_fields:
            scalars_float[k][i] = summary[k]

        for k in scalar_int_fields:
            scalars_int[k][i] = summary[k]

        px = summary["pixel_ref"].astype(np.float32)
        bs = summary["basin_ref"].astype(np.float32)
        dl = summary["delta_ref"].astype(np.float32)

        if px.size:
            ok_sites += 1
            pixel_ref_chunks.append(px)
            basin_ref_chunks.append(bs)
            delta_ref_chunks.append(dl)
            total_ref += int(px.size)

        ref_ptr[i + 1] = total_ref

        if (i + 1) % 250 == 0 or (i + 1) == n_sites:
            print(
                f"[EFFICIENT REF] {i + 1:5d}/{n_sites} "
                f"ok_sites={ok_sites:,} ref_events={total_ref:,}"
            )

    if pixel_ref_chunks:
        pixel_ref = np.concatenate(pixel_ref_chunks).astype(np.float32)
        basin_ref = np.concatenate(basin_ref_chunks).astype(np.float32)
        delta_ref = np.concatenate(delta_ref_chunks).astype(np.float32)
    else:
        pixel_ref = np.array([], dtype=np.float32)
        basin_ref = np.array([], dtype=np.float32)
        delta_ref = np.array([], dtype=np.float32)

    save_dict = {
        "state": np.array(state),
        "site_ids": site_ids.astype("U"),
        "min_events": np.array(min_events, dtype=np.int32),
        "ref_ptr": ref_ptr,
        "pixel_ref": pixel_ref,
        "basin_ref": basin_ref,
        "delta_ref": delta_ref,
    }

    save_dict.update(scalars_float)
    save_dict.update(scalars_int)

    np.savez_compressed(out_fp, **save_dict)

    print("=" * 100)
    print("EFFICIENT EVENT REFERENCE DONE")
    print("=" * 100)
    print(f"output     : {out_fp}")
    print(f"ok sites   : {ok_sites:,}/{n_sites:,}")
    print(f"ref events : {len(pixel_ref):,}")

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
    min_pixel_value: float = 3.0,
    only_stage_response_p50: bool = False,
    index_batch_size: int = 10,
) -> dict[str, Path | None]:
    state = state.upper()
    out_dir = Path(out_dir)

    site_basin_dir = out_dir / "basin_event_history"
    site_pixel_dir = out_dir / "pixel_event_history"
    index_dir = out_dir / "state_pixel_event_index"
    efficient_ref_dir = out_dir / "state_efficient_event_reference"

    site_basin_dir.mkdir(parents=True, exist_ok=True)
    site_pixel_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    efficient_ref_dir.mkdir(parents=True, exist_ok=True)

    index_fp = index_dir / f"{state}_pixel_event_index.npz"
    efficient_ref_fp = efficient_ref_dir / f"{state}_efficient_event_reference.npz"

    if index_fp.exists() and efficient_ref_fp.exists() and not overwrite_index:
        return {"pixel_event_index": index_fp, "efficient_event_reference": efficient_ref_fp}

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
    print(f"efficient output  : {efficient_ref_fp}")
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

    efficient_ref_fp = build_state_efficient_event_reference_npz(
        state=state,
        idx=idx,
        basin_files=sorted(basin_files),
        out_fp=efficient_ref_fp,
        min_events=5,
    )

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"basin files       : {len(basin_files)}")
    print(f"pixel files       : {len(pixel_files)}")
    print(f"pixel event index : {index_fp}")
    print(f"efficient ref     : {efficient_ref_fp}")

    return {
        "pixel_event_index": index_fp,
        "efficient_event_reference": efficient_ref_fp,
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
        min_pixel_value=3.0,
        only_stage_response_p50=False,
        index_batch_size=10,
    )
