from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


STRONG_RAIN_MM_H = 7.5

_WORKER_STATE: dict = {}


def load_npz(fp: Path) -> dict:
    z = np.load(fp, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_state_basin_index(fp: Path) -> dict:
    z = np.load(fp, allow_pickle=True)
    return {
        "state": str(z["state"]),
        "site_ids": z["site_ids"].astype(str),
        "rows": z["rows"].astype(np.int32),
        "cols": z["cols"].astype(np.int32),
        "lon": z["lon"].astype(np.float32),
        "lat": z["lat"].astype(np.float32),
        "basin_ptr": z["basin_ptr"].astype(np.int64),
        "basin_indices": z["basin_indices"].astype(np.int32),
        "basin_n_pixels": z["basin_n_pixels"].astype(np.int32),
        "n_basins": int(z["n_basins"]),
        "n_state_pixels": int(z["n_state_pixels"]),
    }


def load_current_state_rain(fp: Path) -> dict:
    z = np.load(fp, allow_pickle=True)
    rain = z["rain"].astype(np.float32)

    if rain.ndim != 2:
        raise ValueError(f"Expected rain with shape (time, n_state_pixels), got {rain.shape}")

    return {
        "state": str(z["state"]) if "state" in z.files else None,
        "rain": rain,
        "time": z["time"] if "time" in z.files else None,
        "rows": z["rows"].astype(np.int32) if "rows" in z.files else None,
        "cols": z["cols"].astype(np.int32) if "cols" in z.files else None,
        "lat": z["lat"].astype(np.float32) if "lat" in z.files else None,
        "lon": z["lon"].astype(np.float32) if "lon" in z.files else None,
        "missing_times": z["missing_times"] if "missing_times" in z.files else None,
    }


def build_event_lookup(pixel_event_index: dict) -> dict[int, int]:
    pixel_id_state = pixel_event_index["pixel_id_state"].astype(np.int32)
    return {int(pid): i for i, pid in enumerate(pixel_id_state)}


def interpolate_from_matches(
    *,
    hist_pixel_value: np.ndarray,
    hist_basin_accumulation: np.ndarray,
    hist_delta_water_stage: np.ndarray,
    hist_time_to_rain_peak_accumulation_hr: np.ndarray,
    hist_time_to_stage_peak_hr: np.ndarray,
    current_pixel_value: float,
    current_basin_accumulation: float,
    k: int = 5,
) -> dict:
    if len(hist_pixel_value) == 0:
        return {
            "matched": False,
            "matched_n": 0,
            "estimated_delta_water_stage": np.nan,
            "estimated_time_to_rain_peak_accumulation_hr": np.nan,
            "estimated_time_to_stage_peak_hr": np.nan,
            "matched_hist_pixel_value": np.nan,
            "matched_hist_basin_accumulation": np.nan,
            "matched_hist_delta_water_stage": np.nan,
            "match_score": np.nan,
        }

    pv_scale = max(float(current_pixel_value), 1.0)
    ba_scale = max(float(current_basin_accumulation), 1.0)

    score = (
        np.abs(hist_pixel_value - current_pixel_value) / pv_scale
        + np.abs(hist_basin_accumulation - current_basin_accumulation) / ba_scale
    )

    n = min(k, len(score))
    order = np.argpartition(score, n - 1)[:n]
    order = order[np.argsort(score[order])]

    s = score[order]
    weights = 1.0 / (s + 1e-6)
    weights = weights / weights.sum()

    return {
        "matched": True,
        "matched_n": int(n),
        "estimated_delta_water_stage": float(np.sum(hist_delta_water_stage[order] * weights)),
        "estimated_time_to_rain_peak_accumulation_hr": float(
            np.sum(hist_time_to_rain_peak_accumulation_hr[order] * weights)
        ),
        "estimated_time_to_stage_peak_hr": float(
            np.sum(hist_time_to_stage_peak_hr[order] * weights)
        ),
        "matched_hist_pixel_value": float(hist_pixel_value[order[0]]),
        "matched_hist_basin_accumulation": float(hist_basin_accumulation[order[0]]),
        "matched_hist_delta_water_stage": float(hist_delta_water_stage[order[0]]),
        "match_score": float(score[order[0]]),
    }


def classify_alert(
    *,
    current_max_pixel_value: float,
    n_active_pixels: int,
    basin_accumulation_reaches_history: bool,
    matched_pixels: int,
    estimated_delta_water_stage: float,
    severe_delta_threshold: float = 10.0,
    warning_delta_threshold: float = 2.0,
) -> str:
    if current_max_pixel_value < STRONG_RAIN_MM_H or n_active_pixels == 0:
        return "NORMAL"

    if not basin_accumulation_reaches_history:
        return "WATCH"

    if matched_pixels == 0 or not np.isfinite(estimated_delta_water_stage):
        return "WATCH"

    if estimated_delta_water_stage >= severe_delta_threshold:
        return "SEVERE"

    if estimated_delta_water_stage >= warning_delta_threshold:
        return "WARNING"

    return "WATCH"


def _normal_basin_row(
    *,
    state: str,
    site_id: str,
    current_basin_accumulation: float,
    current_max_pixel_value: float,
    current_max_pixel_accumulation: float,
    n_basin_pixels: int,
    strong_threshold: float,
    accumulation_quantile: float,
    warning_delta_threshold: float,
    severe_delta_threshold: float,
) -> dict:
    return {
        "state": state,
        "site_id": site_id,
        "alert_level": "NORMAL",
        "current_basin_accumulation": current_basin_accumulation,
        "current_max_pixel_value": current_max_pixel_value,
        "current_max_pixel_accumulation": current_max_pixel_accumulation,
        "n_basin_pixels": int(n_basin_pixels),
        "n_active_pixels": 0,
        "n_active_pixels_with_history": 0,
        "n_matched_pixels": 0,
        "historical_basin_accumulation_threshold": np.nan,
        "basin_accumulation_reaches_history": False,
        "estimated_delta_water_stage": np.nan,
        "estimated_time_to_rain_peak_accumulation_hr": np.nan,
        "estimated_time_to_stage_peak_hr": np.nan,
        "strong_threshold": strong_threshold,
        "accumulation_quantile": accumulation_quantile,
        "warning_delta_threshold": warning_delta_threshold,
        "severe_delta_threshold": severe_delta_threshold,
    }


def _process_one_basin(
    *,
    basin_i: int,
    site_id: str,
    state: str,
    basin_idx: dict,
    hist: dict,
    hist_lookup: dict[int, int],
    current_pixel_value_state: np.ndarray,
    current_pixel_accum_state: np.ndarray,
    strong_threshold: float,
    k_matches: int,
    accumulation_quantile: float,
    severe_delta_threshold: float,
    warning_delta_threshold: float,
    max_pixels_per_basin_output: int | None,
) -> tuple[dict | None, list[dict], bool]:
    basin_ptr = basin_idx["basin_ptr"]
    basin_indices = basin_idx["basin_indices"]

    a = int(basin_ptr[basin_i])
    b = int(basin_ptr[basin_i + 1])
    basin_pixels = basin_indices[a:b]

    if len(basin_pixels) == 0:
        return None, [], False

    cur_pixel_value = current_pixel_value_state[basin_pixels]
    cur_pixel_accum = current_pixel_accum_state[basin_pixels]

    current_basin_accumulation = float(cur_pixel_accum.sum())
    current_max_pixel_value = float(cur_pixel_value.max())
    current_max_pixel_accumulation = float(cur_pixel_accum.max())

    active_local = np.flatnonzero(cur_pixel_value > strong_threshold)
    n_active_pixels = int(len(active_local))

    if n_active_pixels == 0:
        basin_row = _normal_basin_row(
            state=state,
            site_id=site_id,
            current_basin_accumulation=current_basin_accumulation,
            current_max_pixel_value=current_max_pixel_value,
            current_max_pixel_accumulation=current_max_pixel_accumulation,
            n_basin_pixels=len(basin_pixels),
            strong_threshold=strong_threshold,
            accumulation_quantile=accumulation_quantile,
            warning_delta_threshold=warning_delta_threshold,
            severe_delta_threshold=severe_delta_threshold,
        )
        return basin_row, [], True

    historical_basin_acc_values = []
    active_pixels_with_history = []

    for local_j in active_local:
        pixel_id_state = int(basin_pixels[local_j])
        hist_i = hist_lookup.get(pixel_id_state)

        if hist_i is None:
            continue

        h0 = int(hist["event_ptr"][hist_i])
        h1 = int(hist["event_ptr"][hist_i + 1])

        if h1 <= h0:
            continue

        site_mask = hist["site_index"][h0:h1] == basin_i

        if not np.any(site_mask):
            continue

        rel_idx = np.flatnonzero(site_mask)
        idx0 = h0 + rel_idx

        historical_basin_acc_values.append(hist["basin_accumulation"][idx0])
        active_pixels_with_history.append((local_j, idx0))

    if historical_basin_acc_values:
        historical_basin_acc_all = np.concatenate(historical_basin_acc_values).astype(np.float32)
        historical_basin_acc_threshold = float(
            np.nanquantile(historical_basin_acc_all, accumulation_quantile)
        )
        basin_accumulation_reaches_history = (
            current_basin_accumulation >= historical_basin_acc_threshold
        )
    else:
        historical_basin_acc_threshold = np.nan
        basin_accumulation_reaches_history = False

    matched_delta_values = []
    matched_time_stage_values = []
    matched_time_rain_values = []
    basin_pixel_records = []

    if basin_accumulation_reaches_history:
        for local_j, idx0 in active_pixels_with_history:
            interp = interpolate_from_matches(
                hist_pixel_value=hist["pixel_value"][idx0],
                hist_basin_accumulation=hist["basin_accumulation"][idx0],
                hist_delta_water_stage=hist["delta_water_stage"][idx0],
                hist_time_to_rain_peak_accumulation_hr=hist[
                    "time_to_rain_peak_accumulation_hr"
                ][idx0],
                hist_time_to_stage_peak_hr=hist["time_to_stage_peak_hr"][idx0],
                current_pixel_value=float(cur_pixel_value[local_j]),
                current_basin_accumulation=current_basin_accumulation,
                k=k_matches,
            )

            if not interp["matched"]:
                continue

            pixel_id_state = int(basin_pixels[local_j])

            matched_delta_values.append(interp["estimated_delta_water_stage"])
            matched_time_stage_values.append(interp["estimated_time_to_stage_peak_hr"])
            matched_time_rain_values.append(interp["estimated_time_to_rain_peak_accumulation_hr"])

            basin_pixel_records.append(
                {
                    "state": state,
                    "site_id": site_id,
                    "pixel_id_state": pixel_id_state,
                    "pixel_id_basin": int(local_j),
                    "row": int(basin_idx["rows"][pixel_id_state]),
                    "col": int(basin_idx["cols"][pixel_id_state]),
                    "lat": float(basin_idx["lat"][pixel_id_state]),
                    "lon": float(basin_idx["lon"][pixel_id_state]),
                    "current_pixel_value": float(cur_pixel_value[local_j]),
                    "current_pixel_accumulation": float(cur_pixel_accum[local_j]),
                    "current_basin_accumulation": current_basin_accumulation,
                    "historical_basin_accumulation_threshold": historical_basin_acc_threshold,
                    "estimated_delta_water_stage": interp["estimated_delta_water_stage"],
                    "estimated_time_to_rain_peak_accumulation_hr": interp[
                        "estimated_time_to_rain_peak_accumulation_hr"
                    ],
                    "estimated_time_to_stage_peak_hr": interp[
                        "estimated_time_to_stage_peak_hr"
                    ],
                    "matched_hist_pixel_value": interp["matched_hist_pixel_value"],
                    "matched_hist_basin_accumulation": interp[
                        "matched_hist_basin_accumulation"
                    ],
                    "matched_hist_delta_water_stage": interp[
                        "matched_hist_delta_water_stage"
                    ],
                    "match_score": interp["match_score"],
                    "matched_n": interp["matched_n"],
                }
            )

    matched_pixels = len(basin_pixel_records)

    if matched_delta_values:
        estimated_delta = float(np.nanmax(matched_delta_values))
        estimated_time_stage = float(np.nanmedian(matched_time_stage_values))
        estimated_time_rain = float(np.nanmedian(matched_time_rain_values))
    else:
        estimated_delta = np.nan
        estimated_time_stage = np.nan
        estimated_time_rain = np.nan

    alert_level = classify_alert(
        current_max_pixel_value=current_max_pixel_value,
        n_active_pixels=n_active_pixels,
        basin_accumulation_reaches_history=basin_accumulation_reaches_history,
        matched_pixels=matched_pixels,
        estimated_delta_water_stage=estimated_delta,
        severe_delta_threshold=severe_delta_threshold,
        warning_delta_threshold=warning_delta_threshold,
    )

    basin_row = {
        "state": state,
        "site_id": site_id,
        "alert_level": alert_level,
        "current_basin_accumulation": current_basin_accumulation,
        "current_max_pixel_value": current_max_pixel_value,
        "current_max_pixel_accumulation": current_max_pixel_accumulation,
        "n_basin_pixels": int(len(basin_pixels)),
        "n_active_pixels": n_active_pixels,
        "n_active_pixels_with_history": int(len(active_pixels_with_history)),
        "n_matched_pixels": matched_pixels,
        "historical_basin_accumulation_threshold": historical_basin_acc_threshold,
        "basin_accumulation_reaches_history": bool(basin_accumulation_reaches_history),
        "estimated_delta_water_stage": estimated_delta,
        "estimated_time_to_rain_peak_accumulation_hr": estimated_time_rain,
        "estimated_time_to_stage_peak_hr": estimated_time_stage,
        "strong_threshold": strong_threshold,
        "accumulation_quantile": accumulation_quantile,
        "warning_delta_threshold": warning_delta_threshold,
        "severe_delta_threshold": severe_delta_threshold,
    }

    if basin_pixel_records and max_pixels_per_basin_output is not None:
        basin_pixel_records = sorted(
            basin_pixel_records,
            key=lambda r: (
                r["estimated_delta_water_stage"],
                r["current_pixel_value"],
            ),
            reverse=True,
        )[:max_pixels_per_basin_output]

    return basin_row, basin_pixel_records, False


def _init_worker(
    state: str,
    basin_idx: dict,
    hist: dict,
    hist_lookup: dict[int, int],
    current_pixel_value_state: np.ndarray,
    current_pixel_accum_state: np.ndarray,
    strong_threshold: float,
    k_matches: int,
    accumulation_quantile: float,
    severe_delta_threshold: float,
    warning_delta_threshold: float,
    max_pixels_per_basin_output: int | None,
) -> None:
    global _WORKER_STATE

    _WORKER_STATE = {
        "state": state,
        "basin_idx": basin_idx,
        "hist": hist,
        "hist_lookup": hist_lookup,
        "current_pixel_value_state": current_pixel_value_state,
        "current_pixel_accum_state": current_pixel_accum_state,
        "strong_threshold": strong_threshold,
        "k_matches": k_matches,
        "accumulation_quantile": accumulation_quantile,
        "severe_delta_threshold": severe_delta_threshold,
        "warning_delta_threshold": warning_delta_threshold,
        "max_pixels_per_basin_output": max_pixels_per_basin_output,
    }


def _process_one_basin_from_worker_state(
    task: tuple[int, str],
) -> tuple[int, dict | None, list[dict], bool]:
    basin_i, site_id = task

    basin_row, basin_pixel_records, skipped_normal = _process_one_basin(
        basin_i=basin_i,
        site_id=site_id,
        state=_WORKER_STATE["state"],
        basin_idx=_WORKER_STATE["basin_idx"],
        hist=_WORKER_STATE["hist"],
        hist_lookup=_WORKER_STATE["hist_lookup"],
        current_pixel_value_state=_WORKER_STATE["current_pixel_value_state"],
        current_pixel_accum_state=_WORKER_STATE["current_pixel_accum_state"],
        strong_threshold=_WORKER_STATE["strong_threshold"],
        k_matches=_WORKER_STATE["k_matches"],
        accumulation_quantile=_WORKER_STATE["accumulation_quantile"],
        severe_delta_threshold=_WORKER_STATE["severe_delta_threshold"],
        warning_delta_threshold=_WORKER_STATE["warning_delta_threshold"],
        max_pixels_per_basin_output=_WORKER_STATE["max_pixels_per_basin_output"],
    )

    return basin_i, basin_row, basin_pixel_records, skipped_normal


def compute_current_alerts_for_state(
    *,
    state: str,
    current_rain_npz: Path,
    state_basin_index_npz: Path,
    pixel_event_index_npz: Path,
    out_dir: Path,
    strong_threshold: float = STRONG_RAIN_MM_H,
    k_matches: int = 5,
    accumulation_quantile: float = 0.00,
    severe_delta_threshold: float = 10.0,
    warning_delta_threshold: float = 2.0,
    max_pixels_per_basin_output: int | None = None,
    workers: int = 1,
) -> dict[str, Path]:

    t_total = perf_counter()

    state = state.upper()
    out_dir = Path(out_dir) / state
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(workers))

    print("=" * 100)
    print("COMPUTE CURRENT ALERTS FOR STATE")
    print("=" * 100)
    print(f"state                 : {state}")
    print(f"current_rain_npz      : {current_rain_npz}")
    print(f"state_basin_index_npz : {state_basin_index_npz}")
    print(f"pixel_event_index_npz : {pixel_event_index_npz}")
    print(f"out_dir               : {out_dir}")
    print(f"strong_threshold      : {strong_threshold}")
    print(f"k_matches             : {k_matches}")
    print(f"accumulation_quantile : {accumulation_quantile}")
    print(f"workers               : {workers}")
    print("=" * 100)

    t_load = perf_counter()

    rain_data = load_current_state_rain(current_rain_npz)
    basin_idx = load_state_basin_index(state_basin_index_npz)
    hist = load_npz(pixel_event_index_npz)

    print(f"[TIMING] load inputs: {perf_counter() - t_load:.2f} seconds")
    print(f"[CHECK] rain shape: {rain_data['rain'].shape}")
    print(f"[CHECK] n_basins: {basin_idx['n_basins']:,}")
    print(f"[CHECK] n_state_pixels: {basin_idx['n_state_pixels']:,}")
    print(f"[CHECK] historical pixels: {len(hist['pixel_id_state']):,}")

    rain = rain_data["rain"]

    if rain.shape[1] != basin_idx["n_state_pixels"]:
        raise ValueError(
            f"Rain n_state_pixels mismatch. rain={rain.shape[1]}, "
            f"index={basin_idx['n_state_pixels']}"
        )

    t_preprocess = perf_counter()

    rain = np.where(np.isfinite(rain) & (rain > 0), rain, 0.0).astype(np.float32, copy=False)

    current_pixel_value_state = rain.max(axis=0).astype(np.float32)
    current_pixel_accum_state = rain.sum(axis=0).astype(np.float32)

    print(f"[TIMING] preprocess current rain: {perf_counter() - t_preprocess:.2f} seconds")
    print(f"[CHECK] current max hourly pixel rain: {float(current_pixel_value_state.max()):.3f}")
    print(f"[CHECK] current max accumulated pixel rain: {float(current_pixel_accum_state.max()):.3f}")
    print(f"[CHECK] total accumulated state rain: {float(current_pixel_accum_state.sum()):.3f}")

    t_lookup = perf_counter()
    hist_lookup = build_event_lookup(hist)
    print(f"[TIMING] build historical lookup: {perf_counter() - t_lookup:.2f} seconds")
    print(f"[CHECK] historical lookup entries: {len(hist_lookup):,}")

    basin_rows = []
    pixel_rows = []

    site_ids = basin_idx["site_ids"].astype(str)
    tasks = [(basin_i, str(site_id)) for basin_i, site_id in enumerate(site_ids)]

    skipped_normal_basins = 0

    t_loop = perf_counter()

    if workers <= 1:
        print("[MODE] sequential basin processing")

        for basin_i, site_id in tasks:
            basin_row, basin_pixel_records, skipped_normal = _process_one_basin(
                basin_i=basin_i,
                site_id=site_id,
                state=state,
                basin_idx=basin_idx,
                hist=hist,
                hist_lookup=hist_lookup,
                current_pixel_value_state=current_pixel_value_state,
                current_pixel_accum_state=current_pixel_accum_state,
                strong_threshold=strong_threshold,
                k_matches=k_matches,
                accumulation_quantile=accumulation_quantile,
                severe_delta_threshold=severe_delta_threshold,
                warning_delta_threshold=warning_delta_threshold,
                max_pixels_per_basin_output=max_pixels_per_basin_output,
            )

            if basin_row is not None:
                basin_rows.append(basin_row)

            if basin_pixel_records:
                pixel_rows.extend(basin_pixel_records)

            if skipped_normal:
                skipped_normal_basins += 1

            if (basin_i + 1) % 25 == 0:
                elapsed_loop = perf_counter() - t_loop
                print(
                    f"[{basin_i + 1:5d}/{len(site_ids)}] "
                    f"basins processed | pixel_alert_rows={len(pixel_rows):,} "
                    f"| fast_normal={skipped_normal_basins:,} "
                    f"| elapsed_loop={elapsed_loop:.2f}s"
                )

    else:
        print(f"[MODE] parallel basin processing with {workers} workers")

        results_by_basin: list[tuple[int, dict | None, list[dict], bool]] = []
        completed = 0

        with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            state,
            basin_idx,
            hist,
            hist_lookup,
            current_pixel_value_state,
            current_pixel_accum_state,
            strong_threshold,
            k_matches,
            accumulation_quantile,
            severe_delta_threshold,
            warning_delta_threshold,
            max_pixels_per_basin_output,
        ),
        ) as executor:
            futures = [
                executor.submit(_process_one_basin_from_worker_state, task)
                for task in tasks
            ]

            for future in as_completed(futures):
                result = future.result()
                results_by_basin.append(result)

                completed += 1

                basin_i, _, basin_pixel_records, skipped_normal = result

                if basin_pixel_records:
                    current_pixel_rows = sum(len(r[2]) for r in results_by_basin)
                else:
                    current_pixel_rows = sum(len(r[2]) for r in results_by_basin)

                current_fast_normal = sum(1 for r in results_by_basin if r[3])

                if completed % 25 == 0 or completed == len(tasks):
                    elapsed_loop = perf_counter() - t_loop
                    print(
                        f"[{completed:5d}/{len(site_ids)}] "
                        f"basins completed | pixel_alert_rows={current_pixel_rows:,} "
                        f"| fast_normal={current_fast_normal:,} "
                        f"| elapsed_loop={elapsed_loop:.2f}s"
                    )

        results_by_basin.sort(key=lambda x: x[0])

        for _, basin_row, basin_pixel_records, skipped_normal in results_by_basin:
            if basin_row is not None:
                basin_rows.append(basin_row)

            if basin_pixel_records:
                pixel_rows.extend(basin_pixel_records)

            if skipped_normal:
                skipped_normal_basins += 1

    print(f"[TIMING] basin loop: {perf_counter() - t_loop:.2f} seconds")
    print(f"[CHECK] basin rows: {len(basin_rows):,}")
    print(f"[CHECK] pixel rows: {len(pixel_rows):,}")
    print(f"[CHECK] fast normal basins: {skipped_normal_basins:,}")

    t_dataframe = perf_counter()

    basin_df = pd.DataFrame(basin_rows)
    pixel_df = pd.DataFrame(pixel_rows)

    basin_order = {
        "NORMAL": 0,
        "WATCH": 1,
        "WARNING": 2,
        "SEVERE": 3,
    }

    if not basin_df.empty:
        basin_df["alert_rank"] = basin_df["alert_level"].map(basin_order).astype(int)
        basin_df = basin_df.sort_values(
            ["alert_rank", "estimated_delta_water_stage", "current_max_pixel_value"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    if not pixel_df.empty:
        pixel_df = pixel_df.sort_values(
            ["estimated_delta_water_stage", "current_pixel_value"],
            ascending=[False, False],
        ).reset_index(drop=True)

    print(f"[TIMING] build dataframes and sort: {perf_counter() - t_dataframe:.2f} seconds")
    print(f"[CHECK] basin_df shape: {basin_df.shape}")
    print(f"[CHECK] pixel_df shape: {pixel_df.shape}")

    basin_parquet = out_dir / "basin_alerts.parquet"
    basin_csv = out_dir / "basin_alerts.csv"
    pixel_parquet = out_dir / "pixel_alerts.parquet"
    pixel_csv = out_dir / "pixel_alerts.csv"

    t_write = perf_counter()

    basin_df.to_parquet(basin_parquet, index=False)
    basin_df.to_csv(basin_csv, index=False)

    if not pixel_df.empty:
        pixel_df.to_parquet(pixel_parquet, index=False)
        pixel_df.to_csv(pixel_csv, index=False)
    else:
        pd.DataFrame().to_parquet(pixel_parquet, index=False)
        pd.DataFrame().to_csv(pixel_csv, index=False)

    print(f"[TIMING] write outputs: {perf_counter() - t_write:.2f} seconds")

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"basin alerts : {basin_parquet}")
    print(f"pixel alerts : {pixel_parquet}")
    print(f"fast normal basins skipped: {skipped_normal_basins:,}")
    print("\nALERT COUNTS")

    if not basin_df.empty:
        print(basin_df["alert_level"].value_counts(dropna=False))

    print(f"pixel alert rows: {len(pixel_df):,}")
    print(f"[TIMING] total runtime: {perf_counter() - t_total:.2f} seconds")

    return {
        "basin_alerts_parquet": basin_parquet,
        "basin_alerts_csv": basin_csv,
        "pixel_alerts_parquet": pixel_parquet,
        "pixel_alerts_csv": pixel_csv,
    }