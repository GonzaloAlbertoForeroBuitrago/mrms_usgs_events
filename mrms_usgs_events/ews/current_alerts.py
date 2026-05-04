from __future__ import annotations

from pathlib import Path
import gc

import numpy as np
import pandas as pd


STRONG_RAIN_MM_H = 7.5


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
) -> dict[str, Path]:
    """
    Operational logic:

    1. Detect current pixels with current_pixel_value > strong_threshold.
    2. For each basin, compare current_basin_accumulation against historical
       basin_accumulation from events already filtered as:
          delta_water_stage >= p50
          pixel_value >= 7.5
       in the NPZ index.
    3. If current_basin_accumulation reaches the historical p50-response
       accumulation threshold, search similar historical events for active pixels.
    4. Interpolate:
          delta_water_stage
          time_to_rain_peak_accumulation_hr
          time_to_stage_peak_hr
       using similarity in:
          pixel_value + basin_accumulation
    5. Generate basin_alerts and pixel_alerts.
    """

    state = state.upper()
    out_dir = Path(out_dir) / state
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print("=" * 100)

    rain_data = load_current_state_rain(current_rain_npz)
    basin_idx = load_state_basin_index(state_basin_index_npz)
    hist = load_npz(pixel_event_index_npz)

    rain = rain_data["rain"]

    if rain.shape[1] != basin_idx["n_state_pixels"]:
        raise ValueError(
            f"Rain n_state_pixels mismatch. rain={rain.shape[1]}, "
            f"index={basin_idx['n_state_pixels']}"
        )

    rain = np.where(np.isfinite(rain) & (rain > 0), rain, 0.0).astype(np.float32, copy=False)

    current_pixel_value_state = rain.max(axis=0).astype(np.float32)
    current_pixel_accum_state = rain.sum(axis=0).astype(np.float32)

    hist_lookup = build_event_lookup(hist)

    basin_rows = []
    pixel_rows = []

    site_ids = basin_idx["site_ids"].astype(str)
    basin_ptr = basin_idx["basin_ptr"]
    basin_indices = basin_idx["basin_indices"]

    for basin_i, site_id in enumerate(site_ids):
        a = int(basin_ptr[basin_i])
        b = int(basin_ptr[basin_i + 1])
        basin_pixels = basin_indices[a:b]

        if len(basin_pixels) == 0:
            continue

        cur_pixel_value = current_pixel_value_state[basin_pixels]
        cur_pixel_accum = current_pixel_accum_state[basin_pixels]

        current_basin_accumulation = float(cur_pixel_accum.sum())
        current_max_pixel_value = float(cur_pixel_value.max())
        current_max_pixel_accumulation = float(cur_pixel_accum.max())

        active_local = np.flatnonzero(cur_pixel_value > strong_threshold)
        n_active_pixels = int(len(active_local))

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
                matched_time_rain_values.append(
                    interp["estimated_time_to_rain_peak_accumulation_hr"]
                )

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

        basin_rows.append(
            {
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
        )

        if basin_pixel_records:
            if max_pixels_per_basin_output is not None:
                basin_pixel_records = sorted(
                    basin_pixel_records,
                    key=lambda r: (
                        r["estimated_delta_water_stage"],
                        r["current_pixel_value"],
                    ),
                    reverse=True,
                )[:max_pixels_per_basin_output]

            pixel_rows.extend(basin_pixel_records)

        if (basin_i + 1) % 25 == 0:
            print(
                f"[{basin_i + 1:5d}/{len(site_ids)}] "
                f"basins processed | pixel_alert_rows={len(pixel_rows):,}"
            )
            gc.collect()

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

    basin_parquet = out_dir / "basin_alerts.parquet"
    basin_csv = out_dir / "basin_alerts.csv"
    pixel_parquet = out_dir / "pixel_alerts.parquet"
    pixel_csv = out_dir / "pixel_alerts.csv"

    basin_df.to_parquet(basin_parquet, index=False)
    basin_df.to_csv(basin_csv, index=False)

    if not pixel_df.empty:
        pixel_df.to_parquet(pixel_parquet, index=False)
        pixel_df.to_csv(pixel_csv, index=False)
    else:
        pd.DataFrame().to_parquet(pixel_parquet, index=False)
        pd.DataFrame().to_csv(pixel_csv, index=False)

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"basin alerts : {basin_parquet}")
    print(f"pixel alerts : {pixel_parquet}")
    print("\nALERT COUNTS")
    if not basin_df.empty:
        print(basin_df["alert_level"].value_counts(dropna=False))
    print(f"pixel alert rows: {len(pixel_df):,}")

    return {
        "basin_alerts_parquet": basin_parquet,
        "basin_alerts_csv": basin_csv,
        "pixel_alerts_parquet": pixel_parquet,
        "pixel_alerts_csv": pixel_csv,
    }