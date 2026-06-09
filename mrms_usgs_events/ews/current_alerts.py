from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import os

import numpy as np
import pandas as pd


STRONG_RAIN_MM_H = 3.0

# -----------------------------------------------------------------------------
# Debug controls
# -----------------------------------------------------------------------------
# CURRENT_ALERTS_DEBUG=1/0
# CURRENT_ALERTS_DEBUG_SITES=08165500,08165300
# CURRENT_ALERTS_DEBUG_ONLY_CHANGED=1/0
# CURRENT_ALERTS_DEBUG_MIN_LEVEL=NORMAL/WATCH/WARNING/SEVERE
# -----------------------------------------------------------------------------
DEBUG_EFFICIENT_ALERTS = os.environ.get("CURRENT_ALERTS_DEBUG", "0").strip().lower() not in {"0", "false", "no"}
DEBUG_SITES_RAW = os.environ.get("CURRENT_ALERTS_DEBUG_SITES", "").strip()
DEBUG_SITES = {s.strip() for s in DEBUG_SITES_RAW.split(",") if s.strip()}
DEBUG_ONLY_CHANGED = os.environ.get("CURRENT_ALERTS_DEBUG_ONLY_CHANGED", "0").strip().lower() in {"1", "true", "yes"}
DEBUG_MIN_LEVEL = os.environ.get("CURRENT_ALERTS_DEBUG_MIN_LEVEL", "NORMAL").strip().upper()

# Optional pixel-alert row limit for responsive map popups.
# Default is None: keep all pixel rows unless the caller passes a limit.
# Use CURRENT_ALERTS_MAX_PIXELS_PER_BASIN=N to set a default from the environment.
_DEFAULT_MAX_PIXELS_RAW = os.environ.get("CURRENT_ALERTS_MAX_PIXELS_PER_BASIN", "").strip()
if _DEFAULT_MAX_PIXELS_RAW:
    try:
        DEFAULT_MAX_PIXELS_PER_BASIN_OUTPUT = int(_DEFAULT_MAX_PIXELS_RAW)
    except ValueError:
        DEFAULT_MAX_PIXELS_PER_BASIN_OUTPUT = None
    if DEFAULT_MAX_PIXELS_PER_BASIN_OUTPUT is not None and DEFAULT_MAX_PIXELS_PER_BASIN_OUTPUT <= 0:
        DEFAULT_MAX_PIXELS_PER_BASIN_OUTPUT = None
else:
    DEFAULT_MAX_PIXELS_PER_BASIN_OUTPUT = None

ALERT_RANK = {
    "NORMAL": 0,
    "WATCH": 1,
    "WARNING": 2,
    "SEVERE": 3,
}

_WORKER_STATE: dict = {}


# =============================================================================
# Loaders and helpers
# =============================================================================
def _site_key(site_id: str) -> str:
    return str(site_id).strip()


def should_debug_site(site_id: str, alert_level: str = "NORMAL", legacy_alert_level: str = "NORMAL") -> bool:
    if not DEBUG_EFFICIENT_ALERTS:
        return False

    site_key = _site_key(site_id)

    if DEBUG_SITES and site_key not in DEBUG_SITES:
        return False

    if DEBUG_ONLY_CHANGED and alert_level == legacy_alert_level:
        return False

    min_rank = ALERT_RANK.get(DEBUG_MIN_LEVEL, 0)
    if max(ALERT_RANK.get(alert_level, 0), ALERT_RANK.get(legacy_alert_level, 0)) < min_rank:
        return False

    return True


def load_npz(fp: Path) -> dict:
    fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"NPZ file not found: {fp}")

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


def validate_efficient_reference(efficient_ref: dict, basin_idx: dict, state: str) -> None:
    required = [
        "state",
        "site_ids",
        "min_events",
        "ref_ptr",
        "pixel_ref",
        "basin_ref",
        "delta_ref",
        "n_events_all",
        "n_events_rain_response",
        "n_events_good",
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
        "fastest_event_id",
        "fastest_response_hr",
        "fastest_delta_water_stage",
        "fastest_basin_accumulation",
        "fastest_pixel_accumulation",
        "max_delta_event_id",
        "max_delta_response_hr",
        "max_delta_water_stage",
        "max_delta_basin_accumulation",
        "max_delta_pixel_accumulation",
    ]

    missing = [k for k in required if k not in efficient_ref]
    if missing:
        raise ValueError(
            "efficient_event_reference_npz is missing required arrays: "
            + ", ".join(missing)
        )

    ref_state = str(efficient_ref["state"])
    if ref_state.upper() != state.upper():
        raise ValueError(f"Efficient reference state mismatch. expected={state}, found={ref_state}")

    ref_site_ids = efficient_ref["site_ids"].astype(str)
    basin_site_ids = basin_idx["site_ids"].astype(str)

    if len(ref_site_ids) != len(basin_site_ids):
        raise ValueError(
            f"Efficient reference site count mismatch. "
            f"reference={len(ref_site_ids)}, basin_index={len(basin_site_ids)}"
        )

    if not np.array_equal(ref_site_ids, basin_site_ids):
        raise ValueError(
            "Efficient reference site_ids do not match state_basin_index site_ids. "
            "Rebuild the efficient reference with the same state_basin_index."
        )

    ref_ptr = efficient_ref["ref_ptr"].astype(np.int64)
    if len(ref_ptr) != len(ref_site_ids) + 1:
        raise ValueError(
            f"Invalid ref_ptr length. expected={len(ref_site_ids) + 1}, found={len(ref_ptr)}"
        )

    if int(ref_ptr[-1]) != len(efficient_ref["pixel_ref"]):
        raise ValueError(
            f"Invalid efficient reference pointer. ref_ptr[-1]={int(ref_ptr[-1])}, "
            f"pixel_ref length={len(efficient_ref['pixel_ref'])}"
        )

    if len(efficient_ref["pixel_ref"]) != len(efficient_ref["basin_ref"]):
        raise ValueError("Efficient reference pixel_ref and basin_ref lengths do not match")

    if len(efficient_ref["pixel_ref"]) != len(efficient_ref["delta_ref"]):
        raise ValueError("Efficient reference pixel_ref and delta_ref lengths do not match")


def build_event_lookup(pixel_event_index: dict) -> dict[int, int]:
    pixel_id_state = pixel_event_index["pixel_id_state"].astype(np.int32)
    return {int(pid): i for i, pid in enumerate(pixel_id_state)}


def percentile_rank(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    reference = reference[np.isfinite(reference)]

    if reference.size == 0 or not np.isfinite(value):
        return np.nan

    return float(100.0 * np.mean(reference <= value))


def classify_percentile_alert(percentile_value: float) -> str:
    if not np.isfinite(percentile_value):
        return "NORMAL"
    if percentile_value >= 90.0:
        return "SEVERE"
    if percentile_value >= 75.0:
        return "WARNING"
    if percentile_value >= 50.0:
        return "WATCH"
    return "NORMAL"


def classify_efficient_event_alert(*, pixel_pct: float, basin_pct: float, weighted_pct: float) -> tuple[str, str]:
    """
    Main classifier:

    if pixel_pct >= 90: SEVERE
    elif basin_pct >= 90: SEVERE
    elif weighted_pct >= 75: WARNING
    elif weighted_pct >= 50: WATCH
    else: NORMAL
    """
    if np.isfinite(pixel_pct) and pixel_pct >= 90.0:
        return "SEVERE", "PIXEL_P90"

    if np.isfinite(basin_pct) and basin_pct >= 90.0:
        return "SEVERE", "BASIN_P90"

    if np.isfinite(weighted_pct) and weighted_pct >= 75.0:
        return "WARNING", "WEIGHTED_P75"

    if np.isfinite(weighted_pct) and weighted_pct >= 50.0:
        return "WATCH", "WEIGHTED_P50"

    return "NORMAL", "BELOW_P50"


# =============================================================================
# Efficient reference from cached state_efficient_event_reference/STATE_*.npz
# =============================================================================
def _empty_cached_efficient_alerts() -> dict:
    return {
        "efficient_history_ok": False,
        "efficient_n_events_all": 0,
        "efficient_n_events_rain_response": 0,
        "efficient_n_events_good": 0,
        "efficient_corr_pixel_delta": np.nan,
        "efficient_corr_basin_delta": np.nan,
        "efficient_corr_combined_delta": np.nan,
        "efficient_pixel_weight": np.nan,
        "efficient_basin_weight": np.nan,
        "efficient_pixel_ref_p50": np.nan,
        "efficient_pixel_ref_p75": np.nan,
        "efficient_pixel_ref_p90": np.nan,
        "efficient_basin_ref_p50": np.nan,
        "efficient_basin_ref_p75": np.nan,
        "efficient_basin_ref_p90": np.nan,
        "efficient_delta_ref_p50": np.nan,
        "efficient_delta_ref_p75": np.nan,
        "efficient_delta_ref_p90": np.nan,
        "efficient_p50_eff_pixel": np.nan,
        "efficient_p50_eff_basin": np.nan,
        "efficient_pixel_percentile": np.nan,
        "efficient_basin_percentile": np.nan,
        "efficient_max_percentile": np.nan,
        "efficient_avg_percentile": np.nan,
        "efficient_weighted_percentile": np.nan,
        "efficient_pixel_level": "NORMAL",
        "efficient_basin_level": "NORMAL",
        "efficient_max_level": "NORMAL",
        "efficient_avg_level": "NORMAL",
        "efficient_weighted_level": "NORMAL",
        "historical_fastest_event_id": np.nan,
        "historical_fastest_response_hr": np.nan,
        "historical_fastest_delta_water_stage": np.nan,
        "historical_fastest_basin_accumulation": np.nan,
        "historical_fastest_pixel_accumulation": np.nan,
        "historical_max_delta_event_id": np.nan,
        "historical_max_delta_response_hr": np.nan,
        "historical_max_delta_water_stage": np.nan,
        "historical_max_delta_basin_accumulation": np.nan,
        "historical_max_delta_pixel_accumulation": np.nan,
        "basin_peak_time": np.nan,
        "pixel_peak_time": np.nan,
        "max_delta_basin_peak_time": np.nan,
        "max_delta_pixel_peak_time": np.nan,
    }


def efficient_percentile_alerts_from_cached_ref(
    *,
    site_index: int,
    current_basin_accumulation: float,
    current_max_pixel_accumulation: float,
    efficient_ref: dict,
) -> dict:
    """
    Uses the precomputed state-level efficient-event reference NPZ.

    This replaces the old per-basin runtime construction of the efficient
    historical reference. No historical efficient reference is recalculated here.
    """
    out = _empty_cached_efficient_alerts()

    site_index = int(site_index)
    ref_ptr = efficient_ref["ref_ptr"].astype(np.int64)

    if site_index < 0 or site_index >= len(ref_ptr) - 1:
        return out

    h0 = int(ref_ptr[site_index])
    h1 = int(ref_ptr[site_index + 1])

    n_events_all = int(efficient_ref["n_events_all"][site_index])
    n_events_rain_response = int(efficient_ref["n_events_rain_response"][site_index])
    n_events_good = int(efficient_ref["n_events_good"][site_index])

    pixel_ref = efficient_ref["pixel_ref"][h0:h1].astype(np.float32)
    basin_ref = efficient_ref["basin_ref"][h0:h1].astype(np.float32)

    history_ok = bool(n_events_good > 0 and pixel_ref.size > 0 and basin_ref.size > 0)

    pixel_pct = np.nan
    basin_pct = np.nan
    max_pct = np.nan
    avg_pct = np.nan
    weighted_pct = np.nan

    pixel_weight = float(efficient_ref["pixel_weight"][site_index])
    basin_weight = float(efficient_ref["basin_weight"][site_index])

    if history_ok:
        pixel_pct = percentile_rank(current_max_pixel_accumulation, pixel_ref)
        basin_pct = percentile_rank(current_basin_accumulation, basin_ref)

        max_pct = float(np.nanmax([pixel_pct, basin_pct]))
        avg_pct = float(np.nanmean([pixel_pct, basin_pct]))

        if not np.isfinite(pixel_weight) or not np.isfinite(basin_weight) or (pixel_weight + basin_weight) <= 0:
            pixel_weight = 0.5
            basin_weight = 0.5

        weighted_pct = float(pixel_weight * pixel_pct + basin_weight * basin_pct)

    out.update(
        {
            "efficient_history_ok": history_ok,
            "efficient_n_events_all": n_events_all,
            "efficient_n_events_rain_response": n_events_rain_response,
            "efficient_n_events_good": n_events_good,
            "efficient_corr_pixel_delta": float(efficient_ref["corr_pixel_delta"][site_index]),
            "efficient_corr_basin_delta": float(efficient_ref["corr_basin_delta"][site_index]),
            "efficient_corr_combined_delta": float(efficient_ref["corr_combined_delta"][site_index]),
            "efficient_pixel_weight": float(pixel_weight),
            "efficient_basin_weight": float(basin_weight),
            "efficient_pixel_ref_p50": float(efficient_ref["pixel_p50"][site_index]),
            "efficient_pixel_ref_p75": float(efficient_ref["pixel_p75"][site_index]),
            "efficient_pixel_ref_p90": float(efficient_ref["pixel_p90"][site_index]),
            "efficient_basin_ref_p50": float(efficient_ref["basin_p50"][site_index]),
            "efficient_basin_ref_p75": float(efficient_ref["basin_p75"][site_index]),
            "efficient_basin_ref_p90": float(efficient_ref["basin_p90"][site_index]),
            "efficient_delta_ref_p50": float(efficient_ref["delta_p50"][site_index]),
            "efficient_delta_ref_p75": float(efficient_ref["delta_p75"][site_index]),
            "efficient_delta_ref_p90": float(efficient_ref["delta_p90"][site_index]),
            "efficient_p50_eff_pixel": float(efficient_ref["p50_eff_pixel"][site_index]),
            "efficient_p50_eff_basin": float(efficient_ref["p50_eff_basin"][site_index]),
            "efficient_pixel_percentile": pixel_pct,
            "efficient_basin_percentile": basin_pct,
            "efficient_max_percentile": max_pct,
            "efficient_avg_percentile": avg_pct,
            "efficient_weighted_percentile": weighted_pct,
            "efficient_pixel_level": classify_percentile_alert(pixel_pct),
            "efficient_basin_level": classify_percentile_alert(basin_pct),
            "efficient_max_level": classify_percentile_alert(max_pct),
            "efficient_avg_level": classify_percentile_alert(avg_pct),
            "efficient_weighted_level": classify_percentile_alert(weighted_pct),
            "historical_fastest_event_id": int(efficient_ref["fastest_event_id"][site_index]),
            "historical_fastest_response_hr": float(efficient_ref["fastest_response_hr"][site_index]),
            "historical_fastest_delta_water_stage": float(efficient_ref["fastest_delta_water_stage"][site_index]),
            "historical_fastest_basin_accumulation": float(efficient_ref["fastest_basin_accumulation"][site_index]),
            "historical_fastest_pixel_accumulation": float(efficient_ref["fastest_pixel_accumulation"][site_index]),
            "historical_max_delta_event_id": int(efficient_ref["max_delta_event_id"][site_index]),
            "historical_max_delta_response_hr": float(efficient_ref["max_delta_response_hr"][site_index]),
            "historical_max_delta_water_stage": float(efficient_ref["max_delta_water_stage"][site_index]),
            "historical_max_delta_basin_accumulation": float(efficient_ref["max_delta_basin_accumulation"][site_index]),
            "historical_max_delta_pixel_accumulation": float(efficient_ref["max_delta_pixel_accumulation"][site_index]),
            "basin_peak_time": float(efficient_ref["basin_peak_time"][site_index]),
            "pixel_peak_time": float(efficient_ref["pixel_peak_time"][site_index]),
            "max_delta_basin_peak_time": float(efficient_ref["max_delta_basin_peak_time"][site_index]),
            "max_delta_pixel_peak_time": float(efficient_ref["max_delta_pixel_peak_time"][site_index]),
        }
    )

    return out


def empty_efficient_alerts() -> dict:
    return _empty_cached_efficient_alerts()


# =============================================================================
# Basin and pixel processing
# =============================================================================
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
) -> dict:
    return {
        "state": state,
        "site_id": site_id,
        "alert_level": "NORMAL",
        "legacy_alert_level": "NOT_COMPUTED",
        "efficient_decision_reason": "NO_STRONG_CURRENT_PIXEL",
        "current_basin_accumulation": current_basin_accumulation,
        "current_max_pixel_value": current_max_pixel_value,
        "current_max_pixel_accumulation": current_max_pixel_accumulation,
        "n_basin_pixels": int(n_basin_pixels),
        "n_active_pixels": 0,
        "n_active_pixels_with_history": 0,
        "historical_basin_accumulation_threshold": np.nan,
        "basin_accumulation_reaches_history": False,
        "strong_threshold": strong_threshold,
        "accumulation_quantile": accumulation_quantile,
        **empty_efficient_alerts(),
    }


def _collect_active_history_indices(
    *,
    basin_i: int,
    basin_pixels: np.ndarray,
    active_local: np.ndarray,
    hist: dict,
    hist_lookup: dict[int, int],
) -> tuple[list[tuple[int, np.ndarray]], np.ndarray, np.ndarray]:
    """
    This still uses state_pixel_event_index/STATE_pixel_event_index.npz.

    Purpose here:
    - keep pixel popup support
    - keep historical pixel details for selected active pixels
    - keep existing basin_accumulation_reaches_history diagnostic

    It is NOT used to classify the basin alert level anymore.
    """
    active_pixels_with_history = []
    historical_basin_acc_values = []

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
        active_pixels_with_history.append((int(local_j), idx0))

    if active_pixels_with_history:
        all_hist_idx = np.concatenate([idx0 for _, idx0 in active_pixels_with_history]).astype(np.int64)
    else:
        all_hist_idx = np.array([], dtype=np.int64)

    if historical_basin_acc_values:
        historical_basin_acc_all = np.concatenate(historical_basin_acc_values).astype(np.float32)
    else:
        historical_basin_acc_all = np.array([], dtype=np.float32)

    return active_pixels_with_history, historical_basin_acc_all, all_hist_idx


def _make_pixel_records(
    *,
    state: str,
    site_id: str,
    basin_i: int,
    basin_pixels: np.ndarray,
    active_pixels_with_history: list[tuple[int, np.ndarray]],
    basin_idx: dict,
    hist: dict,
    cur_pixel_value: np.ndarray,
    cur_pixel_accum: np.ndarray,
    current_basin_accumulation: float,
    historical_basin_acc_threshold: float,
    efficient_alerts: dict,
    max_pixels_per_basin_output: int | None,
) -> list[dict]:
    rows = []

    # Speed optimization: the popup/map does not need every active pixel.
    # Keep the strongest current pixels first, then read historical details only for those.
    selected_active_pixels = active_pixels_with_history
    if max_pixels_per_basin_output is not None and len(selected_active_pixels) > max_pixels_per_basin_output:
        selected_active_pixels = sorted(
            selected_active_pixels,
            key=lambda item: (
                float(cur_pixel_accum[item[0]]),
                float(cur_pixel_value[item[0]]),
            ),
            reverse=True,
        )[:max_pixels_per_basin_output]

    for local_j, idx0 in selected_active_pixels:
        pixel_id_state = int(basin_pixels[local_j])

        hist_delta = hist["delta_water_stage"][idx0].astype(np.float32)
        hist_event = hist["event_id"][idx0].astype(np.int64)
        hist_basin = hist["basin_accumulation"][idx0].astype(np.float32)
        hist_pix_acc = hist["pixel_accumulation"][idx0].astype(np.float32)

        if hist_delta.size:
            best_pos = int(np.nanargmax(hist_delta))
            best_event_id = int(hist_event[best_pos])
            best_delta = float(hist_delta[best_pos])
            best_hist_basin = float(hist_basin[best_pos])
            best_hist_pixel = float(hist_pix_acc[best_pos])
        else:
            best_event_id = np.nan
            best_delta = np.nan
            best_hist_basin = np.nan
            best_hist_pixel = np.nan

        rows.append(
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
                "historical_pixel_best_event_id": best_event_id,
                "historical_pixel_best_delta_water_stage": best_delta,
                "historical_pixel_best_basin_accumulation": best_hist_basin,
                "historical_pixel_best_pixel_accumulation": best_hist_pixel,
                "efficient_pixel_percentile": efficient_alerts["efficient_pixel_percentile"],
                "efficient_basin_percentile": efficient_alerts["efficient_basin_percentile"],
                "efficient_weighted_percentile": efficient_alerts["efficient_weighted_percentile"],
            }
        )

    if rows and max_pixels_per_basin_output is not None:
        rows = sorted(
            rows,
            key=lambda r: (
                r["current_pixel_accumulation"],
                r["current_pixel_value"],
            ),
            reverse=True,
        )[:max_pixels_per_basin_output]

    return rows


def _process_one_basin(
    *,
    basin_i: int,
    site_id: str,
    state: str,
    basin_idx: dict,
    hist: dict,
    hist_lookup: dict[int, int],
    efficient_ref: dict,
    current_pixel_value_state: np.ndarray,
    current_pixel_accum_state: np.ndarray,
    strong_threshold: float,
    k_matches: int,  # kept for API compatibility; not used in the percentile classifier
    accumulation_quantile: float,
    severe_delta_threshold: float,  # kept for API compatibility; not used in the percentile classifier
    warning_delta_threshold: float,  # kept for API compatibility; not used in the percentile classifier
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
        )
        return basin_row, [], True

    # Classification uses ONLY state_efficient_event_reference/STATE_efficient_event_reference.npz.
    # It does not rebuild the historical efficient reference from pixel_event_index.
    efficient_alerts = efficient_percentile_alerts_from_cached_ref(
        site_index=basin_i,
        current_basin_accumulation=current_basin_accumulation,
        current_max_pixel_accumulation=current_max_pixel_accumulation,
        efficient_ref=efficient_ref,
    )

    pixel_pct = efficient_alerts["efficient_pixel_percentile"]
    basin_pct = efficient_alerts["efficient_basin_percentile"]
    weighted_pct = efficient_alerts["efficient_weighted_percentile"]

    if current_max_pixel_value < strong_threshold or n_active_pixels == 0:
        alert_level = "NORMAL"
        efficient_decision_reason = "NO_STRONG_CURRENT_PIXEL"
    elif efficient_alerts["efficient_history_ok"]:
        alert_level, efficient_decision_reason = classify_efficient_event_alert(
            pixel_pct=pixel_pct,
            basin_pct=basin_pct,
            weighted_pct=weighted_pct,
        )
    else:
        # No old fallback calculation is used here.
        # If there is strong current rain but no cached efficient history for this basin,
        # keep a conservative WATCH and report the reason explicitly.
        alert_level = "WATCH"
        efficient_decision_reason = "FALLBACK_WATCH_NO_CACHED_EFFICIENT_HISTORY"

    legacy_alert_level = "NOT_COMPUTED"

    # IMPORTANT PERFORMANCE FIX:
    # Do not scan state_pixel_event_index for every basin before classification.
    # The cached efficient reference above is enough to classify basin alerts.
    # Pixel history is read only for SEVERE basins. WATCH/WARNING/NORMAL still
    # appear in basin_alerts, but do not generate pixel_alert rows.
    active_pixels_with_history = []
    historical_basin_accumulation_threshold = np.nan
    basin_accumulation_reaches_history = False
    pixel_records = []

    if alert_level == "SEVERE":
        active_pixels_with_history, historical_basin_acc_all, _all_hist_idx_unused = _collect_active_history_indices(
            basin_i=basin_i,
            basin_pixels=basin_pixels,
            active_local=active_local,
            hist=hist,
            hist_lookup=hist_lookup,
        )

        if historical_basin_acc_all.size:
            historical_basin_accumulation_threshold = float(
                np.nanquantile(historical_basin_acc_all, accumulation_quantile)
            )
            basin_accumulation_reaches_history = (
                current_basin_accumulation >= historical_basin_accumulation_threshold
            )

        pixel_records = _make_pixel_records(
            state=state,
            site_id=site_id,
            basin_i=basin_i,
            basin_pixels=basin_pixels,
            active_pixels_with_history=active_pixels_with_history,
            basin_idx=basin_idx,
            hist=hist,
            cur_pixel_value=cur_pixel_value,
            cur_pixel_accum=cur_pixel_accum,
            current_basin_accumulation=current_basin_accumulation,
            historical_basin_acc_threshold=historical_basin_accumulation_threshold,
            efficient_alerts=efficient_alerts,
            max_pixels_per_basin_output=max_pixels_per_basin_output,
        )

    if should_debug_site(site_id, alert_level, legacy_alert_level):
        print(
            f"[ALERT] "
            f"site={site_id} "
            f"level={alert_level} "
            f"legacy={legacy_alert_level} "
            f"reason={efficient_decision_reason} "
            f"current_basin={current_basin_accumulation:.3f} "
            f"current_max_pix_acc={current_max_pixel_accumulation:.3f} "
            f"current_max_pix_hr={current_max_pixel_value:.3f} "
            f"active={n_active_pixels}/{len(basin_pixels)} "
            f"hist_ok={efficient_alerts['efficient_history_ok']} "
            f"n_hist_all={efficient_alerts['efficient_n_events_all']} "
            f"n_rr={efficient_alerts['efficient_n_events_rain_response']} "
            f"n_good={efficient_alerts['efficient_n_events_good']} "
            f"pixel_pct={pixel_pct:.2f} "
            f"basin_pct={basin_pct:.2f} "
            f"weighted_pct={weighted_pct:.2f} "
            f"max_pct={efficient_alerts['efficient_max_percentile']:.2f} "
            f"avg_pct={efficient_alerts['efficient_avg_percentile']:.2f} "
            f"w_pixel={efficient_alerts['efficient_pixel_weight']:.2f} "
            f"w_basin={efficient_alerts['efficient_basin_weight']:.2f} "
            f"corr_pixel={efficient_alerts['efficient_corr_pixel_delta']:.3f} "
            f"corr_basin={efficient_alerts['efficient_corr_basin_delta']:.3f}",
            flush=True,
        )

        print(
            f"[EFFICIENT HIST CACHED] "
            f"site={site_id} "
            f"pixel_ref_p50/p75/p90="
            f"{efficient_alerts['efficient_pixel_ref_p50']:.2f}/"
            f"{efficient_alerts['efficient_pixel_ref_p75']:.2f}/"
            f"{efficient_alerts['efficient_pixel_ref_p90']:.2f} "
            f"basin_ref_p50/p75/p90="
            f"{efficient_alerts['efficient_basin_ref_p50']:.2f}/"
            f"{efficient_alerts['efficient_basin_ref_p75']:.2f}/"
            f"{efficient_alerts['efficient_basin_ref_p90']:.2f} "
            f"delta_ref_p50/p75/p90="
            f"{efficient_alerts['efficient_delta_ref_p50']:.2f}/"
            f"{efficient_alerts['efficient_delta_ref_p75']:.2f}/"
            f"{efficient_alerts['efficient_delta_ref_p90']:.2f} "
            f"eff_p50_pixel={efficient_alerts['efficient_p50_eff_pixel']:.6f} "
            f"eff_p50_basin={efficient_alerts['efficient_p50_eff_basin']:.8f}",
            flush=True,
        )

        print(
            f"[HISTORICAL EVENT CACHED] "
            f"site={site_id} "
            f"fastest_event={efficient_alerts['historical_fastest_event_id']} "
            f"fastest_response_hr={efficient_alerts['historical_fastest_response_hr']:.3f} "
            f"fastest_delta={efficient_alerts['historical_fastest_delta_water_stage']:.2f} "
            f"fastest_basin={efficient_alerts['historical_fastest_basin_accumulation']:.2f} "
            f"fastest_pixel={efficient_alerts['historical_fastest_pixel_accumulation']:.2f} "
            f"max_delta_event={efficient_alerts['historical_max_delta_event_id']} "
            f"max_delta_response_hr={efficient_alerts['historical_max_delta_response_hr']:.3f} "
            f"basin_peak_time={efficient_alerts['basin_peak_time']:.3f} "
            f"pixel_peak_time={efficient_alerts['pixel_peak_time']:.3f} "
            f"max_delta={efficient_alerts['historical_max_delta_water_stage']:.2f}",
            flush=True,
        )

    basin_row = {
        "state": state,
        "site_id": site_id,
        "alert_level": alert_level,
        "legacy_alert_level": legacy_alert_level,
        "efficient_decision_reason": efficient_decision_reason,
        "current_basin_accumulation": current_basin_accumulation,
        "current_max_pixel_value": current_max_pixel_value,
        "current_max_pixel_accumulation": current_max_pixel_accumulation,
        "n_basin_pixels": int(len(basin_pixels)),
        "n_active_pixels": n_active_pixels,
        "n_active_pixels_with_history": int(len(active_pixels_with_history)),
        "historical_basin_accumulation_threshold": historical_basin_accumulation_threshold,
        "basin_accumulation_reaches_history": bool(basin_accumulation_reaches_history),
        "strong_threshold": strong_threshold,
        "accumulation_quantile": accumulation_quantile,
        **efficient_alerts,
    }

    return basin_row, pixel_records, False

def _init_worker(
    state: str,
    basin_idx: dict,
    hist: dict,
    hist_lookup: dict[int, int],
    efficient_ref: dict,
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
        "efficient_ref": efficient_ref,
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
        efficient_ref=_WORKER_STATE["efficient_ref"],
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


# =============================================================================
# Public API
# =============================================================================
def compute_current_alerts_for_state(
    *,
    state: str,
    current_rain_npz: Path,
    state_basin_index_npz: Path,
    pixel_event_index_npz: Path,
    efficient_event_reference_npz: Path,
    out_dir: Path,
    strong_threshold: float = STRONG_RAIN_MM_H,
    k_matches: int = 5,
    accumulation_quantile: float = 0.00,
    severe_delta_threshold: float = np.nan,
    warning_delta_threshold: float = np.nan,
    max_pixels_per_basin_output: int | None = None,
    workers: int = 1,
) -> dict[str, Path]:
    """
    Compute current alerts for one state.

    Inputs:
    - current_rain_npz: current rainfall array.
    - state_basin_index_npz: state basin/pixel geometry index.
    - pixel_event_index_npz: historical pixel-event index for popup pixel details.
    - efficient_event_reference_npz: cached basin-level efficient-event reference
      used for basin classification.

    Important:
    The basin classification uses ONLY efficient_event_reference_npz.
    The old runtime reconstruction of efficient historical references is not used.
    """
    t_total = perf_counter()

    state = state.upper()
    out_dir = Path(out_dir) / state
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(workers))
    if max_pixels_per_basin_output is None:
        max_pixels_per_basin_output = DEFAULT_MAX_PIXELS_PER_BASIN_OUTPUT

    if efficient_event_reference_npz is None:
        raise ValueError(
            "efficient_event_reference_npz is required. "
            "This version does not use the old runtime efficient-reference calculation."
        )

    print("=" * 100)
    print("COMPUTE CURRENT ALERTS FOR STATE")
    print("=" * 100)
    print(f"state                         : {state}")
    print(f"current_rain_npz              : {current_rain_npz}")
    print(f"state_basin_index_npz         : {state_basin_index_npz}")
    print(f"pixel_event_index_npz         : {pixel_event_index_npz}")
    print(f"efficient_event_reference_npz : {efficient_event_reference_npz}")
    print(f"out_dir                       : {out_dir}")
    print(f"strong_threshold              : {strong_threshold}")
    print(f"k_matches                     : {k_matches} (kept for compatibility; not used)")
    print(f"accumulation_quantile         : {accumulation_quantile}")
    print(f"workers                       : {workers}")
    print(f"max_pixels_per_basin_output   : {max_pixels_per_basin_output}")
    print("classifier                    : cached efficient ref; pixel>=P90 or basin>=P90 => SEVERE; weighted>=P75 => WARNING; weighted>=P50 => WATCH")
    print("=" * 100)

    t_load = perf_counter()

    rain_data = load_current_state_rain(current_rain_npz)
    basin_idx = load_state_basin_index(state_basin_index_npz)
    hist = load_npz(pixel_event_index_npz)
    efficient_ref = load_npz(efficient_event_reference_npz)
    validate_efficient_reference(efficient_ref, basin_idx, state)

    print(f"[TIMING] load inputs: {perf_counter() - t_load:.2f} seconds")
    print(f"[CHECK] rain shape: {rain_data['rain'].shape}")
    print(f"[CHECK] n_basins: {basin_idx['n_basins']:,}")
    print(f"[CHECK] n_state_pixels: {basin_idx['n_state_pixels']:,}")
    print(f"[CHECK] historical pixels for popup: {len(hist['pixel_id_state']):,}")
    print(f"[CHECK] efficient ref sites: {len(efficient_ref['site_ids']):,}")
    print(f"[CHECK] efficient ref events: {len(efficient_ref['pixel_ref']):,}")

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
    print(f"[TIMING] build historical pixel lookup: {perf_counter() - t_lookup:.2f} seconds")
    print(f"[CHECK] historical pixel lookup entries: {len(hist_lookup):,}")

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
                efficient_ref=efficient_ref,
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
        current_pixel_rows = 0
        current_fast_normal = 0

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                state,
                basin_idx,
                hist,
                hist_lookup,
                efficient_ref,
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
                current_pixel_rows += len(result[2])
                if result[3]:
                    current_fast_normal += 1

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

    if not basin_df.empty:
        basin_df["alert_rank"] = basin_df["alert_level"].map(ALERT_RANK).fillna(0).astype(int)
        basin_df = basin_df.sort_values(
            ["alert_rank", "efficient_weighted_percentile", "current_max_pixel_value"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    if not pixel_df.empty:
        pixel_df = pixel_df.sort_values(
            ["current_pixel_accumulation", "current_pixel_value"],
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

        if "efficient_decision_reason" in basin_df.columns:
            print("\nEFFICIENT DECISION REASONS")
            print(basin_df["efficient_decision_reason"].value_counts(dropna=False))

        cols_for_summary = [
            "efficient_weighted_percentile",
            "efficient_pixel_percentile",
            "efficient_basin_percentile",
            "current_basin_accumulation",
            "current_max_pixel_accumulation",
            "historical_fastest_response_hr",
            "historical_fastest_delta_water_stage",
        ]
        existing_summary_cols = [c for c in cols_for_summary if c in basin_df.columns]
        if existing_summary_cols:
            print("\nNUMERIC SUMMARY")
            print(basin_df[existing_summary_cols].describe().to_string())

    print(f"pixel alert rows: {len(pixel_df):,}")
    print(f"[TIMING] total runtime: {perf_counter() - t_total:.2f} seconds")

    return {
        "basin_alerts_parquet": basin_parquet,
        "basin_alerts_csv": basin_csv,
        "pixel_alerts_parquet": pixel_parquet,
        "pixel_alerts_csv": pixel_csv,
    }
