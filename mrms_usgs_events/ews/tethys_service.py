from __future__ import annotations

from pathlib import Path

from .realtime_engine import run_state_alert_engine
from .state_rain import build_current_state_rain_npz
from .tethys_outputs import export_state_alerts_for_tethys


DEFAULT_BASE_DIR = Path("/data/repository_code/unified_data")


def run_state_current_alert_for_tethys(
    *,
    state: str,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    hours_back: int = 12,
    workers: int = 4,
    public_dir: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    state = state.upper()
    base_dir = Path(base_dir)

    current_dir = base_dir / "current_rain"
    current_dir.mkdir(parents=True, exist_ok=True)

    recent_rain_npz = current_dir / f"{state}_current_rain_{hours_back}h.npz"

    state_mask = base_dir / "state_masks" / f"{state}_mrms_mask.npz"
    state_basin_index = base_dir / "state_basin_index" / f"{state}_state_basin_index.npz"
    predictor_dir = base_dir / "ews_predictors"
    historical_summary_dir = base_dir / "ews_history"
    out_dir = base_dir / "ews_operational" / state
    out_dir.mkdir(parents=True, exist_ok=True)

    build_current_state_rain_npz(
        state=state,
        state_mask_fp=state_mask,
        out_npz=recent_rain_npz,
        base_dir=base_dir,
        hours_back=hours_back,
        workers=workers,
        start=start,
        end=end,
    )

    paths = run_state_alert_engine(
        state=state,
        recent_rain_npz=recent_rain_npz,
        state_basin_index=state_basin_index,
        predictor_dir=predictor_dir,
        out_dir=out_dir,
        historical_summary_dir=historical_summary_dir,
    )

    alerts_parquet = Path(paths.get("basin_alerts_parquet", out_dir / "basin_alerts.parquet"))

    tethys_paths = export_state_alerts_for_tethys(
        state=state,
        base_dir=base_dir,
        alerts_parquet=alerts_parquet,
        out_dir=base_dir / "tethys_outputs" / state,
        public_dir=Path(public_dir) if public_dir else None,
    )

    return {
        "state": state,
        "hours_back": hours_back,
        "recent_rain_npz": str(recent_rain_npz),
        "operational_outputs": {k: str(v) for k, v in paths.items()},
        "tethys_outputs": tethys_paths,
    }