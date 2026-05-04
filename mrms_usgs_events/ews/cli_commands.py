from __future__ import annotations

from pathlib import Path

import typer

from .historical_summary import build_site_historical_summary
from .state_historical_summary import build_state_historical_summary_parallel
from .state_rain import build_current_state_rain_npz
from .current_alerts import compute_current_alerts_for_state

ews_app = typer.Typer(help="Operational Early Warning System tools.")


@ews_app.command("build-site-history")
def ews_build_site_history_cmd(
    site_id: str = typer.Option(..., "--site-id", help="USGS site id."),
    state: str = typer.Option(..., "--state", help="State name, e.g. TEXAS."),
    base_dir: Path = typer.Option(Path("/data/repository_code/unified_data"), "--base-dir"),
    out_dir: Path = typer.Option(Path("/data/repository_code/unified_data/hydro_history"), "--out-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    out = build_site_historical_summary(
        base_dir=base_dir,
        site_id=site_id,
        state=state,
        out_dir=out_dir,
        overwrite=overwrite,
    )
    typer.echo(f"Saved: {out}")


@ews_app.command("build-state-history")
def ews_build_state_history_cmd(
    state: str = typer.Option(..., "--state"),
    state_basin_index: Path = typer.Option(..., "--state-basin-index"),
    base_dir: Path = typer.Option(Path("/data/repository_code/unified_data"), "--base-dir"),
    out_dir: Path = typer.Option(Path("/data/repository_code/unified_data/hydro_history"), "--out-dir"),
    workers: int = typer.Option(4, "--workers"),
    overwrite_sites: bool = typer.Option(False, "--overwrite-sites"),
    overwrite_index: bool = typer.Option(True, "--overwrite-index/--no-overwrite-index"),
    min_pixel_value: float = typer.Option(7.5, "--min-pixel-value"),
    only_stage_response_p50: bool = typer.Option(
        True,
        "--only-stage-response-p50/--all-stage-events",
    ),
    index_batch_size: int = typer.Option(10, "--index-batch-size"),
):
    out = build_state_historical_summary_parallel(
        base_dir=base_dir,
        state=state,
        state_basin_index_fp=state_basin_index,
        out_dir=out_dir,
        workers=workers,
        overwrite_sites=overwrite_sites,
        overwrite_index=overwrite_index,
        min_pixel_value=min_pixel_value,
        only_stage_response_p50=only_stage_response_p50,
        index_batch_size=index_batch_size,
    )
    typer.echo(out)


@ews_app.command("state-rain-current")
def ews_state_rain_current_cmd(
    state: str = typer.Option(..., "--state"),
    state_mask: Path = typer.Option(..., "--state-mask"),
    out_npz: Path = typer.Option(..., "--out-npz"),
    base_dir: Path = typer.Option(Path("/data/repository_code/unified_data"), "--base-dir"),
    hours_back: int = typer.Option(12, "--hours-back"),
    workers: int = typer.Option(4, "--workers"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
):
    out = build_current_state_rain_npz(
        state=state,
        state_mask_fp=state_mask,
        out_npz=out_npz,
        base_dir=base_dir,
        hours_back=hours_back,
        workers=workers,
        start=start,
        end=end,
    )
    typer.echo(f"Saved: {out}")


@ews_app.command("run-current-alerts")
def ews_run_current_alerts_cmd(
    state: str = typer.Option(..., "--state"),
    current_rain_npz: Path = typer.Option(..., "--current-rain-npz"),
    state_basin_index: Path = typer.Option(..., "--state-basin-index"),
    pixel_event_index: Path = typer.Option(..., "--pixel-event-index"),
    out_dir: Path = typer.Option(Path("/data/repository_code/unified_data/ews_alerts"), "--out-dir"),
    strong_threshold: float = typer.Option(7.5, "--strong-threshold"),
    k_matches: int = typer.Option(5, "--k-matches"),
    accumulation_quantile: float = typer.Option(0.0, "--accumulation-quantile"),
    warning_delta_threshold: float = typer.Option(2.0, "--warning-delta-threshold"),
    severe_delta_threshold: float = typer.Option(10.0, "--severe-delta-threshold"),
    max_pixels_per_basin_output: int | None = typer.Option(500, "--max-pixels-per-basin-output"),
):
    out = compute_current_alerts_for_state(
        state=state,
        current_rain_npz=current_rain_npz,
        state_basin_index_npz=state_basin_index,
        pixel_event_index_npz=pixel_event_index,
        out_dir=out_dir,
        strong_threshold=strong_threshold,
        k_matches=k_matches,
        accumulation_quantile=accumulation_quantile,
        warning_delta_threshold=warning_delta_threshold,
        severe_delta_threshold=severe_delta_threshold,
        max_pixels_per_basin_output=max_pixels_per_basin_output,
    )
    typer.echo(out)


@ews_app.command("run-state-operational")
def ews_run_state_operational_cmd(
    state: str = typer.Option(..., "--state"),
    base_dir: Path = typer.Option(Path("/data/repository_code/unified_data"), "--base-dir"),
    state_mask: Path | None = typer.Option(None, "--state-mask"),
    state_basin_index: Path | None = typer.Option(None, "--state-basin-index"),
    pixel_event_index: Path | None = typer.Option(None, "--pixel-event-index"),
    current_rain_npz: Path | None = typer.Option(None, "--current-rain-npz"),
    alerts_out_dir: Path | None = typer.Option(None, "--alerts-out-dir"),
    hours_back: int = typer.Option(12, "--hours-back"),
    workers: int = typer.Option(4, "--workers"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    strong_threshold: float = typer.Option(7.5, "--strong-threshold"),
    k_matches: int = typer.Option(5, "--k-matches"),
    accumulation_quantile: float = typer.Option(0.0, "--accumulation-quantile"),
    warning_delta_threshold: float = typer.Option(2.0, "--warning-delta-threshold"),
    severe_delta_threshold: float = typer.Option(10.0, "--severe-delta-threshold"),
    max_pixels_per_basin_output: int | None = typer.Option(500, "--max-pixels-per-basin-output"),
):
    state = state.upper()

    if state_mask is None:
        state_mask = base_dir / "state_masks" / f"{state}_mrms_mask.npz"

    if state_basin_index is None:
        state_basin_index = base_dir / "state_basin_index" / f"{state}_state_basin_index.npz"

    if pixel_event_index is None:
        pixel_event_index = (
            base_dir
            / "hydro_history"
            / "state_pixel_event_index"
            / f"{state}_pixel_event_index.npz"
        )

    if current_rain_npz is None:
        current_rain_npz = base_dir / "current_rain" / f"{state}_current_rain.npz"

    if alerts_out_dir is None:
        alerts_out_dir = base_dir / "ews_alerts"

    rain_out = build_current_state_rain_npz(
        state=state,
        state_mask_fp=state_mask,
        out_npz=current_rain_npz,
        base_dir=base_dir,
        hours_back=hours_back,
        workers=workers,
        start=start,
        end=end,
    )

    alert_out = compute_current_alerts_for_state(
        state=state,
        current_rain_npz=rain_out,
        state_basin_index_npz=state_basin_index,
        pixel_event_index_npz=pixel_event_index,
        out_dir=alerts_out_dir,
        strong_threshold=strong_threshold,
        k_matches=k_matches,
        accumulation_quantile=accumulation_quantile,
        warning_delta_threshold=warning_delta_threshold,
        severe_delta_threshold=severe_delta_threshold,
        max_pixels_per_basin_output=max_pixels_per_basin_output,
    )

    typer.echo(
        {
            "current_rain": rain_out,
            "alerts": alert_out,
        }
    )


@ews_app.command("export-state-tethys")
def ews_export_state_tethys_cmd(
    state: str = typer.Option(..., "--state"),
    base_dir: Path = typer.Option(Path("/data/repository_code/unified_data"), "--base-dir"),
    alerts_parquet: Path | None = typer.Option(None, "--alerts-parquet"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
    public_dir: Path | None = typer.Option(None, "--public-dir"),
):
    from .tethys_outputs import export_state_alerts_for_tethys

    result = export_state_alerts_for_tethys(
        state=state,
        base_dir=base_dir,
        alerts_dir=alerts_parquet.parent if alerts_parquet is not None else None,
        out_dir=out_dir,
        public_dir=public_dir,
        )
    typer.echo(result)


@ews_app.command("run-state-tethys")
def ews_run_state_tethys_cmd(
    state: str = typer.Option(..., "--state"),
    base_dir: Path = typer.Option(Path("/data/repository_code/unified_data"), "--base-dir"),
    hours_back: int = typer.Option(12, "--hours-back"),
    workers: int = typer.Option(4, "--workers"),
    public_dir: Path | None = typer.Option(None, "--public-dir"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
):
    from .tethys_service import run_state_current_alert_for_tethys

    result = run_state_current_alert_for_tethys(
        state=state,
        base_dir=base_dir,
        hours_back=hours_back,
        workers=workers,
        public_dir=public_dir,
        start=start,
        end=end,
    )
    typer.echo(result)