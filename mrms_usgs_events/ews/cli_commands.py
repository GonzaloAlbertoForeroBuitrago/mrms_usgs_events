from __future__ import annotations

from pathlib import Path

import typer

from .historical_summary import build_many_historical_summaries, build_site_historical_summary
from .predictors import fit_basin_predictors
from .realtime_engine import run_state_alert_engine
from .state_rain import build_current_state_rain_npz

ews_app = typer.Typer(help="Operational Early Warning System tools.")


@ews_app.command("build-history")
def ews_build_history_cmd(
    site_id: str = typer.Option(..., "--site-id", help="USGS site id."),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    out_dir: Path = typer.Option(Path("ews_outputs/historical_basin_summaries"), "--out-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    out = build_site_historical_summary(base_dir=base_dir, site_id=site_id, out_dir=out_dir, overwrite=overwrite)
    typer.echo(f"Saved: {out}")


@ews_app.command("build-history-many")
def ews_build_history_many_cmd(
    mask_input: Path = typer.Option(..., "--mask-input", help="TSV with site_id/state/path."),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    out_dir: Path = typer.Option(Path("ews_outputs/historical_basin_summaries"), "--out-dir"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    ok, fail = build_many_historical_summaries(base_dir=base_dir, mask_input=mask_input, out_dir=out_dir, overwrite=overwrite)
    typer.echo(f"Finished. ok={ok} fail={fail}")


@ews_app.command("fit-predictors")
def ews_fit_predictors_cmd(
    summary_dir: Path = typer.Option(..., "--summary-dir"),
    out_dir: Path = typer.Option(Path("ews_outputs/basin_predictors"), "--out-dir"),
):
    out = fit_basin_predictors(summary_dir=summary_dir, out_dir=out_dir)
    typer.echo(f"Saved: {out}")


@ews_app.command("run-state")
def ews_run_state_cmd(
    state: str = typer.Option(..., "--state"),
    recent_rain_npz: Path = typer.Option(..., "--recent-rain-npz"),
    state_basin_index: Path = typer.Option(..., "--state-basin-index"),
    predictor_dir: Path = typer.Option(..., "--predictor-dir"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    historical_summary_dir: Path | None = typer.Option(None, "--historical-summary-dir"),
):
    paths = run_state_alert_engine(
        state=state,
        recent_rain_npz=recent_rain_npz,
        state_basin_index=state_basin_index,
        predictor_dir=predictor_dir,
        out_dir=out_dir,
        historical_summary_dir=historical_summary_dir,
    )
    for name, fp in paths.items():
        typer.echo(f"{name}: {fp}")
@ews_app.command("state-rain-current")
def ews_state_rain_current_cmd(
    state: str = typer.Option(..., "--state"),
    state_mask: Path = typer.Option(..., "--state-mask"),
    out_npz: Path = typer.Option(..., "--out-npz"),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
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