from __future__ import annotations

from pathlib import Path
import json
import typer

from .config import PipelineConfig
from .logger import setup_logging, get_logger
from .pipeline import download_single_site
from .mrms import build_zarr_radaronly_from_timerange
from .usgs_api import fetch_monitoring_location, download_basin_json, download_stage_parquet
from .io import load_stage_with_utc_local, now_utc_iso, resolve_iana_timezone

app = typer.Typer(add_completion=False, help="USGS → events → MRMS RadarOnly Zarr pipeline (resume-safe).")
log = get_logger("usgs_mrms_events.cli")

@app.command("run-site")
def run_site_cmd(
    site_id: str = typer.Argument(..., help="USGS site id (digits or 'USGS-XXXXXXXX')."),
    start: str = typer.Option("2019-04-01", "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2026-01-30", "--end", help="End date (YYYY-MM-DD)."),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir", help="Base output folder."),
    log_dir: Path | None = typer.Option(None, "--log-dir", help="Log folder (default: <base_dir>/logs)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs for this site."),
) -> None:
    """
    Run the full pipeline for a single site.
    """
    cfg = PipelineConfig(base_dir=base_dir.resolve(), log_dir=log_dir)
    paths = setup_logging(log_dir=cfg.log_dir)
    log.info(f"run-site | site_id={site_id} base_dir={cfg.base_dir} log={paths.run_log}")


    result = download_single_site(
        site_id=site_id,
        start_date=start,
        end_date=end,
        base_dir=cfg.base_dir,
        overwrite=overwrite,
        config=cfg,
    )
    typer.echo(f"[{result['site_id']}] completed. Rain Zarr: {result['paths']['rain_zarr']}")


@app.command("run-many")
def run_many_cmd(
    sites_file: Path = typer.Argument(..., help="Text file with one site_id per line."),
    start: str = typer.Option("2019-04-01", "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2026-01-30", "--end", help="End date (YYYY-MM-DD)."),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir", help="Base output folder."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs per site."),
) -> None:
    """
    Run the pipeline for many sites from a file (one site per line).
    """
    from .pipeline import download_many_sites

    site_ids = [ln.strip() for ln in sites_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = download_many_sites(site_ids, start_date=start, end_date=end, base_dir=base_dir, overwrite=overwrite)
    typer.echo(out)



def ensure_manual_inputs(
    cfg: PipelineConfig,
    site_id: str,
    start: str,
    end: str,
    basin_json: Path,
    stage_parquet: Path,
    stage_local_parquet: Path,
    site_meta_json: Path,
) -> None:
    feat = fetch_monitoring_location(cfg, site_id)

    site_meta_json.parent.mkdir(parents=True, exist_ok=True)
    site_meta_json.write_text(json.dumps(feat, indent=2), encoding="utf-8")
    site_meta_json.with_name(site_meta_json.stem.replace("_monitoring_location", "") + ".meta.done").write_text(
        now_utc_iso(),
        encoding="utf-8",
    )

    basin_json.parent.mkdir(parents=True, exist_ok=True)
    download_basin_json(
        cfg,
        site_id,
        basin_json,
        basin_json.with_suffix(".basin.done"),
        overwrite=False,
    )

    stage_parquet.parent.mkdir(parents=True, exist_ok=True)
    download_stage_parquet(
        cfg,
        site_id,
        stage_parquet,
        stage_parquet.with_suffix(".stage.done"),
        start_date=start[:10],
        end_date=end[:10],
        overwrite=False,
    )

    props = feat.get("properties", {}) or {}
    geom = feat.get("geometry", {}) or {}
    coords = geom.get("coordinates", [None, None])

    lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
    lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None

    tz_iana = resolve_iana_timezone(lon, lat, props.get("time_zone_abbreviation"))
    load_stage_with_utc_local(stage_parquet, tz_iana).to_parquet(stage_local_parquet, index=False)


@app.command("rain-manual")
def rain_manual_cmd(
    site_id: str = typer.Option(..., "--site-id"),
    state: str = typer.Option(..., "--state"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
) -> None:
    cfg = PipelineConfig(base_dir=base_dir.resolve())

    p2 = site_id[:2]
    p4 = site_id[:4]

    basin_json = cfg.base_dir / "basins_json" / state / p2 / p4 / f"{site_id}.json"
    site_meta_json = cfg.base_dir / "site_meta" / state / p2 / p4 / f"{site_id}_monitoring_location.json"
    stage_parquet = cfg.base_dir / "stage_parquet" / state / p2 / p4 / f"{site_id}.parquet"
    stage_local_parquet = cfg.base_dir / "stage_parquet" / state / p2 / p4 / f"{site_id}_local.parquet"
    out_zarr = cfg.base_dir / "rain_zarr" / state / p2 / p4 / f"{site_id}_manual.zarr"
    missing_csv = cfg.base_dir / "rain_zarr" / state / p2 / p4 / f"{site_id}_manual_missing_radaronly_hours.csv"

    out_zarr.parent.mkdir(parents=True, exist_ok=True)

    ensure_manual_inputs(
        cfg=cfg,
        site_id=site_id,
        start=start,
        end=end,
        basin_json=basin_json,
        stage_parquet=stage_parquet,
        stage_local_parquet=stage_local_parquet,
        site_meta_json=site_meta_json,
    )

    hours_n, pixels_n, files_ok = build_zarr_radaronly_from_timerange(
        cfg=cfg,
        start=start,
        end=end,
        basin_json=basin_json,
        out_zarr=out_zarr,
        missing_csv=missing_csv,
    )

    typer.echo(
        f"[{site_id}] rain-manual completed.\n"
        f"Meta: {site_meta_json}\n"
        f"Stage UTC: {stage_parquet}\n"
        f"Stage local: {stage_local_parquet}\n"
        f"Basin: {basin_json}\n"
        f"Rain Zarr: {out_zarr}\n"
        f"Hours: {hours_n}\n"
        f"Pixels: {pixels_n}\n"
        f"Files OK: {files_ok}"
    )

if __name__ == "__main__":
    app()