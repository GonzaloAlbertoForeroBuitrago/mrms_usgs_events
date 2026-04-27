from __future__ import annotations

from pathlib import Path
import json
import typer
import pandas as pd
from .config import PipelineConfig
from .logger import setup_logging, get_logger
from .pipeline import download_single_site
from .mrms import build_zarr_radaronly_from_timerange
from .mrms_parallel import (build_zarr_radaronly_from_timerange_parallel, write_current_manifest)
from .usgs_api import fetch_monitoring_location, download_basin_json, download_stage_parquet, build_basin_json
from .io import load_stage_with_utc_local, now_utc_iso, resolve_iana_timezone
from multiprocessing import get_context

app = typer.Typer(add_completion=False, help="USGS → events → MRMS RadarOnly Zarr pipeline (resume-safe).")
masks_app = typer.Typer(help="Build MRMS spatial masks and basin indexes.")
app.add_typer(masks_app, name="masks")
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

@app.command("rain-manual-parallel")
def rain_manual_parallel_cmd(
    site_id: str = typer.Option(..., "--site-id"),
    state: str = typer.Option(..., "--state"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    workers: int = typer.Option(4, "--workers", help="Parallel hourly workers."),
) -> None:
    """
    Download MRMS RadarOnly rainfall for a manual/current time window using
    parallel hourly workers.

    This command does not modify run-site, run-many, or rain-manual.
    Outputs are stored under current_runs/ to keep operational/current data
    separate from historical event products.
    """
    cfg = PipelineConfig(base_dir=base_dir.resolve())

    state = state.upper()
    workers = max(1, min(int(workers), int(cfg.max_workers_cap)))

    p2 = site_id[:2]
    p4 = site_id[:4]

    run_id = (
        f"{pd.Timestamp(start).strftime('%Y%m%dT%H%M%S')}_"
        f"{pd.Timestamp(end).strftime('%Y%m%dT%H%M%S')}"
    )

    shared_dir = cfg.base_dir
    current_dir = cfg.base_dir / "current_runs" / state / p2 / p4 / site_id / run_id

    basin_json = shared_dir / "basins_json" / state / p2 / p4 / f"{site_id}.json"
    site_meta_json = shared_dir / "site_meta" / state / p2 / p4 / f"{site_id}_monitoring_location.json"

    stage_parquet = current_dir / "stage" / f"{site_id}_current.parquet"
    stage_local_parquet = current_dir / "stage" / f"{site_id}_current_local.parquet"

    out_zarr = current_dir / "rain_zarr" / f"{site_id}_current.zarr"
    missing_csv = current_dir / "missing" / f"{site_id}_current_missing_radaronly_hours.csv"
    manifest_json = current_dir / "manifest.json"

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

    hours_n, pixels_n, files_ok = build_zarr_radaronly_from_timerange_parallel(
        cfg=cfg,
        start=start,
        end=end,
        basin_json=basin_json,
        out_zarr=out_zarr,
        missing_csv=missing_csv,
        workers=workers,
    )

    write_current_manifest(
        site_id=site_id,
        state=state,
        start=start,
        end=end,
        workers=workers,
        basin_json=basin_json,
        site_meta_json=site_meta_json,
        stage_parquet=stage_parquet,
        stage_local_parquet=stage_local_parquet,
        out_zarr=out_zarr,
        missing_csv=missing_csv,
        manifest_json=manifest_json,
    )

    typer.echo(
        f"[{site_id}] rain-manual-parallel completed.\n"
        f"Mode: current/manual parallel\n"
        f"Workers: {workers}\n"
        f"Meta: {site_meta_json}\n"
        f"Stage UTC: {stage_parquet}\n"
        f"Stage local: {stage_local_parquet}\n"
        f"Basin: {basin_json}\n"
        f"Rain Zarr: {out_zarr}\n"
        f"Missing CSV: {missing_csv}\n"
        f"Manifest: {manifest_json}\n"
        f"Hours: {hours_n}\n"
        f"Pixels: {pixels_n}\n"
        f"Files OK: {files_ok}"
    )

def _run_one_site(args):
    site_id, state, start, end, cfg, hour_workers = args

    try:
        p2 = site_id[:2]
        p4 = site_id[:4]

        run_id = (
            f"{pd.Timestamp(start).strftime('%Y%m%dT%H%M%S')}_"
            f"{pd.Timestamp(end).strftime('%Y%m%dT%H%M%S')}"
        )

        current_dir = (
            cfg.base_dir
            / "current_runs"
            / state
            / p2
            / p4
            / site_id
            / run_id
        )

        basin_json = cfg.base_dir / "basins_json" / state / p2 / p4 / f"{site_id}.json"
        basin_done = cfg.base_dir / "basins_json" / state / p2 / p4 / f"{site_id}.basin.done"

        site_meta_json = cfg.base_dir / "site_meta" / state / p2 / p4 / f"{site_id}_monitoring_location.json"

        stage_parquet = current_dir / "stage" / f"{site_id}_current.parquet"
        stage_local_parquet = current_dir / "stage" / f"{site_id}_current_local.parquet"

        out_zarr = current_dir / "rain_zarr" / f"{site_id}_current.zarr"
        missing_csv = current_dir / "missing" / f"{site_id}_current_missing_radaronly_hours.csv"
        manifest_json = current_dir / "manifest.json"

        out_zarr.parent.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # BASIN JSON: same robust logic as run-site/run-many
        # 1) direct download
        # 2) fallback build from hydrolocation
        # ============================================================
        basin_json.parent.mkdir(parents=True, exist_ok=True)
        basin_done.parent.mkdir(parents=True, exist_ok=True)

        basin_ok = basin_json.exists()

        if not basin_ok:
            try:
                download_basin_json(
                    cfg,
                    site_id,
                    basin_json,
                    basin_done,
                    overwrite=False,
                )
                basin_ok = basin_json.exists()
            except Exception:
                basin_ok = False

        if not basin_ok:
            try:
                build_basin_json(
                    cfg,
                    site_id,
                    basin_json,
                    basin_done,
                    overwrite=False,
                )
                basin_ok = basin_json.exists()
            except Exception as e:
                return f"[ERROR] {site_id}: basin build failed: {type(e).__name__}: {e}"

        if not basin_ok:
            return f"[ERROR] {site_id}: basin_json missing after download/build"

        # ============================================================
        # METADATA + STAGE
        # ============================================================
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

        # ============================================================
        # RAINFALL ZARR
        # For many sites, hours are processed serially per site.
        # Parallelism is already happening by site.
        # ============================================================
        build_zarr_radaronly_from_timerange(
            cfg=cfg,
            start=start,
            end=end,
            basin_json=basin_json,
            out_zarr=out_zarr,
            missing_csv=missing_csv,
        )

        # ============================================================
        # MANIFEST
        # ============================================================
        write_current_manifest(
            site_id=site_id,
            state=state,
            start=start,
            end=end,
            workers=hour_workers,
            basin_json=basin_json,
            site_meta_json=site_meta_json,
            stage_parquet=stage_parquet,
            stage_local_parquet=stage_local_parquet,
            out_zarr=out_zarr,
            missing_csv=missing_csv,
            manifest_json=manifest_json,
        )

        return f"[OK] {site_id}"

    except Exception as e:
        return f"[ERROR] {site_id}: {type(e).__name__}: {e}"


@app.command("rain-current-many")
def rain_current_many_cmd(
    sites_file: Path = typer.Option(..., "--sites-file"),
    state: str = typer.Option(..., "--state"),
    hours_back: int = typer.Option(12, "--hours-back"),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    site_workers: int = typer.Option(4, "--site-workers"),
    hour_workers: int = typer.Option(1, "--hour-workers"),
) -> None:
    """
    Run current rainfall extraction for MANY sites using last N hours.

    Parallelism is by SITE.
    For many sites and short windows, each site processes hours serially.
    """

    cfg = PipelineConfig(base_dir=base_dir.resolve())
    state = state.upper()

    end_ts = pd.Timestamp.utcnow().floor("h")
    start_ts = end_ts - pd.Timedelta(hours=hours_back)

    start = start_ts.strftime("%Y-%m-%d %H:%M:%S")
    end = end_ts.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("RAIN CURRENT MANY")
    print("=" * 80)
    print(f"State: {state}")
    print(f"Sites file: {sites_file}")
    print(f"Start: {start}")
    print(f"End: {end}")
    print(f"Site workers: {site_workers}")
    print(f"Hour workers per site: {hour_workers}")
    print("=" * 80)

    df_sites = pd.read_csv(
        sites_file,
        sep=r"\s+",
        dtype={"site_id": str, "state": str},
    )

    df_sites["state"] = df_sites["state"].str.upper()

    site_ids = (
        df_sites.loc[df_sites["state"] == state, "site_id"]
        .dropna()
        .astype(str)
        .str.zfill(8)
        .drop_duplicates()
        .tolist()
    )

    if not site_ids:
        raise typer.BadParameter(
            f"No sites found for state={state} in {sites_file}"
        )

    site_workers = max(1, min(int(site_workers), len(site_ids)))

    print(f"Total sites for {state}: {len(site_ids)}")
    print(f"Starting parallel execution with {site_workers} workers...\n")

    tasks = [(sid, state, start, end, cfg, int(hour_workers)) for sid in site_ids]

    ctx = get_context("fork")

    ok = 0
    error = 0

    with ctx.Pool(processes=site_workers) as pool:
        for msg in pool.imap_unordered(_run_one_site, tasks):
            print(msg, flush=True)
            if str(msg).startswith("[OK]"):
                ok += 1
            else:
                error += 1

    print("\nDONE.")
    print(f"OK: {ok}")
    print(f"ERROR: {error}")

@masks_app.command("build-input")
def masks_build_input_cmd(
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    basins_dir: Path | None = typer.Option(None, "--basins-dir"),
    out: Path | None = typer.Option(None, "--out"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    from .masks.build_mask_input import build_mask_input

    base_dir = base_dir.resolve()
    basins_dir = basins_dir or (base_dir / "basins_json")
    out = out or (base_dir / "masks" / "mask_input.tsv")

    build_mask_input(
        basins_dir=basins_dir,
        out_fp=out,
        overwrite=overwrite,
    )


@masks_app.command("build-state-masks")
def masks_build_state_masks_cmd(
    sample_grib_gz: Path = typer.Option(..., "--sample-grib-gz"),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    mask_input: Path | None = typer.Option(None, "--mask-input"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
    state: str | None = typer.Option(None, "--state"),
    dtype: str = typer.Option("float32", "--dtype"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    from .masks.state_masks import build_state_mrms_masks

    base_dir = base_dir.resolve()
    mask_input = mask_input or (base_dir / "masks" / "mask_input.tsv")
    out_dir = out_dir or (base_dir / "masks" / "state_mrms_masks")

    build_state_mrms_masks(
        mask_input=mask_input,
        sample_grib_gz=sample_grib_gz,
        out_dir=out_dir,
        state=state,
        dtype=dtype,
        overwrite=overwrite,
    )


@masks_app.command("build-basin-masks")
def masks_build_basin_masks_cmd(
    sample_grib_gz: Path = typer.Option(..., "--sample-grib-gz"),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    mask_input: Path | None = typer.Option(None, "--mask-input"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
    dtype: str = typer.Option("float32", "--dtype"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    from .masks.basin_masks import build_basin_mrms_masks

    base_dir = base_dir.resolve()
    mask_input = mask_input or (base_dir / "masks" / "mask_input.tsv")
    out_dir = out_dir or (base_dir / "masks" / "basin_mrms_masks")

    build_basin_mrms_masks(
        mask_input=mask_input,
        sample_grib_gz=sample_grib_gz,
        out_dir=out_dir,
        dtype=dtype,
        overwrite=overwrite,
    )


@masks_app.command("build-state-basin-index")
def masks_build_state_basin_index_cmd(
    sample_grib_gz: Path = typer.Option(..., "--sample-grib-gz"),
    base_dir: Path = typer.Option(Path("usgs_mrms_events_data"), "--base-dir"),
    mask_input: Path | None = typer.Option(None, "--mask-input"),
    state_mask_dir: Path | None = typer.Option(None, "--state-mask-dir"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
    state: str | None = typer.Option(None, "--state"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    from .masks.state_basin_index import build_state_basin_index

    base_dir = base_dir.resolve()
    mask_input = mask_input or (base_dir / "masks" / "mask_input.tsv")
    state_mask_dir = state_mask_dir or (base_dir / "masks" / "state_mrms_masks")
    out_dir = out_dir or (base_dir / "masks" / "state_basin_index")

    build_state_basin_index(
        mask_input=mask_input,
        state_mask_dir=state_mask_dir,
        sample_grib_gz=sample_grib_gz,
        out_dir=out_dir,
        state=state,
        overwrite=overwrite,
    )

from .ews.cli_commands import ews_app
app.add_typer(ews_app, name="ews")