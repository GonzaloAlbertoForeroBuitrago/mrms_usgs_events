from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_BASE_DIR = Path("/data/repository_code/unified_data")


def default_pixel_response_out_dir(base_dir: Path) -> Path:
    return Path(base_dir) / "research" / "hydrologic_drivers" / "pixel_response" / "outputs"


def list_sites_from_rain_zarr(base_dir: Path = DEFAULT_BASE_DIR, state: str | None = None) -> list[str]:
    rain_root = Path(base_dir) / "rain_zarr"
    search_root = rain_root / state.upper() if state else rain_root

    sites = []
    for fp in search_root.rglob("*.zarr"):
        if fp.name.endswith(".zarr.done"):
            continue
        sites.append(fp.name.replace(".zarr", "").zfill(8))

    return sorted(set(sites))


def default_research_script() -> Path:
    return Path(__file__).resolve().parents[1] / "research" / "pixel_response_one_basin.py"


def run_pixel_response_one(
    *,
    site_id: str,
    base_dir: Path = DEFAULT_BASE_DIR,
    script: Path | None = None,
    out_dir: Path | None = None,
    log_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    site_id = str(site_id).zfill(8)
    base_dir = Path(base_dir)

    out_dir = Path(out_dir) if out_dir else default_pixel_response_out_dir(base_dir)
    log_dir = Path(log_dir) if log_dir else base_dir / "logs" / "pixel_response"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{site_id}_pixel_response_summary.parquet"
    log_file = log_dir / f"{site_id}.log"

    if out_file.exists() and not overwrite:
        return out_file

    script = Path(script) if script else default_research_script()
    if not script.exists():
        raise FileNotFoundError(f"Pixel response script not found: {script}")

    env = os.environ.copy()
    env["SITE_ID"] = site_id
    env["BASE_DIR"] = str(base_dir)
    env["PIXEL_RESPONSE_OUT_DIR"] = str(out_dir)

    print(f"[RUN] {site_id}")
    print(f"  script : {script}")
    print(f"  out    : {out_file}")
    print(f"  log    : {log_file}")

    with open(log_file, "w") as log:
        proc = subprocess.run(
            ["python", str(script)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    print(f"[SITE {site_id}] returncode={proc.returncode}")

    if proc.returncode != 0:
        raise RuntimeError(f"pixel_response failed for {site_id}. See log: {log_file}")

    if not out_file.exists():
        legacy_out = default_pixel_response_out_dir(base_dir) / f"{site_id}_pixel_response_summary.parquet"
        if legacy_out.exists() and legacy_out != out_file:
            shutil.copy2(legacy_out, out_file)

    if not out_file.exists():
        raise FileNotFoundError(
            f"Expected output was not created: {out_file}. See log: {log_file}"
        )

    return out_file


def run_pixel_response_many(
    *,
    base_dir: Path = DEFAULT_BASE_DIR,
    state: str | None = None,
    script: Path | None = None,
    out_dir: Path | None = None,
    log_dir: Path | None = None,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict:
    base_dir = Path(base_dir)
    out_dir = Path(out_dir) if out_dir else default_pixel_response_out_dir(base_dir)
    log_dir = Path(log_dir) if log_dir else base_dir / "logs" / "pixel_response"

    sites = list_sites_from_rain_zarr(base_dir=base_dir, state=state)
    if limit:
        sites = sites[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    site_list_fp = out_dir / "all_sites_from_zarr.txt"
    site_list_fp.write_text("\n".join(sites) + "\n", encoding="utf-8")

    print("=" * 100)
    print("PIXEL RESPONSE MANY")
    print("=" * 100)
    print(f"base_dir : {base_dir}")
    print(f"state    : {state}")
    print(f"sites    : {len(sites)}")
    print(f"out_dir  : {out_dir}")
    print(f"log_dir  : {log_dir}")

    ok = []
    failed = []

    for i, site_id in enumerate(sites, 1):
        print("-" * 100)
        print(f"[{i}/{len(sites)}] {site_id}")

        try:
            out_fp = run_pixel_response_one(
                site_id=site_id,
                base_dir=base_dir,
                script=script,
                out_dir=out_dir,
                log_dir=log_dir,
                overwrite=overwrite,
            )
            ok.append(str(out_fp))
            print(f"[OK {i}/{len(sites)}] {site_id}")
        except Exception as e:
            failed.append({"site_id": site_id, "error": f"{type(e).__name__}: {e}"})
            print(f"[FAIL {i}/{len(sites)}] {site_id}: {e}")

    return {
        "base_dir": str(base_dir),
        "state": state,
        "n_sites": len(sites),
        "ok": len(ok),
        "failed": failed,
        "site_list": str(site_list_fp),
        "out_dir": str(out_dir),
    }


def load_pixel_response(base_dir: Path, site_id: str) -> pd.DataFrame:
    site_id = str(site_id).zfill(8)
    fp = default_pixel_response_out_dir(Path(base_dir)) / f"{site_id}_pixel_response_summary.parquet"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_parquet(fp)