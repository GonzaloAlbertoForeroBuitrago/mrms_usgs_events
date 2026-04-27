from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

from .utils import open_sample_mrms, load_geometry, rasterize_geometry


def build_index_for_state(
    state: str,
    df_state: pd.DataFrame,
    state_mask_fp: Path,
    sample_ds,
    out_fp: Path,
) -> dict:
    z = np.load(state_mask_fp, allow_pickle=True)

    rows_state = z["rows"].astype(np.int32)
    cols_state = z["cols"].astype(np.int32)
    lon_state = z["lon"].astype(np.float32)
    lat_state = z["lat"].astype(np.float32)

    nx = int(z["nx"])
    ny = int(z["ny"])
    gt = z["gt"]

    linear_state = rows_state.astype(np.int64) * nx + cols_state.astype(np.int64)
    order = np.argsort(linear_state)

    linear_sorted = linear_state[order]
    pos_sorted = order.astype(np.int32)

    site_ids = []
    basin_ptr = [0]
    basin_indices_all = []

    basin_n_pixels = []
    basin_original_pixels = []
    basin_missing_pixels = []
    basin_coverage = []

    for i, row in enumerate(df_state.itertuples(index=False), start=1):
        site_id = str(row.site_id)
        basin_fp = Path(row.path)

        geom = load_geometry(basin_fp)
        rows_basin, cols_basin = rasterize_geometry(
            geom,
            sample_ds,
            include_lon_lat=False,
        )

        linear_basin = rows_basin.astype(np.int64) * nx + cols_basin.astype(np.int64)

        loc = np.searchsorted(linear_sorted, linear_basin)
        valid = (loc < len(linear_sorted)) & (linear_sorted[loc] == linear_basin)

        basin_positions = pos_sorted[loc[valid]]
        basin_positions = np.unique(basin_positions).astype(np.int32)

        n_original = int(len(linear_basin))
        n_inside = int(len(basin_positions))
        n_missing = max(n_original - n_inside, 0)
        coverage = 0.0 if n_original == 0 else n_inside / n_original

        site_ids.append(site_id)
        basin_indices_all.append(basin_positions)
        basin_ptr.append(basin_ptr[-1] + n_inside)

        basin_original_pixels.append(n_original)
        basin_n_pixels.append(n_inside)
        basin_missing_pixels.append(n_missing)
        basin_coverage.append(coverage)

        status = "OK"
        if n_original == 0:
            status = "EMPTY"
        elif coverage < 1.0:
            status = "PARTIAL"

        if i % 50 == 0 or status != "OK":
            print(
                f"[{i:5d}/{len(df_state)}] {site_id} "
                f"basin_pixels={n_original} "
                f"inside_state={n_inside} "
                f"coverage={coverage:.4f} "
                f"{status}",
                flush=True,
            )

    basin_indices = (
        np.concatenate(basin_indices_all).astype(np.int32)
        if basin_indices_all
        else np.array([], dtype=np.int32)
    )

    site_ids = np.array(site_ids, dtype=str)
    basin_ptr = np.array(basin_ptr, dtype=np.int64)
    basin_n_pixels = np.array(basin_n_pixels, dtype=np.int32)
    basin_original_pixels = np.array(basin_original_pixels, dtype=np.int32)
    basin_missing_pixels = np.array(basin_missing_pixels, dtype=np.int32)
    basin_coverage = np.array(basin_coverage, dtype=np.float32)

    np.savez_compressed(
        out_fp,
        state=np.array(state),
        site_ids=site_ids,
        rows=rows_state,
        cols=cols_state,
        lon=lon_state,
        lat=lat_state,
        nx=np.int32(nx),
        ny=np.int32(ny),
        gt=gt,
        basin_ptr=basin_ptr,
        basin_indices=basin_indices,
        basin_n_pixels=basin_n_pixels,
        basin_original_pixels=basin_original_pixels,
        basin_missing_pixels=basin_missing_pixels,
        basin_coverage=basin_coverage,
        n_basins=np.int32(len(site_ids)),
        n_state_pixels=np.int32(len(rows_state)),
        n_basin_index_values=np.int64(len(basin_indices)),
    )

    return {
        "n_basins": int(len(site_ids)),
        "n_state_pixels": int(len(rows_state)),
        "n_basin_index_values": int(len(basin_indices)),
        "min_coverage": float(basin_coverage.min()) if len(basin_coverage) else 0.0,
        "mean_coverage": float(basin_coverage.mean()) if len(basin_coverage) else 0.0,
        "bad": int(np.sum(basin_coverage < 1.0)),
    }


def build_state_basin_index(
    mask_input: Path,
    state_mask_dir: Path,
    sample_grib_gz: Path,
    out_dir: Path,
    state: str | None = None,
    overwrite: bool = False,
) -> Path:
    mask_input = Path(mask_input)
    state_mask_dir = Path(state_mask_dir)
    sample_grib_gz = Path(sample_grib_gz)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(mask_input, sep="\t", dtype={"site_id": str})
    df["state"] = df["state"].astype(str).str.upper()

    if state:
        df = df[df["state"] == state.upper()].copy()

    if df.empty:
        raise RuntimeError("No basins found after filtering. Check --state and --mask-input.")

    print("=" * 90)
    print("BUILD STATE BASIN INDEX")
    print("=" * 90)
    print(f"mask_input     : {mask_input}")
    print(f"state_mask_dir : {state_mask_dir}")
    print(f"out_dir        : {out_dir}")
    print(f"states         : {df['state'].nunique()}")
    print(f"basins         : {len(df)}")

    gdal.UseExceptions()
    sample_ds, vs = open_sample_mrms(sample_grib_gz, "_state_basin_index_sample.grib2")

    try:
        for state_name, df_state in df.groupby("state", sort=True):
            state_mask_fp = state_mask_dir / f"{state_name}_mrms_mask.npz"
            out_fp = out_dir / f"{state_name}_state_basin_index.npz"

            if not state_mask_fp.exists():
                print(f"[SKIP] {state_name}: missing state mask {state_mask_fp}")
                continue

            if out_fp.exists() and not overwrite:
                print(f"[SKIP] {state_name}: exists {out_fp}")
                continue

            print()
            print("-" * 90)
            print(f"[STATE] {state_name}")
            print(f"basins     : {len(df_state)}")
            print(f"state mask : {state_mask_fp}")
            print(f"output     : {out_fp}")

            stats = build_index_for_state(
                state=state_name,
                df_state=df_state,
                state_mask_fp=state_mask_fp,
                sample_ds=sample_ds,
                out_fp=out_fp,
            )

            print()
            print(f"[DONE] {state_name}")
            print(f"n_basins             : {stats['n_basins']}")
            print(f"n_state_pixels       : {stats['n_state_pixels']}")
            print(f"n_basin_index_values : {stats['n_basin_index_values']}")
            print(f"coverage min/mean    : {stats['min_coverage']:.4f} / {stats['mean_coverage']:.4f}")
            print(f"bad coverage < 1     : {stats['bad']}")
            print(f"saved                : {out_fp}")

    finally:
        sample_ds = None
        gdal.Unlink(vs)

    print(f"[DONE] state basin index saved in: {out_dir}")
    return out_dir