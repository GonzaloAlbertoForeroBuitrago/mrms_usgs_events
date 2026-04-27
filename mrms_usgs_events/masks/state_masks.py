from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal

from .utils import open_sample_mrms, load_geometry, rasterize_geometry


def rasterize_state_mask(
    state: str,
    state_df: pd.DataFrame,
    sample_ds,
    dtype: str = "float32",
) -> dict:
    gt = sample_ds.GetGeoTransform()
    proj_wkt = sample_ds.GetProjection()
    nx = sample_ds.RasterXSize
    ny = sample_ds.RasterYSize

    geometries = [load_geometry(Path(fp)) for fp in state_df["path"]]

    gdf = gpd.GeoDataFrame(
        {"id": np.arange(len(geometries), dtype=np.int32)},
        geometry=geometries,
        crs="EPSG:4326",
    )

    union_geom = gdf.geometry.union_all()

    rows, cols, lon, lat = rasterize_geometry(
        union_geom,
        sample_ds,
        include_lon_lat=True,
        dtype=dtype,
    )

    if rows.size == 0:
        raise RuntimeError(f"Empty state mask for {state}")

    return {
        "state": state,
        "site_ids": state_df["site_id"].astype(str).to_numpy(),
        "rows": rows,
        "cols": cols,
        "lon": lon,
        "lat": lat,
        "gt": np.array(gt, dtype=np.float64),
        "proj_wkt": np.array(proj_wkt),
        "nx": np.int32(nx),
        "ny": np.int32(ny),
        "n_basins": np.int32(len(state_df)),
        "n_pixels": np.int32(rows.size),
    }


def build_state_mrms_masks(
    mask_input: Path,
    sample_grib_gz: Path,
    out_dir: Path,
    state: str | None = None,
    overwrite: bool = False,
    dtype: str = "float32",
) -> Path:
    mask_input = Path(mask_input)
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
    print("BUILD STATE MRMS MASKS")
    print("=" * 90)
    print(f"mask_input : {mask_input}")
    print(f"sample_gz  : {sample_grib_gz}")
    print(f"out_dir    : {out_dir}")
    print(f"states     : {df['state'].nunique()}")
    print(f"basins     : {len(df)}")

    gdal.UseExceptions()
    sample_ds, vs = open_sample_mrms(sample_grib_gz, "_state_mask_sample.grib2")

    try:
        for state_name, state_df in df.groupby("state", sort=True):
            out_fp = out_dir / f"{state_name}_mrms_mask.npz"

            if out_fp.exists() and not overwrite:
                print(f"[SKIP] {state_name}: exists {out_fp}")
                continue

            print()
            print("-" * 90)
            print(f"[STATE] {state_name}")
            print(f"basins: {len(state_df)}")

            result = rasterize_state_mask(
                state=state_name,
                state_df=state_df,
                sample_ds=sample_ds,
                dtype=dtype,
            )

            np.savez_compressed(out_fp, **result)

            print(f"pixels: {int(result['n_pixels'])}")
            print(f"saved : {out_fp}")

    finally:
        sample_ds = None
        gdal.Unlink(vs)

    print(f"[DONE] state masks saved in: {out_dir}")
    return out_dir