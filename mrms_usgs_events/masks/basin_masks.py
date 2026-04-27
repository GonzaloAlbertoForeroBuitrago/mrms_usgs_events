from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

from .utils import open_sample_mrms, load_geometry, rasterize_geometry


def build_basin_mrms_masks(
    mask_input: Path,
    sample_grib_gz: Path,
    out_dir: Path,
    overwrite: bool = False,
    dtype: str = "float32",
) -> Path:
    mask_input = Path(mask_input)
    sample_grib_gz = Path(sample_grib_gz)
    out_dir = Path(out_dir)

    df = pd.read_csv(mask_input, sep="\t", dtype={"site_id": str})
    out_dir.mkdir(parents=True, exist_ok=True)

    gdal.UseExceptions()
    sample_ds, vs = open_sample_mrms(sample_grib_gz, "_basin_mask_sample.grib2")

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            site_id = str(row.site_id)
            basin_fp = Path(row.path)
            out_fp = out_dir / f"{site_id}.npz"

            if out_fp.exists() and not overwrite:
                continue

            geom = load_geometry(basin_fp)
            rows, cols, lon, lat = rasterize_geometry(
                geom,
                sample_ds,
                include_lon_lat=True,
                dtype=dtype,
            )

            np.savez_compressed(
                out_fp,
                site_id=np.array(site_id),
                rows=rows,
                cols=cols,
                lon=lon,
                lat=lat,
            )

            if i % 100 == 0:
                print(f"[{i}/{len(df)}] {site_id} pixels={len(rows)}", flush=True)

    finally:
        sample_ds = None
        gdal.Unlink(vs)

    print(f"[DONE] basin masks saved in: {out_dir}")
    return out_dir