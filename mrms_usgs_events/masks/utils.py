from pathlib import Path
import gzip
import json

import numpy as np
from osgeo import gdal, ogr, osr
from shapely.geometry import shape


def open_sample_mrms(sample_gz: Path, vs_name: str = "_mrms_mask_sample.grib2"):
    sample_gz = Path(sample_gz)

    raw = gzip.decompress(sample_gz.read_bytes())
    vs = f"/vsimem/{vs_name}"

    gdal.FileFromMemBuffer(vs, raw)
    ds = gdal.Open(vs)

    if ds is None:
        gdal.Unlink(vs)
        raise RuntimeError(f"GDAL could not open {sample_gz}")

    return ds, vs


def load_geometry(fp: Path):
    fp = Path(fp)
    data = json.loads(fp.read_text(encoding="utf-8"))

    geom = data.get("geometry")
    if geom is None:
        raise RuntimeError(f"No geometry found in {fp}")

    return shape(geom)


def rasterize_geometry(geom, sample_ds, include_lon_lat: bool = True, dtype: str = "float32"):
    gt = sample_ds.GetGeoTransform()
    proj_wkt = sample_ds.GetProjection()
    nx = sample_ds.RasterXSize
    ny = sample_ds.RasterYSize

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    drv = ogr.GetDriverByName("MEM")
    dsv = drv.CreateDataSource("mem")
    lyr = dsv.CreateLayer("geometry", srs=srs, geom_type=ogr.wkbUnknown)

    feat = ogr.Feature(lyr.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkt(geom.wkt))
    lyr.CreateFeature(feat)

    mask_ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(gt)
    mask_ds.SetProjection(proj_wkt)

    gdal.RasterizeLayer(mask_ds, [1], lyr, burn_values=[1])

    mask = mask_ds.ReadAsArray().astype(bool)
    rows, cols = np.where(mask)

    rows = rows.astype(np.int32)
    cols = cols.astype(np.int32)

    if not include_lon_lat:
        return rows, cols

    lon = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
    lat = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]

    return rows, cols, lon.astype(dtype), lat.astype(dtype)