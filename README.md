## Citation

If you use this package in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19378061.svg)](https://doi.org/10.5281/zenodo.19378061)

## Environment Setup & package Installation 

(WSL required for Windows users due to GDAL/GRIB dependencies) 

1. Install Micromamba

curl -Ls https://micro.mamba.pm/install.sh | bash

source ~/.bashrc

micromamba --version

micromamba info

2. Create and activate environment

micromamba create -n mrms_usgs_events -c conda-forge python=3.11 gdal geopandas libgdal-grib -y

micromamba activate mrms_usgs_events

3. Install the package

python -m pip install --upgrade pip               # Upgrade pip

### Development mode (code modifications)

pip install -e .

### User mode (No code modifications)
python -m pip install mrms-usgs-events            # Install the package (Do NOT run if using development mode (-e))

4. Verify installation

gdalinfo --formats | grep -i grib                 # Verify gdal grib

mrms-usgs --help                                  # Verify installation

## Example use for the Texas July 4 2025 event at the Mystic Camp. (Use at least 1 year between start and end) 
#Use event downloader
mrms-usgs run-site 08165500 \
  --start 2023-07-01 \
  --end 2025-07-10 \
  --base-dir data \
  --overwrite
# Create input tsv
mrms-usgs masks build-input \
  --basins-dir "$BASE_DIR/basins_json" \
  --out "$MASK_INPUT" \
  --overwrite
# mrms-usgs masks build-state-masks \
mrms-usgs masks build-state-masks \
  --sample-grib-gz "$SAMPLE_GRIB" \
  --mask-input "$MASK_INPUT" \
  --out-dir "$STATE_MASK_DIR" \
  --overwrite
# 
## Data directory with subfolders created following this structure
data/

├── _mrms_cache/     # Temporary cache of downloaded MRMS .grib2 files to avoid re-downloading 

├── basins_json/     # Watershed boundaries (GeoJSON) for each USGS station (used to mask rainfall)

├── events/          # Detected hydrologic events (CSV files with peaks, volumes, and timing)

├── logs/            # Execution logs for debugging and tracking pipeline progress

├── rain_zarr/       # Processed rainfall data stored in Zarr format (spatial + temporal arrays)

├── site_meta/       # Metadata for each USGS station (location, name, timezone, etc.)

└── stage_parquet/   # Time series of water level (stage) data from USGS in Parquet format


## Acknowledgements

This material is based upon work supported by the U.S. National Science Foundation under Grant No. TI-2303756 and the Tethys Geoscience Foundation.
