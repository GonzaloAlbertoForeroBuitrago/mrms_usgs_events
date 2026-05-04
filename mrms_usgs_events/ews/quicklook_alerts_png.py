from pathlib import Path
import json

import geopandas as gpd
import matplotlib.pyplot as plt


BASE = Path("/data/repository_code/unified_data/ews_tethys/TEXAS")

basin_fp = BASE / "basin_alerts.geojson"
pixel_fp = BASE / "pixel_alerts.geojson"

print("loading basins...")
basins = gpd.read_file(basin_fp)

print("loading pixels...")
pixels = gpd.read_file(pixel_fp)

print("basins:", len(basins))
print("pixels:", len(pixels))

# -------------------------------------------------------------------
# STATEWIDE BASIN ALERT MAP
# -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 14))

order = ["NORMAL", "WATCH", "WARNING", "SEVERE"]

for level in order:
    sub = basins[basins["alert_level"] == level]

    if len(sub) == 0:
        continue

    color = sub["fill_color"].iloc[0]

    sub.plot(
        ax=ax,
        color=color,
        linewidth=0.25,
        edgecolor="black",
        alpha=0.9,
    )

ax.set_title(
    "Texas Basin Alerts\n2025-07-04 Operational Replay",
    fontsize=20,
    weight="bold",
)

ax.set_axis_off()

png1 = BASE / "texas_basin_alerts.png"

plt.savefig(
    png1,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("saved:", png1)

# -------------------------------------------------------------------
# ZOOM BASIN PIXELS
# -------------------------------------------------------------------

SITE = "08165500"

basin_one = basins[basins["site_id"] == SITE]
pixel_one = pixels[pixels["site_id"] == SITE]

print("site pixels:", len(pixel_one))

fig, ax = plt.subplots(figsize=(14, 14))

basin_one.boundary.plot(
    ax=ax,
    linewidth=1.5,
    edgecolor="black",
)

pixel_one.plot(
    ax=ax,
    column="estimated_delta_water_stage",
    legend=True,
    alpha=0.8,
)

ax.set_title(
    f"Pixel Alerts - {SITE}",
    fontsize=18,
    weight="bold",
)

ax.set_axis_off()

png2 = BASE / f"{SITE}_pixel_alerts.png"

plt.savefig(
    png2,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("saved:", png2)