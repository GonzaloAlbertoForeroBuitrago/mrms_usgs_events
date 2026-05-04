from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FP = Path("/data/repository_code/unified_data/research/hydrologic_drivers/pixel_response/outputs/08165500_pixel_response_summary.parquet")
OUT = FP.with_suffix(".map.png")

df = pd.read_parquet(FP)

colors = {
    "HIGH_FAST_RESPONSE": "red",
    "MEDIUM_FAST_RESPONSE": "orange",
    "SLOW_RESPONSE": "blue",
    "LOW_RESPONSE": "purple",
    "NO_RESPONSE": "lightgray",
    "NO_SIGNAL": "lightgray",
}

df["color"] = df["response_class"].map(colors).fillna("black")

score = df["hydrologic_influence_score"].fillna(0)
if score.max() > 0:
    size = 20 + 500 * (score / score.max())
else:
    size = 20

plt.figure(figsize=(11, 9))

plt.scatter(
    df["lon"],
    df["lat"],
    c=df["color"],
    s=size,
    alpha=0.75,
    edgecolors="black",
    linewidths=0.2,
)

for cls, color in colors.items():
    sub = df[df["response_class"] == cls]
    if len(sub):
        plt.scatter([], [], c=color, s=80, label=f"{cls} ({len(sub)})")

plt.xlabel("Longitude", fontsize=13)
plt.ylabel("Latitude", fontsize=13)
plt.title("08165500 Pixel Hydrologic Response\nColor = response class, Size = influence score", fontsize=15)
plt.legend(loc="best", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=250)
print(f"Saved: {OUT}")