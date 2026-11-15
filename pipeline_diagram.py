"""Create a simple project pipeline diagram."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "pipeline_diagram.png"
OUT.parent.mkdir(exist_ok=True)

steps = [
    "NASA C-MAPSS\nFD001 dataset",
    "Preprocessing\n14 sensors, RUL cap 125",
    "30-cycle\nsequence windows",
    "LSTM model\n64 hidden units",
    "Evaluation\nRMSE, MAE, NASA score",
]

fig, ax = plt.subplots(figsize=(14, 3.4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3.4)
ax.axis("off")

xs = [0.35, 3.15, 5.95, 8.75, 11.55]
for i, (x, label) in enumerate(zip(xs, steps)):
    box = FancyBboxPatch(
        (x, 1.05), 2.1, 1.25,
        boxstyle="round,pad=0.12,rounding_size=0.12",
        linewidth=1.5, edgecolor="#34495e", facecolor="#eaf2f8"
    )
    ax.add_patch(box)
    ax.text(x + 1.05, 1.675, label, ha="center", va="center", fontsize=9)
    if i < len(xs) - 1:
        arrow = FancyArrowPatch(
            (x + 2.18, 1.675), (xs[i + 1] - 0.12, 1.675),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#34495e"
        )
        ax.add_patch(arrow)

ax.text(7, 2.85, "Turbofan Remaining Useful Life Prediction Pipeline",
        ha="center", va="center", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved: {OUT}")
