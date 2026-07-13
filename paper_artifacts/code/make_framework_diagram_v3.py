from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path("paper_artifacts")
FIGURES = ROOT / "figures" / "v2"
FIGURES.mkdir(parents=True, exist_ok=True)

def add_box(ax, x, y, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x + w / 2, y + h / 2)

def arrow(ax, start, end, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            lw=1.7,
            connectionstyle=f"arc3,rad={rad}"
        )
    )

fig, ax = plt.subplots(figsize=(12.5, 6.5))

# Main pipeline boxes
normal = add_box(ax, 0.45, 4.35, 1.8, 0.7, "Normal HID\nbehavior data")
seed = add_box(ax, 0.45, 3.15, 1.8, 0.7, "Initial attack\nseed data")

train = add_box(ax, 3.0, 3.75, 1.9, 0.8, "Train detector\nDᵣ")
generate = add_box(ax, 5.7, 3.75, 2.2, 0.8, "Detector-guided\ncandidate generation")
oracle = add_box(ax, 8.7, 3.75, 1.9, 0.8, "Oracle\nevaluation")

decision = add_box(ax, 8.7, 2.35, 1.9, 0.75, "Bypass\nfound?")
hardneg = add_box(ax, 5.7, 1.55, 2.2, 0.8, "Hard-negative\nset")
harden = add_box(ax, 3.0, 1.55, 1.9, 0.8, "Retrain / harden\nDᵣ₊₁")
report = add_box(ax, 10.85, 1.55, 1.7, 0.8, "No bypass\n→ report")

# Data to train
arrow(ax, (2.25, 4.70), (3.0, 4.25))
arrow(ax, (2.25, 3.50), (3.0, 4.05))

# Main flow
arrow(ax, (4.9, 4.15), (5.7, 4.15))
arrow(ax, (7.9, 4.15), (8.7, 4.15))
arrow(ax, (9.65, 3.75), (9.65, 3.10))

# Yes path
arrow(ax, (8.7, 2.70), (7.9, 1.95))
arrow(ax, (5.7, 1.95), (4.9, 1.95))

# Hardened detector loop back
arrow(ax, (3.95, 2.35), (6.0, 3.75), rad=-0.25)

# No path
arrow(ax, (10.6, 2.70), (10.85, 1.95))

# Labels
ax.text(8.05, 2.55, "yes", fontsize=10, ha="center")
ax.text(10.75, 2.55, "no", fontsize=10, ha="center")

ax.text(
    0.45, 5.55,
    "Detector-guided hard-negative hardening framework",
    fontsize=16,
    ha="left"
)

ax.text(
    0.45, 0.65,
    "The loop continues while detector-guided generation discovers bypass candidates under the configured search budget.",
    fontsize=10,
    ha="left"
)

ax.set_xlim(0, 13)
ax.set_ylim(0.35, 5.95)
ax.axis("off")

out = FIGURES / "framework_loop_diagram_v3.png"
plt.tight_layout()
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"[saved] {out}")
