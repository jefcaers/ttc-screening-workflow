#!/usr/bin/env python3
"""Figure 4: structural causal graph of the country-risk engine."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9})

BLUE, GREEN, RED, AMBER, INK, MUT = "#2F6DB5", "#1B8A5A", "#C8102E", "#C98A1E", "#141414", "#666666"

fig, ax = plt.subplots(figsize=(10.5, 6.6))
ax.set_xlim(0, 10.9); ax.set_ylim(-0.2, 7.1); ax.axis("off")

def box(x, y, w, h, text, fc, ec, dashed=False, fs=13, tc="white", lw=1.7):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.06",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle="--" if dashed else "-", zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)

def arrow(x1, y1, x2, y2, color=MUT, lw=1.3, style="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                        color=color, linewidth=lw, linestyle=style, zorder=2,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=8, shrinkB=8)
    ax.add_patch(a)

# latent regime chain (top left)
box(1.3, 5.9, 1.5, 0.62, "R(t−1)", "white", AMBER, dashed=True, tc=INK)
box(3.2, 5.9, 1.5, 0.62, "regime R(t)", "white", AMBER, dashed=True, tc=INK)
arrow(2.05, 5.9, 2.45, 5.9, color=AMBER)

# indicator block (observed, below regime)
box(1.55, 4.55, 2.9, 1.0, "indicator block (observed)\nfiscal stress · FX reserves · inflation\nelections · rhetoric · coalition", BLUE, BLUE, fs=7.6)
arrow(2.9, 5.55, 2.1, 5.1, color=AMBER)

# world price (top right)
box(8.4, 5.9, 2.0, 0.62, "world price P(t)", BLUE, BLUE)

# levers (middle row)
levers = [("export\nrestrictions E", 2.2), ("nationalization\npressure N", 4.1),
          ("local processing\nmandate L", 6.0), ("royalty / tax\nchange T", 7.9)]
for txt, x in levers:
    box(x, 3.1, 1.78, 0.95, txt, "white", RED, tc=INK, fs=12)
    arrow(3.2, 5.56, x, 3.58, color=AMBER, rad=-0.08)      # regime -> lever
    arrow(8.4, 5.56, x, 3.58, color=BLUE, rad=0.10)        # log P -> activation

# NPV node (bottom)
box(5.0, 1.15, 2.4, 0.8, "project NPV", GREEN, GREEN, fs=15)
chan = {"E": "price capture", "N": "discount rate, exit value", "L": "capital scope", "T": "fiscal take"}
for (txt, x), key in zip(levers, ["E", "N", "L", "T"]):
    arrow(x, 2.65, 5.0, 1.55, color=RED, rad=0.05)
# price to revenue
arrow(9.35, 5.6, 6.15, 1.3, color=BLUE, rad=-0.38)
ax.text(9.9, 3.3, "revenue", fontsize=11, color=BLUE, rotation=-72)

ax.text(3.2, 6.55, "latent, yearly transitions", fontsize=12, color=AMBER, ha="center")
ax.text(5.35, 0.42, "one NPV channel per lever:  E → price capture · N → discount rate and exit · L → capital scope · T → fiscal take",
        fontsize=10.5, color=MUT, ha="center")
ax.text(5.35, 0.02, "activation:  P(lever activates) = Φ(αR + β log P + c)", fontsize=13, color=INK, ha="center")

plt.tight_layout(pad=0.4)
plt.savefig(_os.path.join(_HERE,"figures","fig4_causal_graph.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("written fig4_causal_graph.png")
