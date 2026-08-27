#!/usr/bin/env python3
"""Figure 2: Olaroz cost blend. Fonts ~2x; legend and labels de-conflicted."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Liberation Sans"})
INK, GRAY, LB, BLUE, RED = "#141414", "#8A8A8A", "#9BB8D3", "#2F6DB5", "#C8102E"

muP = np.log(1.31); sP = 0.318; s1 = 0.45
x = np.linspace(0.4, 2.9, 600)
post = np.exp(-0.5*((np.log(x)-muP)/sP)**2)/(x*sP)
ins = np.exp(-0.5*((np.log(x))/s1)**2)/(x*s1)

fig, ax = plt.subplots(figsize=(9.6, 4.7))
ax.plot(x, ins/ins.max(), color=GRAY, ls="--", lw=3.0, label="inside view (estimate-centered, AACE class 5)")
ax.plot(x, post/post.max(), color=BLUE, lw=4.2, label="reference-class × inside posterior")

# percentile markers, labels alternated in height so they never touch
for v, lab, c, yl in [(0.87, "P10 ×0.87", LB, 1.10), (1.31, "P50 ×1.31", INK, 1.19),
                      (1.97, "P90 ×1.97", RED, 1.10)]:
    ax.axvline(v, color=c, ls=":", lw=2.2)
    ax.text(v, yl, lab, ha="center", fontsize=21, color=c, fontweight="bold")

# realized build marker, label clear of the curve to its right
ax.axvline(1.75, color=GRAY, lw=2.2)
ax.text(1.79, 0.42, "built ×1.75", rotation=90, ha="left", va="center", fontsize=18, color=INK)

ax.set_xlabel("realized ÷ sanctioned capital cost", fontsize=22)
ax.tick_params(axis="x", labelsize=18)
ax.set_yticks([]); ax.set_ylim(0, 1.30); ax.set_xlim(0.4, 2.9)
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)

# legend above the plot, clear of every in-axes label
ax.legend(frameon=False, fontsize=19, loc="lower center",
          bbox_to_anchor=(0.5, 1.14), ncol=1, handlelength=2.6)

plt.tight_layout()
plt.savefig(_os.path.join(_HERE,"figures","rf13_olaroz_posterior.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("written rf13_olaroz_posterior.png")
