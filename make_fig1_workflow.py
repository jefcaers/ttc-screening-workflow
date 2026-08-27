#!/usr/bin/env python3
"""Figure 1: the screening workflow, stretched vertically; decision engine as the bottom box."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9})
INK, GRAY, BLUE = "#141414", "#8A8A8A", "#2F6DB5"

fig, ax = plt.subplots(figsize=(6.5, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis("off")

def box(x, y, w, h, t, fc="white", ec=INK, fs=8, tc=None, bold=True):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color=tc or ec,
            fontweight="bold" if bold else "normal", linespacing=1.25)

def arr(x1, y1, x2, y2, c=INK, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=c, lw=lw, shrinkA=2, shrinkB=3))

# Row 1: data gathering (wide top box)
box(50, 62, 94, 6,
    "Data gathering: filings, reference classes, demand scenarios, policy events",
    fs=13, ec=GRAY, tc=INK, bold=False)

# Row 2: four inputs
ins_ = [("scoping\nestimate", 12), ("comparable completed\nprojects (reference cases)", 38),
        ("era demand\nscenarios", 64), ("policy-event panel,\nindicator block", 88)]
for t, x in ins_:
    box(x, 48, 24, 8.5, t, fc="#F4F6F9", ec=GRAY, tc=INK, fs=11.5, bold=False)

# Row 3: three engines
eng = [("cost engine\ncapital posterior", 16), ("market engine\nprice paths", 50),
       ("country-risk engine\nintervention · prediction", 84)]
for t, x in eng:
    box(x, 33, 25, 8.5, t, ec=BLUE, fs=12.5)

# Row 4: valuation engine (centered)
box(50, 19, 32, 8.5, "valuation engine\nMonte Carlo NPV distribution", ec=BLUE, fs=12.5)

# Row 5 (bottom): decision engine, wide to avoid overflow
box(50, 6, 50, 8.5, "decision engine\nRAV · V_stage  →  enter / stage / walk", ec=INK, fs=14)

# arrows: data gathering -> each input
for x in [12, 38, 64, 88]:
    arr(x, 58, x, 52.2, c=GRAY, lw=0.9)

# inputs -> engines
arr(12, 44, 15, 37.2)      # scoping -> cost
arr(38, 44, 19, 37.2)      # comparable -> cost
arr(64, 44, 51, 37.2)      # demand -> market
arr(88, 44, 85, 37.2)      # policy -> country-risk

# engines -> valuation / decision
arr(16, 29, 44, 23)        # cost -> valuation
arr(50, 29, 50, 23.2)      # market -> valuation (straight down)
arr(84, 29, 62, 9)         # country-risk -> decision (blue, skips valuation)
# recolor the country-risk arrow blue
ax.patches[-1].set_color(BLUE)
arr(50, 15, 50, 10.2)      # valuation -> decision (straight down)

plt.tight_layout(pad=0.3)
plt.savefig(_os.path.join(_HERE,"figures","rf14_workflow.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("written rf14_workflow.png")
