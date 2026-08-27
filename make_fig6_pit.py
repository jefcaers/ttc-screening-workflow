#!/usr/bin/env python3
"""Figure 6: PIT ECDF. Full-width, ~2x fonts."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from scipy.stats import norm, kstest

plt.rcParams.update({"font.family": "Liberation Sans"})
INK, GRAY, BLUE, GREEN = "#141414", "#8A8A8A", "#2F6DB5", "#1B8A5A"

def pits(src, cases_key, pool_all=True):
    g = {}
    with redirect_stdout(io.StringIO()):
        exec(open(src).read(), g)
    return g

# copper
gc = {}
with redirect_stdout(io.StringIO()):
    exec(open(_os.path.join(_HERE,"backtest_copper.py")).read(), gc)
CU, postC, fpC, PREC, Z = gc["CASES_CU"], gc["posterior"], gc["frozen_pool"], gc["PRE_CU"], gc["Z"]
pcu = np.array([norm.cdf((np.log(c["ratio"]) - postC(fpC(c, CU, PREC), "rc")[0]) / postC(fpC(c, CU, PREC), "rc")[1]) for c in CU])
# lithium
gl = {}
with redirect_stdout(io.StringIO()):
    exec(open(_os.path.join(_HERE,"lithium_engine","backtest_validation.py")).read(), gl)
LI, postL = gl["CASES"], gl["posterior"]
pli = np.array([norm.cdf((np.log(c["ratio"]) - postL(c, "rc")[0]) / postL(c, "rc")[1]) for c in LI])
pits_all = np.concatenate([pcu, pli]); ks = kstest(pits_all, "uniform")

fig, ax = plt.subplots(figsize=(11, 5.4))
xs = np.sort(pits_all)
ax.step(np.r_[0, xs, 1], np.r_[0, np.arange(1, len(xs)+1)/len(xs), 1], where="post", color=INK, lw=3,
        label=f"21 cases (KS p = {ks.pvalue:.2f}, mean {pits_all.mean():.2f})")
ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=2, label="uniform reference")
for v in pli: ax.plot([v, v], [0.0, 0.045], color=BLUE, lw=2.2)
for v in pcu: ax.plot([v, v], [0.055, 0.10], color=GREEN, lw=2.2)
ax.text(0.008, 0.022, "lithium", color=BLUE, fontsize=15, va="center", fontweight="bold")
ax.text(0.008, 0.078, "copper", color=GREEN, fontsize=15, va="center", fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("probability integral transform of realized ratio under frozen posterior", fontsize=17)
ax.set_ylabel("empirical CDF", fontsize=17)
ax.tick_params(labelsize=14)
ax.legend(frameon=False, loc="upper left", fontsize=16)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(_os.path.join(_HERE,"figures","rf02_pit.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("written rf02_pit.png  KS p=%.2f mean=%.2f" % (ks.pvalue, pits_all.mean()))
