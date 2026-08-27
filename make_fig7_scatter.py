#!/usr/bin/env python3
"""Figure 7: copper ex-ante RAV vs realized NPV scatter. Full-width, ~2x fonts, legend center-left."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from scipy.stats import spearmanr

plt.rcParams.update({"font.family": "Liberation Sans"})
INK, GRAY, BLUE, GREEN, RED, AMBER = "#141414", "#8A8A8A", "#2F6DB5", "#1B8A5A", "#C8102E", "#C98A1E"
colm = {"enter": GREEN, "stage": AMBER, "walk": RED}

g = {}
with redirect_stdout(io.StringIO()):
    exec(open(_os.path.join(_HERE,"backtest_copper.py")).read(), g)
CU, era, post, fp, npv, PRE, GAM, realized = (g["CASES_CU"], g["era_cu"], g["posterior"], g["frozen_pool"],
                                              g["npv_engine"], g["PRE_CU"], g["GAM_CU"], g["realized_npv"])
rng = np.random.default_rng(7); disc = 0.08
def run(c, N=2000):
    p0, scen, sup = era(c["yr"]); mu, s = post(fp(c, CU, PRE))
    draws = c["est"]*np.exp(mu + s*rng.standard_normal(N)); tot = sum(w for _, w in scen)
    o = np.empty(N)
    for i in range(N):
        r = rng.random()*tot; gg = scen[0][0]
        for cg, w in scen:
            r -= w
            if r <= 0: gg = cg; break
        pp = lambda t, gg=gg: min(max(p0*((1+gg)/(1+sup))**(GAM*t), 0.5*p0), 2.5*p0)
        c2 = dict(c); c2["cc"] = c["cc"]*np.exp(0.15*rng.standard_normal())
        o[i] = npv(c2, draws[i], pp, disc=disc)
    m = o.mean(); cv = np.sort(o)[:max(1, N//20)].mean(); rav = m - 0.5*(m - cv)
    return rav, ("walk" if rav <= 0 and m <= 0 else ("stage" if rav <= 0 else "enter"))
res = [(c,) + run(c) + (realized(c, disc=disc),) for c in CU]
ravs = np.array([r[1] for r in res]); rns = np.array([r[3] for r in res]); rho, pv = spearmanr(ravs, rns)

fig, ax = plt.subplots(figsize=(11, 6.2))
for c, rav, v, rn in res:
    ax.plot(rav/1000, rn/1000, "o", color=colm[v], ms=13, mec=INK, mew=.7)
    if "Sierra" in c["name"] or "Quebrada" in c["name"]:
        ax.annotate(("Sierra Gorda" if "Sierra" in c["name"] else "QB2") + " (Sumitomo entered; engine: walk)",
                    (rav/1000, rn/1000), textcoords="offset points", xytext=(12, -6), fontsize=14, color=RED)
ax.axhline(0, color=GRAY, lw=1); ax.axvline(0, color=GRAY, lw=1)
ax.set_xlabel("ex-ante RAV on frozen sanction-date information (US$B, 8% discount)", fontsize=17)
ax.set_ylabel("realized NPV, reconstruction (US$B)", fontsize=17)
ax.tick_params(labelsize=14)
ax.text(.98, .04, f"Spearman ρ = {rho:.2f} (p = {pv:.2f});\n0.57 to 0.80 across the 9-cell grid",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=15, linespacing=1.3)
for v in ["enter", "stage", "walk"]:
    ax.plot([], [], "o", color=colm[v], mec=INK, mew=.7, ms=13, label="choice: " + v)
ax.legend(frameon=False, loc="center left", fontsize=16, handletextpad=0.4)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(_os.path.join(_HERE,"figures","rf04_copper_scatter.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("written rf04_copper_scatter.png  rho=%.2f p=%.2f" % (rho, pv))
