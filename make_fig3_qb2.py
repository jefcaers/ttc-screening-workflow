#!/usr/bin/env python3
"""Figure 3: QB2 decision-statistic distribution. Full-width, ~2x fonts."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

plt.rcParams.update({"font.family": "Liberation Sans"})
INK, LB, BLUE, RED = "#141414", "#9BB8D3", "#2F6DB5", "#C8102E"

g = {}
with redirect_stdout(io.StringIO()):
    exec(open(_os.path.join(_HERE,"backtest_copper.py")).read(), g)
CU, era, post, fp, npv, PRE, GAM = (g["CASES_CU"], g["era_cu"], g["posterior"],
                                    g["frozen_pool"], g["npv_engine"], g["PRE_CU"], g["GAM_CU"])
qb = [c for c in CU if "Quebrada" in c["name"]][0]
rng = np.random.default_rng(7); N = 4000; disc = 0.08
p0, scen, sup = era(qb["yr"]); mu, s = post(fp(qb, CU, PRE))
draws = qb["est"]*np.exp(mu + s*rng.standard_normal(N)); tot = sum(w for _, w in scen)
npvs = np.empty(N)
for i in range(N):
    r = rng.random()*tot; gg = scen[0][0]
    for cg, w in scen:
        r -= w
        if r <= 0: gg = cg; break
    pp = lambda t, gg=gg: min(max(p0*((1+gg)/(1+sup))**(GAM*t), 0.5*p0), 2.5*p0)
    c2 = dict(qb); c2["cc"] = qb["cc"]*np.exp(0.15*rng.standard_normal())
    npvs[i] = npv(c2, draws[i], pp, disc=disc)
srt = np.sort(npvs); cv = srt[:len(srt)//20].mean(); m = npvs.mean(); rav = m - 0.5*(m - cv)

fig, ax = plt.subplots(figsize=(11, 4.7))
ax.hist(npvs/1000, bins=44, color=LB, edgecolor="white")
ax.hist(srt[:len(srt)//20]/1000, bins=8, color=RED, edgecolor="white")
ymax = ax.get_ylim()[1]; ax.set_ylim(0, ymax*1.16)
for val, lab in [(cv, "CVaR₅"), (rav, "RAV"), (m, "E[NPV]")]:
    c = {"CVaR₅": RED, "RAV": INK, "E[NPV]": BLUE}[lab]
    ax.axvline(val/1000, color=c, lw=2.6, ls="--")
    ax.text(val/1000, ymax*1.11, f"{lab} {val/1000:+.1f}", color=c, fontsize=21,
            fontweight="bold", ha="center", va="bottom")
ax.set_xlabel("NPV, Quebrada Blanca 2 at 2018 sanction (US$B, 8% discount, frozen information)", fontsize=21)
ax.tick_params(axis="x", labelsize=18)
ax.set_yticks([])
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(_os.path.join(_HERE,"figures","rf12_qb2_dist.png"), dpi=300, bbox_inches="tight", facecolor="white")
print("written rf12_qb2_dist.png  RAV=%.2f CVaR=%.2f E=%.2f" % (rav/1000, cv/1000, m/1000))
