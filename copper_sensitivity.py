"""Copper cohort: verdict sensitivity to discount rate, risk aversion, and loss weights.
Addresses the caveat that harmonized 10%-discount assumptions make the copper verdict
distribution degenerate (10/11 walk). Reuses the cohort engine unchanged."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, numpy as np
from contextlib import redirect_stdout
from scipy.stats import spearmanr

src = open(_os.path.join(_HERE,"backtest_copper.py")).read()
g = {}
with redirect_stdout(io.StringIO()):
    exec(src, g)
CASES, era_cu, posterior, frozen_pool = g["CASES_CU"], g["era_cu"], g["posterior"], g["frozen_pool"]
npv_engine, realized_npv, PRE_CU, GAM = g["npv_engine"], g["realized_npv"], g["PRE_CU"], g["GAM_CU"]
rng = np.random.default_rng(7)

def system_run(case, disc, lam, N=2000):
    p0, scen, sup = era_cu(case["yr"])
    mu, s = posterior(frozen_pool(case, CASES, PRE_CU))
    draws = case["est"]*np.exp(mu + s*rng.standard_normal(N))
    tot = sum(w for _, w in scen)
    npvs = np.empty(N)
    for i in range(N):
        r = rng.random()*tot; gg = scen[0][0]
        for cg, w in scen:
            r -= w
            if r <= 0: gg = cg; break
        pp = lambda t, gg=gg: min(max(p0*((1+gg)/(1+sup))**(GAM*t), 0.5*p0), 2.5*p0)
        c2 = dict(case); c2["cc"] = case["cc"]*np.exp(0.15*rng.standard_normal())
        npvs[i] = npv_engine(c2, draws[i], pp, disc=disc)
    mu_n = npvs.mean(); srt = np.sort(npvs); cvar = srt[:max(1, N//20)].mean()
    rav = mu_n - lam*(mu_n - cvar)
    verdict = "walk" if rav <= 0 and mu_n <= 0 else ("stage" if rav <= 0 else "enter")
    return rav, verdict

print("="*96)
print("GRID A: verdict mix, Sumitomo calls, and RAV ranking across discount rate x risk aversion")
print("="*96)
print("%-14s %-5s | enter/stage/walk | SierraGorda  QB2   | Spearman(RAV, realizedNPV@disc)" % ("discount", "lam"))
for disc in [0.08, 0.09, 0.10]:
    rns = np.array([realized_npv(c, disc=disc) for c in CASES])
    for lam in [0.3, 0.5, 0.7]:
        out = [system_run(c, disc, lam) for c in CASES]
        ravs = np.array([o[0] for o in out]); vs = [o[1] for o in out]
        sg = vs[[i for i, c in enumerate(CASES) if "Sierra" in c["name"]][0]]
        qb = vs[[i for i, c in enumerate(CASES) if "Quebrada" in c["name"]][0]]
        rho, pv = spearmanr(ravs, rns)
        print("%-14s %-5s |   %d / %d / %d     | %-11s %-5s | rho=%.2f (p=%.2f)" % (
            "%.0f%%" % (100*disc), "%.1f" % lam,
            vs.count("enter"), vs.count("stage"), vs.count("walk"), sg, qb, rho, pv))

print()
print("="*96)
print("GRID B: cost-weighted verdict loss at disc=8%%, lam=0.5 (copper-calibrated configuration)")
print("="*96)
disc = 0.08
rns = np.array([realized_npv(c, disc=disc) for c in CASES])
verd = [system_run(c, disc, 0.5)[1] for c in CASES]
def loss(vlist, es, chi):
    E = {"enter": 1.0, "stage": es, "walk": 0.0}; L = 0.0
    for v, npv in zip(vlist, rns):
        e = E[v]; L += e*max(0, -npv) + (1-e)*chi*max(0, npv)
    return L
print("verdicts@8%%/0.5:", list(zip([c["name"].split(" (")[0] for c in CASES], verd)))
hdr = "stage exp \\ chi |" + "".join("   %.2f          " % c for c in [0.25, 0.5, 0.75, 1.0])
print(hdr); print("-"*len(hdr))
worst, best = 100, -100
for es in [0.1, 0.3, 0.5]:
    cells = []
    for chi in [0.25, 0.5, 0.75, 1.0]:
        Ls = loss(verd, es, chi); Le = loss(["enter"]*11, es, chi); Lw = loss(["walk"]*11, es, chi)
        Lo = loss(["enter" if x > 0 else "walk" for x in rns], es, chi)
        Lr = np.mean([loss(rng.choice(["enter", "stage", "walk"], 11), es, chi) for _ in range(1000)])
        pct = 100*(Le-Ls)/max(1e-9, Le-Lo)
        beats = Ls < min(Le, Lw, Lr)
        worst = min(worst, pct); best = max(best, pct)
        cells.append("%6.0f|%3.0f%%|%s" % (Ls, pct, "Y" if beats else "N"))
    print("   %.1f          |" % es + "  ".join(cells))
print("captured-improvement range across grid: %.0f%% to %.0f%%" % (worst, best))
