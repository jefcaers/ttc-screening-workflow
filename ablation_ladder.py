"""Three-rung ablation ladder, copper cohort. Pre-specified in ablation_spec.md (A).
Reuses backtest_copper.py unchanged; all rungs at disc=8%, realized NPV at 8%."""
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

DISC = 0.08
rng = np.random.default_rng(7)
rns = np.array([realized_npv(c, disc=DISC) for c in CASES])

def clamppath(p0, gg, sup):
    return lambda t: min(max(p0*((1+gg)/(1+sup))**(GAM*t), 0.5*p0), 2.5*p0)

# ---- R1: corrected point + deterministic single-deck + threshold ----
def rung1(case):
    p0, scen, sup = era_cu(case["yr"])
    mu, s = posterior(frozen_pool(case, CASES, PRE_CU))
    capex = case["est"]*np.exp(mu)                       # corrected point (posterior median)
    gg = max(scen, key=lambda x: x[1])[0]                # highest-weight scenario only
    npv = npv_engine(dict(case), capex, clamppath(p0, gg, sup), disc=DISC)
    return npv, ("enter" if npv > 0 else "walk")

# ---- R2/R3 share one MC (common random numbers) ----
def mc(case, N=2000):
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
        c2 = dict(case); c2["cc"] = case["cc"]*np.exp(0.15*rng.standard_normal())
        npvs[i] = npv_engine(c2, draws[i], clamppath(p0, gg, sup), disc=DISC)
    return npvs

def rung23(case):
    npvs = mc(case)
    mu_n = npvs.mean(); cvar = np.sort(npvs)[:max(1, len(npvs)//20)].mean()
    rav = mu_n - 0.5*(mu_n - cvar)
    v2 = "enter" if mu_n > 0 else "walk"
    v3 = "walk" if (rav <= 0 and mu_n <= 0) else ("stage" if rav <= 0 else "enter")
    return mu_n, rav, v2, v3

R1 = [rung1(c) for c in CASES]
R23 = [rung23(c) for c in CASES]
stat1 = np.array([x[0] for x in R1]); v1 = [x[1] for x in R1]
stat2 = np.array([x[0] for x in R23]); v2 = [x[2] for x in R23]
stat3 = np.array([x[1] for x in R23]); v3 = [x[3] for x in R23]

def loss(vlist, es, chi):
    E = {"enter": 1.0, "stage": es, "walk": 0.0}
    return sum(E[v]*max(0, -n) + (1-E[v])*chi*max(0, n) for v, n in zip(vlist, rns))

ES = [0.1, 0.3, 0.5]; CHI = [0.25, 0.5, 0.75, 1.0]
def gridshare(vlist):
    shares = []
    beats_all = True
    for es in ES:
        for chi in CHI:
            Ls = loss(vlist, es, chi); Le = loss(["enter"]*11, es, chi)
            Lw = loss(["walk"]*11, es, chi)
            Lo = loss(["enter" if x > 0 else "walk" for x in rns], es, chi)
            Lr = np.mean([loss(rng.choice(["enter", "stage", "walk"], 11), es, chi) for _ in range(1000)])
            shares.append(100*(Le-Ls)/max(1e-9, Le-Lo))
            if not (Ls < min(Le, Lw, Lr)): beats_all = False
    return min(shares), max(shares), shares[ES.index(0.3)*len(CHI)+CHI.index(0.5)], beats_all

names = [c["name"].split(" (")[0] for c in CASES]
ix_sg = names.index("Sierra Gorda"); ix_qb = [i for i, n in enumerate(names) if "Quebrada" in n][0]

print("="*100)
print("ABLATION LADDER, copper cohort, disc=8%, realized NPV @8% (spec: ablation_spec.md A)")
print("="*100)
print(f"{'case':<26} {'realized':>9} | {'R1 detNPV':>9} {'v1':>5} | {'R2 E[NPV]':>9} {'v2':>5} | {'R3 RAV':>8} {'v3':>5}")
for i, n in enumerate(names):
    print(f"{n:<26} {rns[i]:>9.0f} | {stat1[i]:>9.0f} {v1[i]:>5} | {stat2[i]:>9.0f} {v2[i]:>5} | {stat3[i]:>8.0f} {v3[i]:>5}")

for tag, stat, v in [("R1 corrected-point deterministic", stat1, v1),
                     ("R2 stochastic RC, expected value", stat2, v2),
                     ("R3 full workflow (RAV 0.5)", stat3, v3)]:
    rho, pv = spearmanr(stat, rns)
    mn, mx, ctr, beats = gridshare(v)
    mix = (v.count("enter"), v.count("stage"), v.count("walk"))
    print(f"\n{tag}")
    print(f"  choice mix e/s/w = {mix} | Sierra Gorda: {v[ix_sg]} | QB2: {v[ix_qb]}")
    print(f"  Spearman(statistic, realized) = {rho:.2f} (p={pv:.2f})")
    print(f"  captured share of enter-to-oracle gap: center cell {ctr:.0f}%, grid range {mn:.0f}% to {mx:.0f}%, beats all baselines in every cell: {beats}")

# entered losers / walked winners per rung
for tag, v in [("R1", v1), ("R2", v2), ("R3", v3)]:
    el = [names[i] for i in range(11) if v[i] == "enter" and rns[i] < 0]
    ww = [names[i] for i in range(11) if v[i] == "walk" and rns[i] > 0]
    print(f"{tag}: entered losers = {el} | walked winners = {ww}")
