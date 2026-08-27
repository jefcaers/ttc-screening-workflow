"""Three-rung ablation ladder, lithium cohort (spec: ablation_spec.md D, logged before run).
Reuses the staged lithium engine unchanged; audited configuration disc=10%, lam=0.5,
gamma=1.2, seed 11; loss grid exactly as loss_sensitivity.py (es x chi, 20 settings)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, numpy as np
from contextlib import redirect_stdout
from scipy.stats import spearmanr

src = open(_os.path.join(_HERE,"lithium_engine","backtest_validation.py")).read()
g = {}
with redirect_stdout(io.StringIO()):
    exec(src, g)
CASES, era, posterior, npv_engine, realized_npv = g["CASES"], g["era"], g["posterior"], g["npv_engine"], g["realized_npv"]

rng = np.random.default_rng(11)
rns = np.array([realized_npv(c) for c in CASES])
GAM = 1.2

def clamppath(p0, gg, sup):
    return lambda t: min(max(p0*((1+gg)/(1+sup))**(GAM*t), 0.4*p0), 4.0*p0)

def rung1(case):
    p0, scen, sup = era(case["yr"])
    mu, s = posterior(case, "rc")
    capex = case["est"]*np.exp(mu)
    gg = max(scen, key=lambda x: x[1])[0]
    npv = npv_engine(dict(case), capex, clamppath(p0, gg, sup))
    return npv, ("enter" if npv > 0 else "walk")

def rung23(case, N=2000):
    p0, scen, sup = era(case["yr"])
    mu, s = posterior(case, "rc")
    draws = case["est"]*np.exp(mu + s*rng.standard_normal(N))
    tot = sum(w for _, w in scen)
    npvs = np.empty(N)
    for i in range(N):
        r = rng.random()*tot; gg = scen[0][0]
        for cg, w in scen:
            r -= w
            if r <= 0: gg = cg; break
        c2 = dict(case); c2["cc"] = case["cc"]*np.exp(0.18*rng.standard_normal())
        npvs[i] = npv_engine(c2, draws[i], clamppath(p0, gg, sup))
    mu_n = npvs.mean(); cvar = np.sort(npvs)[:max(1, N//20)].mean()
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

ES = [0.1, 0.2, 0.3, 0.4, 0.5]; CHI = [0.25, 0.5, 0.75, 1.0]
def gridshare(vlist):
    shares = []; beat = 0
    for es in ES:
        for chi in CHI:
            Ls = loss(vlist, es, chi); Le = loss(["enter"]*10, es, chi)
            Lw = loss(["walk"]*10, es, chi)
            Lo = loss(["enter" if x > 0 else "walk" for x in rns], es, chi)
            Lr = np.mean([loss(rng.choice(["enter", "stage", "walk"], 10), es, chi) for _ in range(1000)])
            shares.append(100*(Le-Ls)/max(1e-9, Le-Lo))
            if Ls < min(Le, Lw, Lr): beat += 1
    ctr = shares[ES.index(0.3)*len(CHI)+CHI.index(0.5)]
    return min(shares), max(shares), ctr, beat, len(shares)

names = [c["name"].split(" (")[0] for c in CASES]
print("="*104)
print("ABLATION LADDER, lithium cohort, disc=10%, audited configuration (spec: ablation_spec.md D)")
print("="*104)
print(f"{'case':<26} {'truth':>6} {'realized':>9} | {'R1 detNPV':>9} {'v1':>5} | {'R2 E[NPV]':>9} {'v2':>5} | {'R3 RAV':>8} {'v3':>5}")
for i, n in enumerate(names):
    print(f"{n:<26} {CASES[i]['truth']:>6} {rns[i]:>9.0f} | {stat1[i]:>9.0f} {v1[i]:>5} | {stat2[i]:>9.0f} {v2[i]:>5} | {stat3[i]:>8.0f} {v3[i]:>5}")

for tag, stat, v in [("R1 corrected-point deterministic", stat1, v1),
                     ("R2 stochastic RC, expected value", stat2, v2),
                     ("R3 full workflow (RAV 0.5)", stat3, v3)]:
    rho, pv = spearmanr(stat, rns)
    mn, mx, ctr, beat, ncell = gridshare(v)
    mix = (v.count("enter"), v.count("stage"), v.count("walk"))
    el = [names[i] for i in range(10) if v[i] == "enter" and rns[i] < 0]
    ww = [names[i] for i in range(10) if v[i] == "walk" and rns[i] > 0]
    print(f"\n{tag}")
    print(f"  choice mix e/s/w = {mix} | entered losers: {el} | walked winners: {ww}")
    print(f"  Spearman(statistic, realized) = {rho:.2f} (p={pv:.2f})")
    print(f"  captured share: center {ctr:.0f}%, range {mn:.0f}% to {mx:.0f}%, beats all baselines in {beat}/{ncell} settings")
