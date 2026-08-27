"""Full metric suite over the 10-case lithium back-test (layers 1-4)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, sys, itertools, numpy as np
from contextlib import redirect_stdout
from scipy.stats import norm, kstest, spearmanr

# load the study (suppresses its prints); brings CASES, rows, posterior, system_run, era, etc.
src = open(_os.path.join(_HERE,"backtest_validation.py")).read()
buf = io.StringIO()
g = {}
with redirect_stdout(buf):
    exec(src, g)
CASES, rows, posterior, system_run = g["CASES"], g["rows"], g["posterior"], g["system_run"]
Z = 1.2816
rng = np.random.default_rng(23)

print("="*78); print("LAYER 1 — COST MODULE (probabilistic forecast quality)"); print("="*78)
# per-case PIT and CRPS (on log-ratio scale) for four forecasters
def crps_norm(x, mu, s):
    z = (x-mu)/s
    return s*(z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))
pits, tab = [], []
crps = {"RC":[], "inside":[], "flat10":[], "rule1.4":[]}
widths = {"RC":[], "inside":[]}
for r in rows:
    c = r["c"]; x = np.log(c["ratio"])
    muR,sR = posterior(c,"rc"); muI,sI = posterior(c,"inside")
    pits.append(norm.cdf((x-muR)/sR))
    crps["RC"].append(crps_norm(x,muR,sR)); crps["inside"].append(crps_norm(x,muI,sI))
    crps["flat10"].append(crps_norm(x,0.0,0.058))          # the original ±10% perturbation
    crps["rule1.4"].append(abs(x-np.log(1.4)))             # degenerate point rule
    widths["RC"].append(np.exp(muR+Z*sR)-np.exp(muR-Z*sR))
    widths["inside"].append(np.exp(muI+Z*sI)-np.exp(muI-Z*sI))
ks = kstest(pits, "uniform")
print("PIT values: " + " ".join("%.2f"%p for p in pits))
print("PIT uniformity KS: stat=%.2f p=%.2f  (p>0.10 = no detectable miscalibration at this n)" % (ks.statistic, ks.pvalue))
print("mean PIT = %.2f  (0.5 = unbiased; <0.5 = posterior still too optimistic)" % np.mean(pits))
print("mean CRPS (log-ratio, lower=better):")
for k in crps: print("   %-8s %.3f" % (k, np.mean(crps[k])))
print("skill vs inside-only: %.0f%%   vs flat ±10%%: %.0f%%   vs ×1.4 rule: %.0f%%" % (
  100*(1-np.mean(crps["RC"])/np.mean(crps["inside"])),
  100*(1-np.mean(crps["RC"])/np.mean(crps["flat10"])),
  100*(1-np.mean(crps["RC"])/np.mean(crps["rule1.4"]))))
print("sharpness (mean 80%% band width, ratio units): RC %.2f vs inside %.2f" % (np.mean(widths["RC"]), np.mean(widths["inside"])))
print("coverage 80%% band: RC %d/10 (target 8)   inside %d/10" % (sum(r["covR"] for r in rows), sum(r["covI"] for r in rows)))

print(); print("="*78); print("LAYER 2 — DECISION ENGINE"); print("="*78)
rns = np.array([r["rn"] for r in rows]); ravs = np.array([r["sys"]["rav"] for r in rows])
verd = [r["sys"]["verdict"] for r in rows]
EXP = {"enter":1.0, "stage":0.3, "walk":0.0}; CHI = 0.5
def loss(vlist):
    L=0.0
    for v,npv in zip(vlist,rns):
        e=EXP[v]; L += e*max(0,-npv) + (1-e)*CHI*max(0,npv)
    return L
Lsys = loss(verd)
Lenter = loss(["enter"]*10); Lwalk = loss(["walk"]*10)
Lrand = np.mean([loss(rng.choice(["enter","stage","walk"],10)) for _ in range(2000)])
Lora  = loss(["enter" if x>0 else "walk" for x in rns])
print("cost-weighted verdict loss (US$M; exposure enter=1.0 stage=0.3 walk=0; missed-upside capture chi=0.5):")
print("   system %.0f | always-enter %.0f | always-walk %.0f | random %.0f | oracle %.0f" % (Lsys,Lenter,Lwalk,Lrand,Lora))
print("   -> system captures %.0f%% of the always-enter -> oracle improvement" % (100*(Lenter-Lsys)/max(1,Lenter-Lora)))
# error decomposition
com = [(r["c"]["name"]) for r in rows if r["sys"]["verdict"]=="enter" and r["rn"]<0]
omi = [(r["c"]["name"]) for r in rows if r["sys"]["verdict"]!="enter" and r["rn"]>0]
print("commission errors (entered, lost): %s" % (com or "none"))
print("omission errors (didn't enter, won): %s" % (omi or "none"))
# rank correlation with era-block bootstrap
blocks = {0:[i for i,r in enumerate(rows) if r["c"]["yr"]<=2013],
          1:[i for i,r in enumerate(rows) if 2014<=r["c"]["yr"]<=2017],
          2:[i for i,r in enumerate(rows) if r["c"]["yr"]>=2018]}
rhos=[]
for _ in range(4000):
    idx=[i for b in rng.integers(0,3,3) for i in blocks[b]]
    if len(set(ravs[idx]))>2:
        rh,_=spearmanr(ravs[idx],rns[idx])
        if not np.isnan(rh): rhos.append(rh)
rho0,_=spearmanr(ravs,rns)
print("Spearman(RAV, realized NPV) = %.2f;  era-block bootstrap 90%% CI [%.2f, %.2f]" % (
  rho0, np.percentile(rhos,5), np.percentile(rhos,95)))
# top-3 regret incl. random baseline
oracle3=np.sort(rns)[-3:].sum(); sys3=rns[np.argsort(-ravs)[:3]].sum()
rand3=np.mean([rns[list(cmb)].sum() for cmb in itertools.combinations(range(10),3)])
print("top-3 portfolio realized value: system $%.0fM | random $%.0fM | oracle $%.0fM" % (sys3,rand3,oracle3))
# verdict stability under supply-CAGR +/-2pp
era0 = g["era"]
flips=0; frag=[]
for r in rows:
    base=r["sys"]["verdict"]; fl=False
    for d in (-0.02,0.02):
        g["era"]=lambda yr,d=d,e=era0: (lambda p,s,su: (p,s,su+d))(*e(yr))
        v=g["system_run"](r["c"])["verdict"]
        if v!=base: fl=True
    g["era"]=era0
    if fl: flips+=1; frag.append(r["c"]["name"])
print("verdict fragility under supply-CAGR ±2pp: %d/10 flip  (%s)" % (flips, ", ".join(frag) or "-"))

print(); print("="*78); print("LAYER 3 — ATTRIBUTION & COMPLEMENTARITY"); print("="*78)
hits=sum(1 for r in rows if r["sys"]["top"]==r["c"]["truth"])
print("Sobol top-driver hit rate: %d/10 (misses are the capex-decided cases, structurally)" % hits)
neg=[r for r in rows if r["rn"]<0]
caught=0
for r in neg:
    muR,sR=posterior(r["c"],"rc")
    pov25 = 1-norm.cdf((np.log(1.25)-muR)/sR)
    audit = pov25>0.5
    ok = (r["sys"]["top"]==r["c"]["truth"]) or (r["c"]["truth"]=="capex" and audit)
    caught+=ok
    print("   %-28s decider=%-6s sobol=%-6s auditP(ovr>25%%)=%.0f%%  -> %s" % (
      r["c"]["name"], r["c"]["truth"], r["sys"]["top"], 100*pov25, "caught" if ok else "MISSED"))
print("complementarity: %d/%d failures caught by Sobol OR cost audit jointly" % (caught,len(neg)))

print(); print("="*78); print("LAYER 4 — VALIDITY"); print("="*78)
print("effective n ~ 3.2 (see study); all CIs above use era-block bootstrap accordingly")
print("not runnable offline: leakage audit of agent proposals (needs corpus-restricted agent),")
print("selection-rule reproduction by an independent party, expert-interaction metrics (needs TTC use)")
