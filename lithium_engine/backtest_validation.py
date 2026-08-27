"""
Multi-case back-test validation of the TTC decision-dossier components.
Cohort rule (fixed BEFORE looking at outcomes, to avoid survivorship bias):
  Every greenfield lithium project worldwide that reached a formal construction
  decision (sanction/FID) between 2010 and 2019, with a publicly disclosed
  sanction-stage CapEx estimate and nameplate >= 5 kt LCE-eq/yr, tracked to
  first production, sale, or termination. No case may be dropped for outcome.
Leakage rule: for each case, the overrun reference pool contains ONLY outcomes
  realized before its sanction year; market parameters are set from that year's
  price level; nothing post-date enters the engines. Realized data are used
  only for scoring.
All figures approximate, from public disclosures (NI 43-101 / ASX / company
  reports); confidence flagged per case. To be curated before publication.
"""
import numpy as np
rng = np.random.default_rng(11)

# ---------------- realized market path (score-side only) ----------------
# Battery-grade Li2CO3, approx annual average US$/t (China spot / contract blend)
PRICE = {2010:5500,2011:5700,2012:5900,2013:5400,2014:5500,2015:6500,2016:12000,
         2017:13500,2018:14000,2019:9500,2020:7000,2021:12500,2022:37000,
         2023:23000,2024:11000,2025:9000}
# route price capture per LCE-eq tonne (spodumene concentrate captures ~45% of carbonate)
F = {"brine":1.00,"spod":0.45,"hydrox":1.05}

# ---------------- pre-2010 overrun pool (knowable by any 2010+ decision) ----------------
PRE = [("HombreMuerto","brine",1998,1.35),("Zabuye","brine",2006,1.85),
       ("Qinghai","brine",2007,1.60),("SilverPeak","brine",1990,1.00),
       ("Rincon_pilot","brine",2009,1.50),
       ("Greenbushes_exp","spod",2000,1.10),("MtCattlin_precursor","spod",2009,1.20)]

# ---------------- cohort (rule-based; NOTHING dropped for outcome) ----------------
# fields: name, route, sanction yr, est capex US$M, realized ratio, outcome-known yr,
#   nameplate ktLCE-eq, cash cost US$/tLCE-eq, ramp = per-year utilization actually achieved
#   (from sanction+build), fate, ground-truth decider, data confidence
C = lambda **k: k
CASES = [
 C(name="Olaroz S1 (Orocobre)", route="brine", yr=2012, est=229, ratio=1.75, known=2016,
   cap=17.5, cc=4200, build=3, ramp=[0.3,0.5,0.65,0.8,0.85], fate="producing, profitable in boom",
   truth="price", conf="high"),
 C(name="Mt Cattlin (Galaxy)", route="spod", yr=2010, est=70, ratio=1.30, known=2013,
   cap=13.0, cc=3400, build=1, ramp=[0.5,0.6,0.0,0.0,0.0,0.6,0.8,0.8], fate="shut 2013 on price, restart 2016",
   truth="price", conf="med"),
 C(name="Mt Marion (Neometals/Ganfeng)", route="spod", yr=2015, est=90, ratio=1.15, known=2017,
   cap=26.0, cc=2500, build=1, ramp=[0.6,0.85,0.9,0.9,0.9], fate="producing through boom",
   truth="price", conf="med"),
 C(name="Pilgangoora (Pilbara)", route="spod", yr=2017, est=180, ratio=1.17, known=2019,
   cap=40.0, cc=2300, build=1, ramp=[0.4,0.6,0.5,0.6,0.9,0.95], fate="survived trough, boom winner",
   truth="price", conf="high"),
 C(name="Altura Pilgangoora", route="spod", yr=2017, est=110, ratio=1.30, known=2019,
   cap=28.0, cc=2700, build=1, ramp=[0.5,0.8,0.85,0.0,0.0], fate="receivership 2020 (price+debt), equity wiped",
   truth="price", conf="med"),
 C(name="Nemaska Whabouchi", route="hydrox", yr=2018, est=880, ratio=1.45, known=2019,
   cap=29.0, cc=3800, build=3, ramp=[0.0,0.0,0.0,0.0,0.0], fate="C$375M shortfall revealed 2019; CCAA, ~55% spent, zero production",
   truth="capex", conf="high"),
 C(name="Bald Hill (Tawana/Alita)", route="spod", yr=2018, est=42, ratio=1.25, known=2019,
   cap=11.0, cc=3600, build=1, ramp=[0.6,0.7,0.0,0.0], fate="administration 2019 on price",
   truth="price", conf="med"),
 C(name="Wodgina concentrator (MinRes)", route="spod", yr=2018, est=420, ratio=1.10, known=2019,
   cap=100.0, cc=2900, build=1, ramp=[0.3,0.0,0.0,0.5,0.8,0.85], fate="mothballed 2019 on price; 60% JV sale to ALB monetized it",
   truth="price", conf="med"),
 C(name="Cauchari-Olaroz (LAC/Ganfeng)", route="brine", yr=2019, est=565, ratio=1.73, known=2023,
   cap=40.0, cc=6500, build=4, ramp=[0.0,0.0,0.1,0.5,0.75,0.85], fate="producing 2023 into the price collapse",
   truth="capex", conf="high"),
 C(name="AMG Mibra (Brazil)", route="spod", yr=2017, est=60, ratio=1.05, known=2019,
   cap=13.0, cc=2200, build=1, ramp=[0.6,0.9,0.9,0.9,0.9], fate="on budget, byproduct economics, steady",
   truth="price", conf="low"),
]

AACE_SIG = {5:0.45,4:0.34,3:0.24,2:0.15,1:0.09}
Z = 1.2816

def frozen_pool(case):
    """overrun ratios knowable strictly before the sanction year; same-route preferred if >=4"""
    pool = [(r,rt) for (_,rt,ky,r) in PRE if ky < case["yr"]]
    pool += [(c["ratio"],c["route"]) for c in CASES if c["known"] < case["yr"]]
    same = [r for r,rt in pool if rt == case["route"]]
    return (same if len(same) >= 4 else [r for r,_ in pool]), len(same) >= 4

def posterior(case, mode="rc", iw=0.30, aace=3):
    pool,_ = frozen_pool(case)
    lr = np.log(pool); mu0, s0 = lr.mean(), max(0.12, lr.std())
    s1 = AACE_SIG[aace]
    if mode == "inside": mu, s = 0.0, s1
    else:
        w1, w0 = iw/(s1*s1), (1-iw)/(s0*s0)
        mu = w0*mu0/(w0+w1); s = np.sqrt(1/(w0+w1)) + 0.5*min(s0,s1)
    return mu, s

# ---------------- decision-time market setup (era parameters, pre-date only) ----------------
def era(yr):
    p0 = PRICE[yr]  # sanction-year price level was knowable
    if yr <= 2013:  scen=[(0.07,3),(0.12,1),(0.04,1)]; sup=0.07
    elif yr <= 2017: scen=[(0.13,3),(0.18,1),(0.06,1)]; sup=0.13   # EV real; supply wave forming
    else:            scen=[(0.14,3),(0.20,1),(0.05,1)]; sup=0.19   # visible oversupply wave
    return p0, scen, sup

def npv_engine(case, capex, price_path_fn, disc=0.10, tax=0.30, roy=0.03, T=15):
    f = F[case["route"]]; npv = 0.0
    for t in range(T):
        u = 0.0 if t < case["build"] else 0.9  # planned: 90% from end of build (yr1 at 50%)
        if t == case["build"]: u = 0.45
        price = price_path_fn(t)
        rev = f*price*case["cap"]*u/1000.0     # US$M (cap in kt)
        eb = rev*(1-roy) - case["cc"]*case["cap"]*u/1000.0
        cx = capex/case["build"] if t < case["build"] else 0.0
        dep = capex/10 if (t >= case["build"] and t < case["build"]+10) else 0.0
        npv += (eb - tax*max(0, eb-dep) - cx)/(1+disc)**t
    return npv

def system_run(case, mode="rc", N=2000):
    """decision-time verdict + RAV using only pre-date info"""
    p0, scen, sup = era(case["yr"])
    mu, s = posterior(case, mode)
    draws = case["est"]*np.exp(mu + s*rng.standard_normal(N))
    tot = sum(w for _,w in scen)
    npvs = np.empty(N)
    for i in range(N):
        r = rng.random()*tot; g = scen[0][0]
        for cg,w in scen:
            r -= w
            if r <= 0: g = cg; break
        gam = 1.2  # long-run tightness sensitivity, era-agnostic
        pp = lambda t, g=g: min(max(p0*((1+g)/(1+sup))**(gam*t), 0.4*p0), 4.0*p0)
        cc_mult = np.exp(0.18*rng.standard_normal())
        c2 = dict(case); c2["cc"] = case["cc"]*cc_mult
        npvs[i] = npv_engine(c2, draws[i], pp)
    mu_n = npvs.mean(); srt = np.sort(npvs); cvar = srt[:max(1,N//20)].mean()
    rav = mu_n - 0.5*(mu_n - cvar)
    # Sobol-lite (total effects via one-at-a-time variance at extremes is too crude; use Saltelli k=3)
    Ns, k = 128, 3  # capex, cashcost, demandCAGR
    def fm(x):
        cap_d = case["est"]*np.exp(mu + s*_qn(x[0]))
        g = 0.04 + x[2]*0.16
        pp = lambda t: min(max(p0*((1+g)/(1+sup))**(1.2*t), 0.4*p0), 4.0*p0)
        c2 = dict(case); c2["cc"] = case["cc"]*np.exp(0.18*_qn(x[1]))
        return npv_engine(c2, cap_d, pp)
    A = rng.random((Ns,k)); B = rng.random((Ns,k))
    fA = np.array([fm(a) for a in A]); fB = np.array([fm(b) for b in B])
    V = np.concatenate([fA,fB]).var() or 1.0
    ST = []
    for j in range(k):
        AB = A.copy(); AB[:,j] = B[:,j]
        fAB = np.array([fm(x) for x in AB])
        ST.append(max(0, np.mean((fA-fAB)**2)/(2*V)))
    top = ["capex","opex","price"][int(np.argmax(ST))]
    verdict = "walk" if max(rav,0)==0 and mu_n<=0 else ("enter" if rav>0 else "stage")
    if rav <= 0 and mu_n > 0: verdict = "stage"
    return dict(rav=rav, enpv=mu_n, cvar=cvar, top=top, ST=dict(zip(["capex","opex","price"],np.round(ST,2))), verdict=verdict)

from scipy.stats import norm
def _qn(u): return norm.ppf(np.clip(u,1e-9,1-1e-9))

def realized_npv(case, disc=0.10, tax=0.30, roy=0.03):
    """scoring side: actual price path, realized capex, realized ramp"""
    f = F[case["route"]]; capex = case["est"]*case["ratio"]
    spent = capex*(0.55 if "Nemaska" in case["name"] else 1.0)
    npv = 0.0; T = 2026 - case["yr"]
    for t in range(T):
        yr = case["yr"]+t
        price = PRICE.get(yr, PRICE[2025])
        u = case["ramp"][t-case["build"]] if (t >= case["build"] and t-case["build"] < len(case["ramp"])) else \
            (case["ramp"][-1] if t >= case["build"] else 0.0)
        rev = f*price*case["cap"]*u/1000.0
        eb = rev*(1-roy) - case["cc"]*case["cap"]*u/1000.0
        cx = spent/case["build"] if t < case["build"] else 0.0
        dep = spent/10 if (t>=case["build"] and t<case["build"]+10) else 0.0
        npv += (eb - tax*max(0,eb-dep) - cx)/(1+disc)**t
    # continuation value at 2026 for still-producing assets: last-year margin annuity (8y @ disc)
    if case["ramp"][-1] > 0:
        u = case["ramp"][-1]; price = PRICE[2025]
        eb = (f*price*case["cap"]*u/1000.0)*(1-roy) - case["cc"]*case["cap"]*u/1000.0
        ann = eb*(1-tax*0.8)*(1-(1+disc)**-8)/disc
        npv += ann/(1+disc)**T
    if "Wodgina" in case["name"]: npv += 700/(1+disc)**1.5  # ALB JV monetization (approx, 60% at ~US$1.15B)
    return npv

# ================= RUN =================
print("%-32s %4s %5s | pool  n(rt) | RC P10-P90   in? | IN P10-P90   in? | realizedNPV  RAV(sys) top(sys) truth verdict" % ("case","yr","ratio"))
rows=[]
for c in CASES:
    pool, same = frozen_pool(c)
    muR,sR = posterior(c,"rc"); muI,sI = posterior(c,"inside")
    bandR = (np.exp(muR-Z*sR), np.exp(muR+Z*sR)); bandI = (np.exp(muI-Z*sI), np.exp(muI+Z*sI))
    covR = bandR[0] <= c["ratio"] <= bandR[1]; covI = bandI[0] <= c["ratio"] <= bandI[1]
    # log score of realized ratio under each posterior (lognormal density)
    def logsc(mu,s): x=np.log(c["ratio"]); return -0.5*((x-mu)/s)**2 - np.log(s) - 0.5*np.log(2*np.pi)
    sysr = system_run(c)
    rn = realized_npv(c)
    rows.append(dict(c=c, covR=covR, covI=covI, lsR=logsc(muR,sR), lsI=logsc(muI,sI),
                     rn=rn, sys=sysr, pool=len(pool), same=same))
    print("%-32s %4d %5.2f | %3d %5s | %.2f-%.2f  %3s | %.2f-%.2f  %3s | %9.0f  %8.0f  %-6s %-5s %s" % (
      c["name"], c["yr"], c["ratio"], len(pool), same, bandR[0],bandR[1], "YES" if covR else "no",
      bandI[0],bandI[1], "YES" if covI else "no", rn, sysr["rav"], sysr["top"], c["truth"], sysr["verdict"]))

n=len(rows)
covR=sum(r["covR"] for r in rows); covI=sum(r["covI"] for r in rows)
lsR=np.mean([r["lsR"] for r in rows]); lsI=np.mean([r["lsI"] for r in rows])
print("\n--- 1. CapEx calibration (80%% band should cover ~8/10) ---")
print("reference-class coverage: %d/%d   inside-only coverage: %d/%d" % (covR,n,covI,n))
print("mean log score: RC %.2f vs inside %.2f (higher=better, diff %.2f)" % (lsR,lsI,lsR-lsI))

print("\n--- 2. Decision ranking & regret ---")
from scipy.stats import spearmanr
ravs=np.array([r["sys"]["rav"] for r in rows]); rns=np.array([r["rn"] for r in rows])
rho,pv = spearmanr(ravs,rns)
print("Spearman(system RAV, realized NPV) = %.2f (p=%.2f)" % (rho,pv))
for k in [3,5]:
    pick=np.argsort(-ravs)[:k]; oracle=np.argsort(-rns)[:k]
    regret=rns[oracle].sum()-rns[pick].sum()
    print("top-%d portfolio: system captures $%.0fM of oracle $%.0fM (regret $%.0fM); overlap %d/%d" %
          (k, rns[pick].sum(), rns[oracle].sum(), regret, len(set(pick)&set(oracle)), k))
neg=[(r["c"]["name"],r["sys"]["verdict"]) for r in rows if r["rn"]<0]
print("cases with negative realized NPV and the system's ex-ante verdict:", neg)

print("\n--- 3. Attribution: Sobol top driver vs what actually decided the outcome ---")
hits=sum(1 for r in rows if r["sys"]["top"]==("price" if r["c"]["truth"]=="price" else r["c"]["truth"]))
print("hit rate %d/%d" % (hits,n))
for r in rows:
    print("  %-32s sobol=%-6s truth=%-6s %s" % (r["c"]["name"], r["sys"]["top"], r["c"]["truth"],
          "HIT" if r["sys"]["top"]==r["c"]["truth"] else "miss"))

print("\n--- 4. Effective n (shared price realization) ---")
yrs=sorted(PRICE); pv_=np.array([PRICE[y] for y in yrs]); lp=np.log(pv_)
def window(c): a=c["yr"]-2010; return lp[a:a+7]
rhos=[]
for i in range(n):
    for j in range(i+1,n):
        wi,wj=window(rows[i]["c"]),window(rows[j]["c"])
        m=min(len(wi),len(wj))
        if m>=4:
            r_=np.corrcoef(wi[:m],wj[:m])[0,1]
            if not np.isnan(r_): rhos.append(r_)
rho_bar=float(np.mean(rhos))
neff=n/(1+(n-1)*rho_bar)
print("mean pairwise price-window correlation = %.2f  ->  effective n = %.1f (nominal %d)" % (rho_bar,neff,n))
print("\nsanction-era clusters: 2010-12 (2 cases), 2015-17 (4), 2018-19 (4) -> ~3 quasi-independent price regimes")
