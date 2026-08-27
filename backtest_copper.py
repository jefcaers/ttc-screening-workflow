"""
Chilean copper cohort back-test — second commodity for the workflow-validation paper.
COHORT RULE (fixed ex ante, mirrors the lithium study): every copper project in Chile
with a formal construction decision (FID) 2008-2019, publicly disclosed sanction-stage
CapEx >= US$500M and nameplate >= 40 kt Cu/yr (or equivalent increment), tracked to
2025. Nothing dropped for outcome. State-owned (Codelco) projects included.
LEAKAGE RULE: identical to lithium — overrun pools contain only outcomes realized
strictly before each case's sanction year; market parameters use sanction-year price
levels and era-knowable ranges; realized data used only for scoring.
CONFIDENCE: high = web-verified July 2026 session; med = documented public knowledge;
low = approximate, must be curated before publication.
Fiscal/discount assumptions harmonized with the lithium engine (disc .10, tax .30,
roy .03) so cross-cohort comparisons isolate the components under test.
"""
import numpy as np
from scipy.stats import norm, spearmanr
rng = np.random.default_rng(7)

# ---------------- copper price path, LME annual avg approx US$/t (score-side) ----------
CU = {2008:6950,2009:5150,2010:7535,2011:8820,2012:7950,2013:7325,2014:6860,
      2015:5495,2016:4860,2017:6165,2018:6525,2019:6000,2020:6180,2021:9315,
      2022:8820,2023:8480,2024:9150,2025:9400}
# lithium price path (from the validated lithium engine, for combined n_eff)
LI = {2010:5500,2011:5700,2012:5900,2013:5400,2014:5500,2015:6500,2016:12000,
      2017:13500,2018:14000,2019:9500,2020:7000,2021:12500,2022:37000,
      2023:23000,2024:11000,2025:9000}

F = {"conc":0.94,"cath":1.00}   # payability net of TC/RC (concentrate) vs SX-EW cathode

# ---- by-product co-products (payable per year at steady state, 100% basis) ----
# Verified July 2026 from operator/JORC/6-K disclosures; see CURATION_REGISTER.md.
# moly in tonnes/yr, gold in koz/yr. Credits reduce net copper cost -> raise realized NPV.
BYPROD = {
  "Sierra Gorda (KGHM/Sumitomo)": {"mo": 5000, "au": 54.0},   # 2021 basis: 5kt Mo, 54koz Au
  "Caserones (JX/Mitsui)":        {"mo": 2450, "au": 0.0},     # ~2.4-2.5kt Mo, negligible Au
}
# by-product prices, USD/unit (long-run, conservative): Mo USD/t, Au USD/oz
MO_P = 22000.0     # ~USD 10/lb Mo
AU_P = 1600.0      # USD/oz, cohort-era average
def byproduct_credit_musd(case):
    """Annual by-product revenue in US$MM at steady state, or 0 if none disclosed."""
    b = BYPROD.get(case["name"])
    if not b: return 0.0
    return (b["mo"] * MO_P + b["au"] * 1000.0 * AU_P) / 1e6


# ------------- pre-2008 Chilean/major copper overrun seeds (knowable by 2008+) ---------
PRE_CU = [("EscondidaPh4",2002,1.05),("Collahuasi_exp",2004,1.15),("Spence_orig",2006,1.10),
          ("Gaby",2008,1.25),("LosPelambres_exp",2009,1.15)]

C = lambda **k: k
CASES_CU = [
 C(name="Caserones (JX/Mitsui)",        yr=2010, est=2000, ratio=2.10, known=2015, route="conc",
   cap=130, cc=5500, build=4, ramp=[0.3,0.5,0.6,0.7,0.75,0.8],
   fate="chronic ramp problems; JX sold control to Lundin 2023", truth="capex", conf="high"),
 C(name="Ministro Hales (Codelco)",     yr=2010, est=2300, ratio=1.30, known=2015, route="conc",
   cap=160, cc=3300, build=3, ramp=[0.5,0.7,0.8,0.85,0.9],
   fate="producing; roaster ramp issues 2014-15", truth="price", conf="med"),
 C(name="Sierra Gorda (KGHM/Sumitomo)", yr=2011, est=2900, ratio=1.40, known=2016, route="conc",
   cap=120, cc=5000, build=3, ramp=[0.4,0.5,0.55,0.6,0.7,0.8,0.85],
   fate="years below design; writedowns; Sumitomo exit 2022", truth="capex", conf="high"),
 C(name="Escondida OGP1 (BHP)",         yr=2012, est=3800, ratio=1.10, known=2016, route="conc",
   cap=180, cc=2900, build=3, ramp=[0.6,0.8,0.9,0.9],
   fate="delivered; grade-driven output dips", truth="price", conf="low"),
 C(name="Chuquicamata UG (Codelco)",    yr=2012, est=4200, ratio=1.31, known=2021, route="conc",
   cap=320, cc=3500, build=7, ramp=[0.2,0.35,0.5,0.6,0.7],
   fate="slow underground ramp; 2025 irregularities review", truth="ops", conf="med"),
 C(name="Antucoya (AMSA/Marubeni)",     yr=2012, est=1700, ratio=1.12, known=2016, route="cath",
   cap=80,  cc=4600, build=3, ramp=[0.6,0.8,0.85,0.85],
   fate="suspended 2012 for review, completed; low-grade, price-sensitive", truth="price", conf="med"),
 C(name="El Teniente NML (Codelco)",    yr=2011, est=3300, ratio=1.60, known=2023, route="conc",
   cap=140, cc=3000, build=9, ramp=[0.2,0.4,0.5],
   fate="2015 suspension after rock stress; restructured phased build", truth="capex", conf="low"),
 C(name="Encuentro Oxides (AMSA)",      yr=2015, est=636,  ratio=1.01, known=2018, route="cath",
   cap=50,  cc=4000, build=2, ramp=[0.7,0.9,0.9],
   fate="on time, on budget", truth="price", conf="med"),
 C(name="Spence Growth Option (BHP)",   yr=2017, est=2460, ratio=1.02, known=2021, route="conc",
   cap=185, cc=3300, build=4, ramp=[0.5,0.75,0.85,0.9],
   fate="near budget; COVID schedule slip", truth="price", conf="med"),
 C(name="Quebrada Blanca 2 (Teck/Sumitomo)", yr=2018, est=5200, ratio=1.67, known=2024, route="conc",
   cap=285, cc=3100, build=5, ramp=[0.3,0.6,0.75],
   fate="US$5.2B approved -> ~US$8.7B; tailings/ramp problems 2023-24", truth="capex", conf="high"),
 C(name="Rajo Inca (Codelco Salvador)", yr=2019, est=1300, ratio=1.38, known=2024, route="conc",
   cap=90,  cc=4200, build=4, ramp=[0.4,0.6],
   fate="producing 2023-24 after delays", truth="capex", conf="low"),
]

AACE_SIG = {5:0.45,4:0.34,3:0.24,2:0.15,1:0.09}
Z = 1.2816

def frozen_pool(case, cases, pre):
    pool = [r for (_,ky,r) in pre if ky < case["yr"]]
    pool += [c["ratio"] for c in cases if c["known"] < case["yr"]]
    return pool

def posterior(pool, mode="rc", iw=0.30, aace=3):
    lr = np.log(pool if len(pool) else [1.2,1.4])
    mu0, s0 = lr.mean(), max(0.12, lr.std())
    s1 = AACE_SIG[aace]
    if mode == "inside": return 0.0, s1
    w1, w0 = iw/(s1*s1), (1-iw)/(s0*s0)
    return w0*mu0/(w0+w1), np.sqrt(1/(w0+w1)) + 0.5*min(s0, s1)

# ---------------- era parameters (copper; pre-date knowable only) ----------------------
def era_cu(yr):
    p0 = CU[yr]
    if yr <= 2013:  scen=[(0.035,3),(0.05,1),(0.015,1)]; sup=0.030   # China boom visible
    elif yr <= 2016: scen=[(0.020,3),(0.035,1),(0.005,1)]; sup=0.025 # trough era
    else:            scen=[(0.025,3),(0.040,1),(0.010,1)]; sup=0.020 # EV/grid narrative
    return p0, scen, sup

GAM_CU = 0.6   # copper price far less D/S-elastic than lithium (deep, liquid market)

def npv_engine(case, capex, ppath, disc=0.10, tax=0.30, roy=0.03, T=20):
    f = F[case["route"]]; npv = 0.0
    for t in range(T):
        u = 0.0 if t < case["build"] else 0.9
        if t == case["build"]: u = 0.45
        rev = f*ppath(t)*case["cap"]*u/1000.0
        eb = rev*(1-roy) - case["cc"]*case["cap"]*u/1000.0
        cx = capex/case["build"] if t < case["build"] else 0.0
        dep = capex/10 if (t >= case["build"] and t < case["build"]+10) else 0.0
        npv += (eb - tax*max(0, eb-dep) - cx)/(1+disc)**t
    return npv

def system_run(case, N=2000):
    p0, scen, sup = era_cu(case["yr"])
    mu, s = posterior(frozen_pool(case, CASES_CU, PRE_CU))
    draws = case["est"]*np.exp(mu + s*rng.standard_normal(N))
    tot = sum(w for _,w in scen)
    npvs = np.empty(N)
    for i in range(N):
        r = rng.random()*tot; g = scen[0][0]
        for cg,w in scen:
            r -= w
            if r <= 0: g = cg; break
        pp = lambda t,g=g: min(max(p0*((1+g)/(1+sup))**(GAM_CU*t), 0.5*p0), 2.5*p0)
        c2 = dict(case); c2["cc"] = case["cc"]*np.exp(0.15*rng.standard_normal())
        npvs[i] = npv_engine(c2, draws[i], pp)
    mu_n = npvs.mean(); srt = np.sort(npvs); cvar = srt[:max(1,N//20)].mean()
    rav = mu_n - 0.5*(mu_n - cvar)
    verdict = "walk" if rav<=0 and mu_n<=0 else ("stage" if rav<=0 else "enter")
    return dict(rav=rav, enpv=mu_n, cvar=cvar, verdict=verdict)

def realized_npv(case, disc=0.10, tax=0.30, roy=0.03, roy_bp=0.0):
    f = F[case["route"]]; capex = case["est"]*case["ratio"]
    npv = 0.0; T = 2026 - case["yr"]
    for t in range(T):
        yr = case["yr"]+t
        price = CU.get(yr, CU[2025])
        u = case["ramp"][t-case["build"]] if (t >= case["build"] and t-case["build"] < len(case["ramp"])) else \
            (case["ramp"][-1] if t >= case["build"] else 0.0)
        rev = f*price*case["cap"]*u/1000.0 + byproduct_credit_musd(case)*u*(1-roy_bp)
        eb = rev - roy*(f*price*case["cap"]*u/1000.0) - case["cc"]*case["cap"]*u/1000.0
        cx = capex/case["build"] if t < case["build"] else 0.0
        dep = capex/10 if (t>=case["build"] and t<case["build"]+10) else 0.0
        npv += (eb - tax*max(0,eb-dep) - cx)/(1+disc)**t
    if case["ramp"][-1] > 0:
        u = case["ramp"][-1]; price = CU[2025]
        eb = (f*price*case["cap"]*u/1000.0)*(1-roy) + byproduct_credit_musd(case)*u - case["cc"]*case["cap"]*u/1000.0
        ann = eb*(1-tax*0.8)*(1-(1+disc)**-10)/disc
        npv += ann/(1+disc)**T
    return npv

# ============================== RUN: copper cohort =====================================
print("="*100)
print("CHILEAN COPPER COHORT (n=%d) — RC calibration, decisions, cross-commodity tests" % len(CASES_CU))
print("="*100)
rows=[]
print("%-34s %4s %5s conf | pool | RC P10-P90  in? | IN P10-P90  in? | realizedNPV RAV     verdict" % ("case","yr","ratio"))
for c in CASES_CU:
    pool = frozen_pool(c, CASES_CU, PRE_CU)
    muR,sR = posterior(pool,"rc"); muI,sI = posterior(pool,"inside")
    bR=(np.exp(muR-Z*sR),np.exp(muR+Z*sR)); bI=(np.exp(muI-Z*sI),np.exp(muI+Z*sI))
    covR=bR[0]<=c["ratio"]<=bR[1]; covI=bI[0]<=c["ratio"]<=bI[1]
    def logsc(mu,s): x=np.log(c["ratio"]); return -0.5*((x-mu)/s)**2-np.log(s)-0.5*np.log(2*np.pi)
    sysr=system_run(c); rn=realized_npv(c)
    rows.append(dict(c=c,covR=covR,covI=covI,lsR=logsc(muR,sR),lsI=logsc(muI,sI),rn=rn,sys=sysr,npool=len(pool)))
    print("%-34s %4d %5.2f %4s | %3d  | %.2f-%.2f %3s | %.2f-%.2f %3s | %9.0f %7.0f  %s" % (
      c["name"][:34],c["yr"],c["ratio"],c["conf"],len(pool),bR[0],bR[1],"YES" if covR else "no",
      bI[0],bI[1],"YES" if covI else "no",rn,sysr["rav"],sysr["verdict"]))

covR=sum(r["covR"] for r in rows); covI=sum(r["covI"] for r in rows); n=len(rows)
lsR=np.mean([r["lsR"] for r in rows]); lsI=np.mean([r["lsI"] for r in rows])
print("\n--- 1. CapEx calibration (80%% band, target ~%d/%d) ---" % (round(0.8*n),n))
print("reference-class coverage: %d/%d   inside-only: %d/%d" % (covR,n,covI,n))
print("mean log score: RC %.2f vs inside %.2f (diff %.2f nats/case)" % (lsR,lsI,lsR-lsI))
missI=[r["c"]["name"] for r in rows if not r["covI"]]
print("inside-view misses:", missI)

print("\n--- 2. Decision layer ---")
ravs=np.array([r["sys"]["rav"] for r in rows]); rns=np.array([r["rn"] for r in rows])
rho,pv=spearmanr(ravs,rns)
print("Spearman(RAV, realized NPV) = %.2f (p=%.2f)" % (rho,pv))
neg=[(r["c"]["name"],r["sys"]["verdict"]) for r in rows if r["rn"]<0]
print("negative-NPV cases & ex-ante verdict:", neg)

print("\n--- 3. Trading-house decision audit (the TTC-analog cases) ---")
for nm in ["Sierra Gorda","Quebrada Blanca"]:
    r=[x for x in rows if nm in x["c"]["name"]][0]
    print("%-34s actual: ENTER (Sumitomo) | engine (frozen info): %-5s | RAV %6.0f | realized NPV %7.0f | overrun x%.2f" % (
      r["c"]["name"][:34], r["sys"]["verdict"].upper(), r["sys"]["rav"], r["rn"], r["c"]["ratio"]))

# ============================== combined n_eff =========================================
print("\n--- 4. Combined effective n (both cohorts, shared price realizations) ---")
LI_CASES_YR=[2012,2010,2015,2017,2017,2018,2018,2018,2019,2017]  # lithium cohort sanction yrs
def logwin(series, yr, L=7):
    yrs=sorted(series); v=np.log([series[y] for y in yrs])
    a=yrs.index(yr) if yr in yrs else 0
    return v[a:a+L]
wins=[("li",logwin(LI,y)) for y in LI_CASES_YR]+[("cu",logwin(CU,c["yr"])) for c in CASES_CU]
rhos=[]
for i in range(len(wins)):
    for j in range(i+1,len(wins)):
        wi,wj=wins[i][1],wins[j][1]; m=min(len(wi),len(wj))
        if m>=4:
            r_=np.corrcoef(wi[:m],wj[:m])[0,1]
            if not np.isnan(r_): rhos.append(r_)
rb=float(np.mean(rhos)); N=len(wins)
neff=min(N, N/(1+(N-1)*rb)) if rb>-1/(N-1) else N
print("mean pairwise window correlation (21 cases, cross-commodity included) = %.2f" % rb)
print("combined effective n ~ %.0f of nominal %d (rho~0 -> near-independent; capped at nominal)" % (neff, N))
print("(lithium-only n_eff was ~3.2; adding copper's independent price realization is the whole point)")

# ============================== leave-one-commodity-out ================================
print("\n--- 5. Leave-one-commodity-out: does one commodity's overrun class cover the other? ---")
LI_RATIOS=[("Olaroz",2012,1.75),("MtCattlin",2010,1.30),("MtMarion",2015,1.15),("Pilgangoora",2017,1.17),
  ("Altura",2017,1.30),("Nemaska",2018,1.45),("BaldHill",2018,1.25),("Wodgina",2018,1.10),
  ("Cauchari",2019,1.73),("Mibra",2017,1.05)]
def loco(target_list, src_cases, src_pre, label):
    cov=0; used=0
    for nm,yr,ratio in target_list:
        pool=[r for (_,ky,r) in src_pre if ky<yr]+[c["ratio"] for c in src_cases if c["known"]<yr]
        if len(pool)<3: continue
        used+=1
        mu,s=posterior(pool,"rc")
        if np.exp(mu-Z*s)<=ratio<=np.exp(mu+Z*s): cov+=1
    print("%s: %d/%d covered (cases with >=3 cross-commodity priors available)" % (label,cov,used))
loco(LI_RATIOS, CASES_CU, PRE_CU, "copper-trained prior -> lithium outcomes")
CU_AS_TARGET=[(c["name"],c["yr"],c["ratio"]) for c in CASES_CU]
PRE_LI=[("HombreMuerto",1998,1.35),("Zabuye",2006,1.85),("Qinghai",2007,1.60),
        ("SilverPeak",1990,1.00),("Rincon_pilot",2009,1.50)]
LI_AS_SRC=[C(name=n,yr=y,ratio=r,known={"Olaroz":2016,"MtCattlin":2013,"MtMarion":2017,"Pilgangoora":2019,
  "Altura":2019,"Nemaska":2019,"BaldHill":2019,"Wodgina":2019,"Cauchari":2023,"Mibra":2019}[n])
  for (n,y,r) in LI_RATIOS]
loco(CU_AS_TARGET, LI_AS_SRC, PRE_LI, "lithium-trained prior -> copper outcomes")

print("\nAll figures approximate; conf flags per case; curate before publication.")
