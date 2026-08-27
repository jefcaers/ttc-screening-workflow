"""Python port of the Dashboard v4 JS engine (default settings) — the ground truth
the Excel edition is verified against, and the source of the precomputed Sobol table.
Ported line-for-line from TTC_Decision_Dashboard_v4.html; any deviation is a bug."""
import numpy as np

START, END = 2009, 2028
YEARS = list(range(START, END + 1))
NY = len(YEARS)
SCEN = [dict(key="ref",  name="Reference (Roskill-style)",  demandCAGR=0.07, subHaircut=0.00, wt=3, on=True),
        dict(key="ev",   name="EV acceleration",            demandCAGR=0.12, subHaircut=0.00, wt=1, on=True),
        dict(key="slow", name="Slow transition / post-GFC", demandCAGR=0.04, subHaircut=0.00, wt=1, on=True),
        dict(key="sub",  name="Battery substitution",       demandCAGR=0.07, subHaircut=0.20, wt=1, on=True)]
A0 = dict(refPrice=5800, gamma=2.0, supplyCAGR=0.07, discount=0.10, cashCost=2300,
          nameplate=17500, util=0.90, capex=100, construction=2, taxRate=0.35,
          royalty=0.03, depYears=10, priceFloor=2500, includeCapex=True)
AACE_SIG = {5: 0.45, 4: 0.34, 3: 0.24, 2: 0.15, 1: 0.09}
DFS_COST = 15
# defaults of the UI sliders
DEF = dict(dem=0.0, sup=0.0, gamma=2.0, disc=0.10, aace=5, capex=100, iw=0.25,
           lam=0.50, Sprem=25, tax=0.35, rc=True)
# reference cases (overrun analogs among default-selected)
OVR = [1.35, 1.50, 1.85, 1.60, 1.00]          # HM, Rincon, Zabuye, Qinghai, SilverPeak
ARG = [(2500, 17), (2700, 10)]                # (opex, cap) Argentine analogs
BRINE_OPEX = [2500, 2700, 2800, 3000, 3200, 1500, 1800]  # all selected brine cases

def derive():
    cash = round(sum(o * c for o, c in ARG) / sum(c for _, c in ARG) * 0.9)
    m = np.mean(BRINE_OPEX)
    cv = float(np.sqrt(np.mean((np.array(BRINE_OPEX) - m) ** 2)) / m)
    osig = round(min(cv, 0.35), 2)
    return cash, osig

def price_path(sc, A, iv):
    out = []
    for i, year in enumerate(YEARS):
        demand = (1 + sc["demandCAGR"] + iv[0]) ** i
        sub_ramp = min(max((year - 2015) / 5, 0), 1)
        eff = demand * (1 - sc["subHaircut"] * sub_ramp)
        supply = (1 + A["supplyCAGR"] + iv[1]) ** i
        p = A["refPrice"] * (eff / supply) ** A["gamma"]
        out.append(min(max(p, A["priceFloor"]), 30000))
    return np.array(out)

def npv_of(sc, A, iv):
    pp = price_path(sc, A, iv)
    npv = 0.0
    for i in range(NY):
        q = 0.0
        if i >= A["construction"]:
            py = i - A["construction"]
            q = (0.5 if py == 0 else 1.0) * A["nameplate"] * A["util"]
        capex_yr = A["capex"] / A["construction"] if (i < A["construction"] and A["includeCapex"]) else 0.0
        rev = pp[i] * q / 1e6
        ebitda = rev - A["cashCost"] * q / 1e6 - A["royalty"] * rev
        dep = A["capex"] / A["depYears"] if (A["includeCapex"] and i >= A["construction"]
                                             and (i - A["construction"]) < A["depYears"]) else 0.0
        tax = A["taxRate"] * max(0.0, ebitda - dep)
        npv += (ebitda - tax - capex_yr) / (1 + A["discount"]) ** i
    return npv

def cost_posterior_params(ovr=OVR, aace=5, iw=0.25, rc=True):
    lr = np.log(np.array(ovr if ovr else [1.35, 1.6]))
    mu0 = float(lr.mean())
    s0 = max(0.12, float(np.sqrt(((lr - mu0) ** 2).mean())) or 0.25)
    s1 = AACE_SIG[aace]
    w1 = iw / s1**2; w0 = (1 - iw) / s0**2
    muRC = w0 * mu0 / (w0 + w1)
    sRC = np.sqrt(1 / (w0 + w1)) + 0.5 * min(s0, s1)
    return (muRC, sRC) if rc else (0.0, s1)

def run_mc(N=3000, seed=7, D=DEF):
    rng = np.random.default_rng(seed)
    cash, osig = derive()
    A = dict(A0, cashCost=cash, gamma=D["gamma"], discount=D["disc"], taxRate=D["tax"])
    iv = (D["dem"], D["sup"])
    muP, sP = cost_posterior_params(aace=D["aace"], iw=D["iw"], rc=D["rc"])
    base = D["capex"]
    pool = [s for s in SCEN if s["on"]]
    totW = sum(s["wt"] for s in pool)
    sTight = AACE_SIG[max(1, D["aace"] - 2)]
    npvs, staged, scen_ix = [], [], []
    draws = base * np.exp(muP + sP * rng.standard_normal(N))
    for t in range(N):
        r = rng.random() * totW; sc = pool[0]; ix = 0
        for j, s in enumerate(pool):
            r -= s["wt"]
            if r <= 0: sc, ix = s, j; break
        capex = draws[t]
        cash_t = A["cashCost"] * np.exp(osig * rng.standard_normal())
        tax_t = min(0.6, max(0.0, A["taxRate"] * (1 + 0.10 * (rng.random() * 2 - 1))))
        Ap = dict(A, capex=capex, cashCost=cash_t, taxRate=tax_t)
        v = npv_of(sc, Ap, iv)
        npvs.append(v); scen_ix.append(ix)
        capex2 = base * np.exp(muP + (sTight / AACE_SIG[D["aace"]]) * sP * rng.standard_normal())
        staged.append(max(0.0, npv_of(sc, dict(Ap, capex=capex2), iv)) - DFS_COST)
    npvs = np.array(npvs); staged = np.array(staged); scen_ix = np.array(scen_ix)
    mu = npvs.mean()
    srt = np.sort(npvs)
    cvar = srt[:max(1, int(N * 0.05))].mean()
    rav = mu - D["lam"] * (mu - cvar) + D["Sprem"]
    opt = staged.mean()
    # bull/base/bear
    z = 1.2816
    cP10 = base * np.exp(muP + sP * -z); cP50 = base * np.exp(muP); cP90 = base * np.exp(muP + sP * z)
    sc_means = [npvs[scen_ix == j].mean() if (scen_ix == j).any() else 0 for j in range(len(pool))]
    ixBear = int(np.argmin(sc_means)); ixBull = int(np.argmax(sc_means))
    scBase = max(pool, key=lambda s: s["wt"])
    def bbb(sc, cap, om):
        return npv_of(sc, dict(A, capex=cap, cashCost=A["cashCost"] * np.exp(om * osig)), iv)
    nBear = bbb(pool[ixBear], cP90, +z); nBase = bbb(scBase, cP50, 0.0); nBull = bbb(pool[ixBull], cP10, -z)
    return dict(mu=mu, cvar=cvar, rav=rav, opt=opt, pPos=(npvs > 0).mean(),
                p10=srt[int(0.10 * N)], p50=srt[int(0.50 * N)], p90=srt[int(0.90 * N)],
                nBear=nBear, nBase=nBase, nBull=nBull, cP10=cP10, cP50=cP50, cP90=cP90,
                muP=muP, sP=sP, cash=cash, osig=osig,
                sc_means=sc_means, npvs=npvs, staged=staged,
                bear_sc=pool[ixBear]["name"], bull_sc=pool[ixBull]["name"], base_sc=scBase["name"])

def sobol(N=256, seed=11, D=DEF):
    rng = np.random.default_rng(seed)
    cash, osig = derive()
    A = dict(A0, cashCost=cash, gamma=D["gamma"], discount=D["disc"], taxRate=D["tax"])
    iv = (D["dem"], D["sup"])
    muP, sP = cost_posterior_params(aace=D["aace"], iw=D["iw"], rc=D["rc"])
    from scipy.stats import norm
    def f(x):
        capex = D["capex"] * np.exp(muP + sP * norm.ppf(x[0]))
        cashc = A["cashCost"] * np.exp(osig * norm.ppf(x[1]))
        tax = min(0.6, max(0.0, A["taxRate"] * (1 + 0.10 * (2 * x[2] - 1))))
        cagr = 0.04 + x[3] * 0.08
        sup = A["supplyCAGR"] - 0.02 + x[4] * 0.04
        sub = x[5] * 0.20
        return npv_of(dict(demandCAGR=cagr, subHaircut=sub),
                      dict(A, capex=capex, cashCost=cashc, taxRate=tax, supplyCAGR=sup), iv)
    k = 6
    Am = rng.random((N, k)); Bm = rng.random((N, k))
    fA = np.array([f(x) for x in Am]); fB = np.array([f(x) for x in Bm])
    allv = np.concatenate([fA, fB]); f0 = allv.mean(); V = ((allv - f0) ** 2).mean() or 1
    Si, STi = [], []
    for j in range(k):
        s = st = 0.0
        for i in range(N):
            x = Am[i].copy(); x[j] = Bm[i][j]
            fab = f(x)
            s += fB[i] * (fab - fA[i]); st += (fA[i] - fab) ** 2
        stv = max(0.0, st / (2 * N) / V)
        STi.append(stv); Si.append(min(stv, max(0.0, s / N / V)))
    return Si, STi

if __name__ == "__main__":
    cash, osig = derive()
    muP, sP = cost_posterior_params()
    print(f"derive: cash={cash} osig={osig}")
    print(f"posterior: muP={muP:.5f} sP={sP:.5f}")
    z = 1.2816
    print(f"capex P10/P50/P90 = {100*np.exp(muP-sP*z):.1f} / {100*np.exp(muP):.1f} / {100*np.exp(muP+sP*z):.1f}")
    r = run_mc()
    for k in ["mu", "cvar", "rav", "opt", "pPos", "p10", "p90", "nBear", "nBase", "nBull"]:
        print(f"  {k:6s} = {r[k]:.1f}" if not isinstance(r[k], str) else "")
    print(f"  bear={r['bear_sc']}  base={r['base_sc']}  bull={r['bull_sc']}")
    # deterministic anchors for Excel verification (no randomness)
    A = dict(A0, cashCost=cash)
    print("deterministic ref-scenario NPV @capex100:", round(npv_of(SCEN[0], A, (0, 0)), 4))
    print("mean price ref:", round(price_path(SCEN[0], A, (0, 0)).mean(), 2))
    Si, STi = sobol()
    names = ["CapEx", "Cash opex", "Tax rate", "Demand CAGR", "Supply CAGR", "Substitution"]
    for n, s, t in sorted(zip(names, Si, STi), key=lambda x: -x[2]):
        print(f"  Sobol {n:12s} Si={s:.3f} STi={t:.3f}")
