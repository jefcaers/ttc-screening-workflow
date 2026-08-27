"""Intervention-function ablation, Olaroz 2009 (spec: ablation_spec.md C).
Ported dashboard engine, audited defaults; overlay = first-passage E/N/L (T blocked,
zero statutory headroom), panel-estimated interventionist hazards, sourced magnitudes.
Common random numbers across the two variants."""
import numpy as np
import excel_engine_port as EP

D = dict(EP.DEF)
N = 3000
SEED = 7
P5 = {"E": 0.24, "N": 0.26, "L": 0.21}          # panel-estimated, interventionist row
HAZ = {k: 1 - (1 - p)**(1/5) for k, p in P5.items()}
DUTY = 0.0287                                    # NI 43-101 net duty incidence on revenue
L_SCOPE = 0.42                                   # Naraha-derived capital scope multiplier
N_PREM = 0.05                                    # YPF-anchored discount premium

def npv_overlay(sc, A, iv, fire):
    """npv_of with lever overlay; fire = dict of firing year index (or None)."""
    pp = EP.price_path(sc, A, iv)
    npv = 0.0
    disc0 = A["discount"]
    for i in range(EP.NY):
        q = 0.0
        if i >= A["construction"]:
            py = i - A["construction"]
            q = (0.5 if py == 0 else 1.0) * A["nameplate"] * A["util"]
        capex_yr = A["capex"] / A["construction"] if (i < A["construction"] and A["includeCapex"]) else 0.0
        if fire.get("L") is not None and i == fire["L"]:
            capex_yr += L_SCOPE * A["capex"]                     # one-time mandated plant capital
        rev = pp[i] * q / 1e6
        if fire.get("E") is not None and i >= fire["E"]:
            rev *= (1 - DUTY)                                    # export-duty incidence
        ebitda = rev - A["cashCost"] * q / 1e6 - A["royalty"] * rev
        dep = A["capex"] / A["depYears"] if (A["includeCapex"] and i >= A["construction"]
                                             and (i - A["construction"]) < A["depYears"]) else 0.0
        tax = A["taxRate"] * max(0.0, ebitda - dep)
        disc = disc0 + (N_PREM if (fire.get("N") is not None and i >= fire["N"]) else 0.0)
        npv += (ebitda - tax - capex_yr) / (1 + disc) ** i
    return npv

def run(with_intervention, N=N, seed=SEED):
    rng = np.random.default_rng(seed)
    cash, osig = EP.derive()
    A = dict(EP.A0, cashCost=cash, gamma=D["gamma"], discount=D["disc"], taxRate=D["tax"])
    iv = (D["dem"], D["sup"])
    muP, sP = EP.cost_posterior_params(aace=D["aace"], iw=D["iw"], rc=D["rc"])
    base = D["capex"]
    pool = [s for s in EP.SCEN if s["on"]]
    totW = sum(s["wt"] for s in pool)
    sTight = EP.AACE_SIG[max(1, D["aace"] - 2)]
    draws = base * np.exp(muP + sP * rng.standard_normal(N))
    npvs, staged = [], []
    for t in range(N):
        r = rng.random() * totW; sc = pool[0]
        for s in pool:
            r -= s["wt"]
            if r <= 0: sc = s; break
        cash_t = A["cashCost"] * np.exp(osig * rng.standard_normal())
        tax_t = min(0.6, max(0.0, A["taxRate"] * (1 + 0.10 * (rng.random() * 2 - 1))))
        # first-passage firing years (same rng stream in both variants via common draws)
        u = rng.random(3)
        if with_intervention:
            fire = {}
            for (k, uu) in zip(["E", "N", "L"], u):
                yr = int(np.ceil(np.log(1 - uu) / np.log(1 - HAZ[k]))) if uu < 1 else EP.NY + 1
                fire[k] = yr if yr < EP.NY else None
        else:
            fire = {"E": None, "N": None, "L": None}
        Ap = dict(A, capex=draws[t], cashCost=cash_t, taxRate=tax_t)
        npvs.append(npv_overlay(sc, Ap, iv, fire))
        capex2 = base * np.exp(muP + (sTight / EP.AACE_SIG[D["aace"]]) * sP * rng.standard_normal())
        staged.append(max(0.0, npv_overlay(sc, dict(Ap, capex=capex2), iv, fire)) - EP.DFS_COST)
    npvs = np.array(npvs); staged = np.array(staged)
    mu = npvs.mean()
    cvar = np.sort(npvs)[:max(1, int(N * 0.05))].mean()
    rav = mu - D["lam"] * (mu - cvar) + D["Sprem"]
    opt = staged.mean()
    choice = "stage" if (opt > 0 and opt > rav) else ("enter" if rav > 0 else "walk")
    return dict(mu=mu, cvar=cvar, rav=rav, opt=opt, choice=choice, pPos=(npvs > 0).mean())

print("=" * 88)
print("INTERVENTION-FUNCTION ABLATION, Olaroz 2009, audited configuration (spec C)")
print("=" * 88)
print(f"hazards/yr from panel interventionist row: " +
      ", ".join(f"{k} {h:.3f}" for k, h in HAZ.items()) + " | T blocked (zero statutory headroom)")
for tag, w in [("WITHOUT intervention (published engine)", False),
               ("WITH intervention overlay (E duty, N premium, L scope)", True)]:
    r = run(w)
    print(f"\n{tag}")
    print(f"  E[NPV] {r['mu']:8.1f}  CVaR5 {r['cvar']:8.1f}  RAV {r['rav']:8.1f}  "
          f"V_stage {r['opt']:8.1f}  P(NPV>0) {r['pPos']:.2f}  ->  choice: {r['choice'].upper()}")
