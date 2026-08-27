"""Adversarial verification of causal_validation_multiyear.py. Three checks:

CHECK 1 — hazard identity: frozen_hazard must reproduce the prior runs'
  frozen_phor exactly via 1-(1-p)^5 (same counts, same smoothing).

CHECK 2 — EXACT path expectation. The main script compounds survival under the
  MARGINAL belief each year:  P_marg = 1 - prod_k (1 - b_k . p).  The exact
  object under the chain is  P_exact = 1 - E_path[ prod_k (1 - p_{r_k}) ],
  computed by DP over (state, survived):  v_k = (v_{k-1} @ A) * (1 - p[:,j]).
  Because the chain is sticky (positively correlated survivals), P_marg can
  OVERSTATE the model's probability. If the pass only holds under P_marg and
  dies under P_exact, the pass is an artifact and must be reported as such.
  Cross-check the DP with brute-force Monte Carlo path simulation.

CHECK 3 — the pre-registered criterion re-scored under P_exact, incl. the
  negative control and the truncated-fit robustness beliefs.
"""
import numpy as np, sys
_os = __import__('os')
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import activation_panel as ap
import hmm_regime as hr
from causal_validation_multiyear import (frozen_hazard, base_rate, window_prob,
                                         train_truncated, logs, TESTS, H)
LEV = ap.LEVERS
rng = np.random.default_rng(11)

# ---- CHECK 1: hazard identity vs prior frozen_phor -------------------------
def frozen_phor_prior(cutoff, exclude_country):   # verbatim from causal_validation_hmm.py
    reg = {}
    for c, periods in ap.REGIME_PERIODS.items():
        if c == exclude_country: continue
        for y0, y1, r in periods:
            for y in range(max(y0, ap.YEAR0), min(y1, cutoff) + 1): reg[(c, y)] = r
    exp = np.zeros(3); cnt = np.zeros((3, 4))
    for r in reg.values(): exp[r] += 1
    for c, y, L, _ in ap.EVENTS:
        if y <= cutoff and c != exclude_country and (c, y) in reg:
            cnt[reg[(c, y)], LEV.index(L)] += 1
    p = (cnt + 0.5) / (exp[:, None] + 20.5); return 1 - (1 - p) ** 5

ok = True
for cut, ctry in [(2018, "Argentina"), (2021, "Chile"), (2018, "Australia")]:
    d = np.abs(frozen_phor_prior(cut, ctry) - (1 - (1 - frozen_hazard(cut, ctry)) ** 5)).max()
    print(f"CHECK 1 hazard identity ({ctry} @{cut}): max |diff| = {d:.2e}")
    ok &= d < 1e-12
print(f"CHECK 1: {'PASS' if ok else 'FAIL'}")

# ---- CHECK 2: exact DP vs marginal compounding vs Monte Carlo --------------
def window_prob_exact(belief0, A, p_yr, j, k_hi=H):
    v = belief0.copy()
    for _ in range(k_hi):
        v = (v @ A) * (1 - p_yr[:, j])
    return 1 - v.sum()

def window_prob_mc(belief0, A, p_yr, j, k_hi=H, n=400_000):
    states = rng.choice(3, size=n, p=belief0 / belief0.sum())
    alive = np.ones(n, bool)
    for _ in range(k_hi):
        u = rng.random(n)
        cum = A.cumsum(1)
        states = (u[:, None] > cum[states]).sum(1)
        alive &= rng.random(n) >= p_yr[states, j]
    return 1 - alive.mean()

m = hr._label_states(hr.train()); A = m["A"]
print("\nCHECK 2 (event: marginal / exact-DP / MC):")
ok2 = True
for title, cut, ctry, fired in TESTS:
    p_yr = frozen_hazard(cut, ctry)
    b0 = m["filtered"][(ctry, cut)]
    for lev, yr in fired:
        j = LEV.index(lev)
        pmar = window_prob(b0, A, p_yr, j)
        pex = window_prob_exact(b0, A, p_yr, j)
        pmc = window_prob_mc(b0, A, p_yr, j)
        print(f"  {ctry} {lev} {yr}: marginal {pmar:.4f}   exact {pex:.4f}   MC {pmc:.4f}"
              f"   (marg-exact {pmar-pex:+.4f})")
        ok2 &= abs(pex - pmc) < 0.004
print(f"CHECK 2 DP==MC: {'PASS' if ok2 else 'FAIL'}")

# ---- CHECK 3: criterion re-scored under the EXACT object -------------------
print("\nCHECK 3 — pre-registered criterion under EXACT path expectation:")
rows = []
for title, cut, ctry, fired in TESTS:
    p_yr = frozen_hazard(cut, ctry)
    b0 = m["filtered"][(ctry, cut)]
    for lev, yr in fired:
        j = LEV.index(lev)
        pex = window_prob_exact(b0, A, p_yr, j)
        pb = base_rate(cut, lev, ctry)
        print(f"  {ctry} {lev} {yr}: exact {pex:.3f}  base {pb:.3f}  "
              f"x{pex/pb:.2f} [{'BEATS' if pex>pb else 'below'}]")
        rows.append((pex, pb))
pe, pb = map(np.array, zip(*rows))
nb = int((pe > pb).sum())
print(f"  exact beats base on {nb}/4;  mean log-score vs base "
      f"{np.mean([logs(a)-logs(b) for a,b in zip(pe,pb)]):+.2f} nats/event")
print(f"  criterion (>=2/4) under EXACT object: {'PASS' if nb>=2 else 'FAIL'}")

b_au = m["filtered"][("Australia", 2018)]
p_au = window_prob_exact(b_au, A, frozen_hazard(2018, "Australia"), LEV.index("N"))
print(f"  negative control exact P(N Australia) = {p_au:.3f} (base {base_rate(2018,'N','Australia'):.3f})")

print("\n  truncated-fit (no-leak) beliefs under EXACT object:")
rowsT = []
for title, cut, ctry, fired in TESTS:
    mt = train_truncated(cut)
    p_yr = frozen_hazard(cut, ctry)
    b0 = mt["filtered"][(ctry, cut)]
    for lev, yr in fired:
        j = LEV.index(lev)
        pex = window_prob_exact(b0, mt["A"], p_yr, j)
        pb = base_rate(cut, lev, ctry)
        print(f"    {ctry} {lev} {yr}: exact {pex:.3f}  base {pb:.3f} "
              f"[{'BEATS' if pex>pb else 'below'}]")
        rowsT.append((pex, pb))
peT, pbT = map(np.array, zip(*rowsT))
print(f"    trunc-fit exact beats base on {int((peT>pbT).sum())}/4;  mean log-score "
      f"{np.mean([logs(a)-logs(b) for a,b in zip(peT,pbT)]):+.2f} nats/event")
