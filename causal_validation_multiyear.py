"""Held-out validation, RE-RUN WITH MULTI-YEAR PREDICTED BELIEF (HMM_STATUS next-step #1).

THE PRE-REGISTERED FIX (stated in HMM_STATUS.md before this run):
  "Integrate the activation window over the k-step belief path (filtered belief
   pushed k years through the transition matrix, for k=1..5) instead of one step.
   Success criterion: HMM beats base rate on >=2/4 without changing the frozen
   indicator codings."

WHAT CHANGES vs causal_validation_hmm.py (and what does not):
  CHANGED:  the belief object. The one-step run scored the whole 5-year window
            under a single belief vector (predicted at the freeze year, i.e.
            filtered(freeze-1) pushed one step). Here the window is scored year
            by year: belief about window-year k is filtered(freeze) @ A^k, and
            the 5-year activation probability compounds the per-year hazard
            under each year's own belief:
               P(fire in window) = 1 - prod_{k=1..5} (1 - b_k . p_yr[:,lever])
            This is the correct probabilistic object for a 5-year window: a
            2021 freeze no longer scores 2025 under a 2021-vintage belief.
  UNCHANGED: indicator codings (frozen), event set, freeze years, target-country
            exclusion, hazard smoothing (a0=0.5, b0=20), the 5-year horizon, the
            regime-blind base rate, placebo design, negative control.
  NOTE:     filtered(freeze) is the legitimate information set: the freeze is
            END-of-year, and every indicator is coded as observable at year end
            (indicator_panel.py line 23). The one-step run's use of
            predicted(freeze)=filtered(freeze-1)@A was one year STALER than the
            freeze allows; HMM_STATUS diagnosed exactly this.

HONESTY GUARDS in this file:
  - The success bar (>=2/4 vs base) was written down in HMM_STATUS before this
    script existed. No parameter is tuned here; there are no free knobs.
  - Robustness A re-fits the HMM on pre-cutoff years ONLY (params can't leak
    post-cutoff information). Robustness B sweeps the window construction.
  - md5 of the frozen inputs is printed so "no codings changed" is checkable.
"""
import numpy as np, sys, hashlib, io, contextlib
_os = __import__('os')
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import activation_panel as ap
import indicator_panel as ip
import hmm_regime as hr

rng = np.random.default_rng(5)
LEV = ap.LEVERS; REG = ap.REGIME_NAME
H = 5  # activation window, years (unchanged)

for f in ["indicator_panel.py", "activation_panel.py", "hmm_regime.py"]:
    print(f"frozen-input md5 {f}: {hashlib.md5(open(_os.path.join(_HERE,f),'rb').read()).hexdigest()}")

# ---------------------------------------------------------------- frozen hazards
def frozen_hazard(cutoff, exclude_country, shuffle=False):
    """Per-YEAR hazard p[regime,lever] from data <= cutoff, target country excluded.
    Identical counts+smoothing to frozen_phor in the prior runs; the only
    difference is we return the per-year hazard, not 1-(1-p)^5, because the
    multi-year construction compounds the window itself."""
    reg = {}
    for c, periods in ap.REGIME_PERIODS.items():
        if c == exclude_country: continue
        for y0, y1, r in periods:
            for y in range(max(y0, ap.YEAR0), min(y1, cutoff) + 1):
                reg[(c, y)] = r
    if shuffle:
        vals = rng.permutation(list(reg.values()))
        reg = {k: int(v) for k, v in zip(reg, vals)}
    exp = np.zeros(3); cnt = np.zeros((3, 4))
    for r in reg.values(): exp[r] += 1
    for c, y, L, _ in ap.EVENTS:
        if y <= cutoff and c != exclude_country and (c, y) in reg:
            cnt[reg[(c, y)], LEV.index(L)] += 1
    return (cnt + 0.5) / (exp[:, None] + 20.5)

def base_rate(cutoff, lever, excl):
    ny = sum(min(y1, cutoff) - max(y0, ap.YEAR0) + 1
             for c, ps in ap.REGIME_PERIODS.items() if c != excl for y0, y1, r in ps)
    k = sum(1 for c, y, L, _ in ap.EVENTS if y <= cutoff and c != excl and L == lever)
    return 1 - (1 - (k + 0.5) / (ny + 20.5)) ** H

def logs(p): return np.log(max(min(p, 1 - 1e-6), 1e-6))

# ------------------------------------------------------- multi-year window prob
def window_prob(belief0, A, p_yr, j, k_hi=H):
    """P(lever j fires in years 1..k_hi after the freeze), compounding the
    per-year hazard under the k-step-ahead belief b_k = belief0 @ A^k."""
    b = belief0.copy(); surv = 1.0
    for _ in range(k_hi):
        b = b @ A                      # push belief one more year ahead
        surv *= 1.0 - float(b @ p_yr[:, j])
    return 1.0 - surv

def onestep_prob(belief_pred, p_yr, j):
    """The PRIOR run's object, rebuilt from the same hazards for comparison:
    single (stale) belief vector applied to the whole 5-year window."""
    p1 = float(belief_pred @ p_yr[:, j])
    return 1 - (1 - p1) ** H

# ---------------------------------------------------- truncated re-fit (Rob. A)
def train_truncated(cutoff, n_outer=12):
    """Re-fit the HMM using ONLY years <= cutoff, so not even the fitted
    parameters (mu, var, A) can carry post-cutoff information. Same init, same
    EM, same variance floor (0.6) as hmm_regime.train — nothing re-tuned.
    (The z-scoring transform from ip.matrix() is a full-panel monotone scaling;
    noted, not re-derived, to keep the codings byte-identical.)"""
    Mz, keys, M = ip.matrix()
    T_end = cutoff - ip.YEAR0 + 1
    y0 = np.array([ap.regime_of(c, yr) for (c, yr) in keys])
    keep = np.array([yr <= cutoff for (c, yr) in keys])
    mu = np.array([Mz[keep & (y0 == k)].mean(0) for k in range(3)])
    var = np.array([Mz[keep & (y0 == k)].var(0) + 0.8 for k in range(3)])
    A = np.full((3, 3), 0.06); np.fill_diagonal(A, 0.88); A /= A.sum(1, keepdims=True)
    pi = np.array([0.5, 0.35, 0.15])
    seqs = {c: np.array([Mz[keys.index((c, y))] for y in range(ip.YEAR0, cutoff + 1)])
            for c in ip.COUNTRIES}
    for it in range(n_outer):
        num_mu = np.zeros((3, 6)); den = np.zeros(3); num_var = np.zeros((3, 6))
        Atot = np.zeros((3, 3))
        for c in ip.COUNTRIES:
            al, ga, pred, B = hr.fit_country(seqs[c], mu, var, A, pi)
            for k in range(3):
                num_mu[k] += (ga[:, k:k+1] * seqs[c]).sum(0); den[k] += ga[:, k].sum()
            for t in range(len(seqs[c]) - 1):
                xi = (al[t][:, None] * A * B[t+1][None, :]); xi /= xi.sum() + 1e-300
                Atot += xi
        mu = num_mu / den[:, None]
        for c in ip.COUNTRIES:
            al, ga, pred, B = hr.fit_country(seqs[c], mu, var, A, pi)
            for k in range(3):
                num_var[k] += (ga[:, k:k+1] * (seqs[c] - mu[k])**2).sum(0)
        var = np.maximum(num_var / den[:, None], 0.6)
        A = Atot / Atot.sum(1, keepdims=True)
    filt = {}
    for c in ip.COUNTRIES:
        al, ga, pred, B = hr.fit_country(seqs[c], mu, var, A, pi)
        for i, y in enumerate(range(ip.YEAR0, cutoff + 1)):
            filt[(c, y)] = al[i]
    m = dict(mu=mu, var=var, A=A, filtered=filt, smoothed=dict(filt), predicted=dict(filt))
    return hr._label_states(m)   # canonical ordering by stress, same rule

# ================================================================== MAIN RUN
print("\n" + "=" * 84)
print("HELD-OUT VALIDATION — MULTI-YEAR PREDICTED BELIEF (k=1..5 through A)")
print("=" * 84)
m = hr._label_states(hr.train())
A = m["A"]

TESTS = [("T1 freeze end-2018 -> Argentina 2019", 2018, "Argentina", [("T", 2019)]),
         ("T2 freeze end-2021 -> Chile 2023",     2021, "Chile",
          [("N", 2023), ("T", 2023), ("L", 2023)])]

rows = []
for title, cut, ctry, fired in TESTS:
    p_yr = frozen_hazard(cut, ctry)
    plc = np.mean([frozen_hazard(cut, ctry, shuffle=True) for _ in range(400)], axis=0)
    b_filt = m["filtered"][(ctry, cut)]
    b_pred = m["predicted"][(ctry, cut)]
    hard = ap.regime_of(ctry, cut)
    print(f"\n{title}")
    print(f"  static label at freeze: {REG[hard]}")
    print(f"  filtered belief at freeze (info <= end-{cut}): "
          f"ortho {b_filt[0]:.2f} interv {b_filt[1]:.2f} crisis {b_filt[2]:.2f}")
    bk = b_filt.copy()
    path = []
    for k in range(1, H + 1):
        bk = bk @ A; path.append(bk.copy())
    print("  k-step belief path (ortho/interv/crisis): " +
          "  ".join(f"y+{k}:[{b[0]:.2f},{b[1]:.2f},{b[2]:.2f}]" for k, b in enumerate(path, 1)))
    for lev, yr in fired:
        j = LEV.index(lev)
        p_static = 1 - (1 - p_yr[hard, j]) ** H
        p_one = onestep_prob(b_pred, p_yr, j)
        p_multi = window_prob(b_filt, A, p_yr, j)
        p_plc = window_prob(b_filt, A, plc, j)
        p_base = base_rate(cut, lev, ctry)
        beat = "BEATS base" if p_multi > p_base else "below base"
        print(f"    {lev} {yr}:  static {p_static:.3f}  1-step {p_one:.3f}  "
              f"MULTI {p_multi:.3f}  base {p_base:.3f}  placebo {p_plc:.3f}"
              f"   | multi vs base x{p_multi/p_base:.2f} [{beat}]  vs placebo x{p_multi/p_plc:.2f}")
        rows.append((p_static, p_one, p_multi, p_base, p_plc))

ps, p1, pm, pb, pp = map(np.array, zip(*rows))
n = len(rows)
print("\n" + "=" * 84)
print(f"AGGREGATE ({n} held-out events)")
print(f"  mean prob:  static {ps.mean():.3f}   1-step {p1.mean():.3f}   "
      f"MULTI {pm.mean():.3f}   base {pb.mean():.3f}   placebo {pp.mean():.3f}")
print(f"  MULTI beats base rate on {int((pm>pb).sum())}/{n}   "
      f"(1-step: {int((p1>pb).sum())}/{n};  static: {int((ps>pb).sum())}/{n})")
print(f"  MULTI beats placebo   on {int((pm>pp).sum())}/{n}")
print(f"  MULTI beats 1-step    on {int((pm>p1).sum())}/{n}")
print(f"  mean log-score vs base:  static {np.mean([logs(a)-logs(b) for a,b in zip(ps,pb)]):+.2f}"
      f"   1-step {np.mean([logs(a)-logs(b) for a,b in zip(p1,pb)]):+.2f}"
      f"   MULTI {np.mean([logs(a)-logs(b) for a,b in zip(pm,pb)]):+.2f} nats/event")
crit = int((pm > pb).sum()) >= 2
print(f"\n  PRE-REGISTERED SUCCESS CRITERION (>=2/{n} beat base): "
      f"{'PASS' if crit else 'FAIL'}")

# ------------------------------------------------------------- negative control
print("\nNegative control (Australia orthodox, N, freeze end-2018, same multi-year object):")
p_yr_au = frozen_hazard(2018, "Australia")
b_au = m["filtered"][("Australia", 2018)]
p_au = window_prob(b_au, A, p_yr_au, LEV.index("N"))
print(f"  multi-year P(N) = {p_au:.3f}  base = {base_rate(2018,'N','Australia'):.3f}"
      f"   (no nationalization occurred; must stay at-or-below base)")

# ============================================================ ROBUSTNESS A
print("\n" + "=" * 84)
print("ROBUSTNESS A — HMM re-fit on pre-cutoff years ONLY (no parameter leakage)")
print("=" * 84)
rowsA = []
for title, cut, ctry, fired in TESTS:
    mt = train_truncated(cut)
    p_yr = frozen_hazard(cut, ctry)
    b0 = mt["filtered"][(ctry, cut)]
    print(f"  {title}: truncated-fit filtered belief "
          f"ortho {b0[0]:.2f} interv {b0[1]:.2f} crisis {b0[2]:.2f}")
    for lev, yr in fired:
        j = LEV.index(lev)
        p_multi = window_prob(b0, mt["A"], p_yr, j)
        p_base = base_rate(cut, lev, ctry)
        print(f"    {lev} {yr}:  MULTI(trunc-fit) {p_multi:.3f}  base {p_base:.3f}"
              f"   x{p_multi/p_base:.2f} [{'BEATS' if p_multi>p_base else 'below'}]")
        rowsA.append((p_multi, p_base))
pmA, pbA = map(np.array, zip(*rowsA))
print(f"  -> truncated-fit MULTI beats base on {int((pmA>pbA).sum())}/{n};"
      f"  mean log-score vs base {np.mean([logs(a)-logs(b) for a,b in zip(pmA,pbA)]):+.2f} nats/event")
mtc = train_truncated(2018)
b_auA = mtc["filtered"][("Australia", 2018)]
p_auA = window_prob(b_auA, mtc["A"], frozen_hazard(2018, "Australia"), LEV.index("N"))
print(f"  negative control (trunc-fit): P(N Australia) = {p_auA:.3f}")

# ============================================================ ROBUSTNESS B
print("\n" + "=" * 84)
print("ROBUSTNESS B — window-length sweep (is k=5 doing the work, or the path?)")
print("=" * 84)
print("  P(fire in years 1..k) for each held-out event; base recomputed at same k:")
for title, cut, ctry, fired in TESTS:
    p_yr = frozen_hazard(cut, ctry)
    b_filt = m["filtered"][(ctry, cut)]
    for lev, yr in fired:
        j = LEV.index(lev)
        cells = []
        for k in range(1, H + 1):
            pmk = window_prob(b_filt, A, p_yr, j, k_hi=k)
            ny = sum(min(y1, cut) - max(y0, ap.YEAR0) + 1
                     for c, ps2 in ap.REGIME_PERIODS.items() if c != ctry for y0, y1, r in ps2)
            ke = sum(1 for c, y, L, _ in ap.EVENTS if y <= cut and c != ctry and L == lev)
            pbk = 1 - (1 - (ke + 0.5) / (ny + 20.5)) ** k
            cells.append(f"k={k}:{pmk:.3f}/{pbk:.3f}{'*' if pmk>pbk else ' '}")
        print(f"    {ctry} {lev} {yr}:  " + "  ".join(cells) + "   (model/base, * = beats)")

print("\nStill 4 held-out events: directional, pre-registered, with placebo + control.")
print("The bar and the construction were both written down (HMM_STATUS.md) before this run.")
