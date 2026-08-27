"""Held-out validation of the causal activation model.
The spec's validation design: freeze the model on data before a cutoff, then score
the probability it assigned to policy-lever events that fired AFTER the cutoff and
that it therefore never saw. This is the causal analog of the cost module's
leakage-controlled back-test. Two frozen tests:
  T1: freeze end-2018 -> score Argentina's 2019 Solidarity-Law duty package (T fired)
  T2: freeze end-2021 -> score Chile's 2023 national-lithium / royalty package (N,T,L)
Scoring: the model's 5-year-window activation probability for the (regime, lever) of
each held-out event, versus (a) a naive base rate that ignores regime and (b) the
same model with regimes shuffled (a placebo). A model that has learned the regime
signal should score the fired events HIGHER than the base rate and than the placebo,
and should do so out of sample.
Small-n honesty: a handful of held-out events cannot 'prove' calibration; this is a
directional, pre-registered check with an explicit placebo, reported as such."""
import numpy as np, sys
_os = __import__('os')
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import activation_panel as ap
rng = np.random.default_rng(5)

REG = ap.REGIME_NAME; LEV = ap.LEVERS

def frozen_estimate(cutoff, exclude_country=None, shuffle=False):
    """Re-derive activation probabilities using ONLY country-years and events with
    year <= cutoff. Optionally drop a country (to prevent the target country's own
    pre-cutoff events from informing its held-out score), or shuffle regime labels
    (placebo). Returns p_hor[regime,lever] (5-yr window probabilities)."""
    # exposure: count country-years <= cutoff, by regime
    exp = np.zeros(3); counts = np.zeros((3, 4))
    reg_lookup = {}
    for c, periods in ap.REGIME_PERIODS.items():
        if c == exclude_country: continue
        for y0, y1, r in periods:
            for y in range(max(y0, ap.YEAR0), min(y1, cutoff) + 1):
                reg_lookup[(c, y)] = r
    regs = list(reg_lookup.values())
    if shuffle:
        vals = rng.permutation(regs)
        reg_lookup = {k: int(v) for k, v in zip(reg_lookup, vals)}
    for r in reg_lookup.values(): exp[r] += 1
    for c, y, L, _ in ap.EVENTS:
        if y > cutoff or c == exclude_country: continue
        if (c, y) in reg_lookup:
            counts[reg_lookup[(c, y)], LEV.index(L)] += 1
    a0, b0, H = 0.5, 20.0, 5
    p = (counts + a0) / (exp[:, None] + a0 + b0)
    return 1 - (1 - p) ** H, exp, counts

def base_rate(cutoff, lever, exclude_country=None):
    """Regime-blind 5-yr activation base rate for a lever (ignores regime signal)."""
    ny = 0; k = 0
    for c, periods in ap.REGIME_PERIODS.items():
        if c == exclude_country: continue
        for y0, y1, r in periods:
            ny += min(y1, cutoff) - max(y0, ap.YEAR0) + 1
    for c, y, L, _ in ap.EVENTS:
        if y <= cutoff and c != exclude_country and L == lever: k += 1
    p = (k + 0.5) / (ny + 20.5)
    return 1 - (1 - p) ** 5

def logscore(p): return np.log(max(min(p, 1 - 1e-6), 1e-6))

print("=" * 84)
print("HELD-OUT VALIDATION OF THE CAUSAL ACTIVATION MODEL")
print("=" * 84)

TESTS = [
    ("T1: freeze end-2018 -> Argentina 2019 duty package",
     2018, "Argentina", [("T", 2019, "Solidarity-Law export-duty hikes")]),
    ("T2: freeze end-2021 -> Chile 2023 lithium/royalty package",
     2021, "Chile", [("N", 2023, "National Lithium Strategy (state control)"),
                       ("T", 2023, "mining royalty law"),
                       ("L", 2023, "domestic value-add via state JV")]),
]

allrows = []
for title, cutoff, country, fired in TESTS:
    print("\n" + title)
    reg = ap.regime_of(country, cutoff)   # the regime the country was in at the wall
    print(f"  {country} regime at end-{cutoff}: {REG[reg]}")
    # model frozen on <=cutoff, EXCLUDING the target country's own events (strict)
    phor, exp, cnts = frozen_estimate(cutoff, exclude_country=country)
    # placebo: same freeze, regime labels shuffled, averaged over draws
    plc = np.mean([frozen_estimate(cutoff, exclude_country=country, shuffle=True)[0]
                   for _ in range(400)], axis=0)
    for lev, yr, desc in fired:
        j = LEV.index(lev)
        p_model = phor[reg, j]
        p_base = base_rate(cutoff, lev, exclude_country=country)
        p_plc = plc[reg, j]
        lift_base = p_model / p_base
        lift_plc = p_model / p_plc
        print(f"    fired: {lev} in {yr} ({desc})")
        print(f"      model P(fire in 5yr | {REG[reg]})   = {p_model:.3f}")
        print(f"      regime-blind base rate               = {p_base:.3f}   (model lift x{lift_base:.2f})")
        print(f"      regime-shuffled placebo              = {p_plc:.3f}   (model lift x{lift_plc:.2f})")
        print(f"      log-score gain over base             = {logscore(p_model)-logscore(p_base):+.2f} nats")
        allrows.append((country, lev, yr, p_model, p_base, p_plc))

# ---- aggregate ----
print("\n" + "=" * 84)
print("AGGREGATE OVER ALL HELD-OUT EVENTS")
print("=" * 84)
pm = np.array([r[3] for r in allrows]); pb = np.array([r[4] for r in allrows]); pp = np.array([r[5] for r in allrows])
print(f"  held-out events scored: {len(allrows)}")
print(f"  mean model prob {pm.mean():.3f}  vs base {pb.mean():.3f}  vs placebo {pp.mean():.3f}")
print(f"  model beats base rate on {int((pm>pb).sum())}/{len(allrows)} events")
print(f"  model beats placebo    on {int((pm>pp).sum())}/{len(allrows)} events")
print(f"  mean log-score gain over base: {np.mean([logscore(a)-logscore(b) for a,b in zip(pm,pb)]):+.2f} nats/event")
print(f"  mean log-score gain over placebo: {np.mean([logscore(a)-logscore(b) for a,b in zip(pm,pp)]):+.2f} nats/event")

# ---- negative control: a lever that DID NOT fire in an orthodox country ----
print("\n" + "-" * 84)
print("NEGATIVE CONTROL: nationalization in orthodox Australia, freeze end-2018")
phor, _, _ = frozen_estimate(2018, exclude_country="Australia")
r_au = ap.regime_of("Australia", 2018)
print(f"  Australia regime: {REG[r_au]};  model P(N in 5yr) = {phor[r_au, LEV.index('N')]:.3f}")
print(f"  (Australia had no nationalization 2019-2024 -> low probability is correct.)")

print("\nSmall-n directional check with explicit placebo; 4 held-out events cannot")
print("establish calibration, only consistency with the regime signal. Report as such.")
