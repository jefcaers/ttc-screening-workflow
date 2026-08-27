"""
Activation panel (Stage-B, first pass)  ·  Argentina country-risk SCM
=====================================================================
Replaces the ELICITED placeholder lever-activation intercepts (ALPHA) in
stage_a_engine.py with estimates grounded in ACTUAL resource-nationalism
history: a coded cross-country lever-event panel with regime-tagged
exposure, from which regime-conditional activation probabilities are
derived and converted to probit intercepts.

Exposes at module level (consumed by stage_a_engine as the default):
    ESTIMATED_ALPHA          {lever: np.array([orth, intv, cris])}
    ESTIMATED_REGIME_PRIOR   np.array([orth, intv, cris]) exposure shares

Honest scope: regimes are an OBSERVABLE classification (documented political
economy), cross-checked below against the real Freedom House / V-Dem panel
(pulled from Our World in Data). This is NOT yet the latent-regime HMM; the
FH cross-check shows exactly which dimension the hand tags capture (crisis-
as-autocratization) and which they don't (crisis-as-macro-collapse), which
is what motivates adding IMF macro indicators in the full model.
"""

import numpy as np
from scipy.stats import norm

ORTH, INTV, CRIS = 0, 1, 2
REGIME_NAME = ["orthodox", "interventionist", "crisis"]
LEVERS = ["E", "N", "L", "T"]
YEAR0, YEAR1 = 2007, 2026

# --------------------------------------------------------------------------
# 1. REGIME PERIODS (observable classification; documented political economy)
# --------------------------------------------------------------------------
REGIME_PERIODS = {
    "Argentina": [(2007,2015,INTV),(2016,2017,ORTH),(2018,2019,CRIS),(2020,2023,INTV),(2024,2026,ORTH)],
    "Chile":     [(2007,2021,ORTH),(2022,2026,INTV)],
    "Bolivia":   [(2007,2026,INTV)],
    "Mexico":    [(2007,2018,ORTH),(2019,2026,INTV)],
    "Indonesia": [(2007,2026,INTV)],
    "Zimbabwe":  [(2007,2009,CRIS),(2010,2026,INTV)],
    "Namibia":   [(2007,2026,ORTH)],
    "Ghana":     [(2007,2026,ORTH)],
    "Nigeria":   [(2007,2026,INTV)],
    "Zambia":    [(2007,2026,INTV)],
    "DRC":       [(2007,2026,CRIS)],
    "Mali":      [(2007,2020,ORTH),(2021,2026,CRIS)],
    "Niger":     [(2007,2022,ORTH),(2023,2026,CRIS)],
    "Guinea":    [(2007,2020,ORTH),(2021,2026,CRIS)],
    "Venezuela": [(2007,2026,CRIS)],
    "Peru":      [(2007,2026,ORTH)],
    "Australia": [(2007,2026,ORTH)],
    "Canada":    [(2007,2026,ORTH)],
}

# --------------------------------------------------------------------------
# 2. CODED EVENT PANEL  (country, year, lever, instrument)
# --------------------------------------------------------------------------
EVENTS = [
    ("Argentina",2012,"N","YPF 51% expropriation (Repsol)"),
    ("Argentina",2018,"E","export duties on mining (DNU 793/2018)"),
    ("Argentina",2018,"T","export-duty package"),
    ("Argentina",2019,"T","Solidarity Law duty hikes"),
    ("Chile",2023,"N","National Lithium Strategy: state control (Codelco/SQM)"),
    ("Chile",2023,"T","mining royalty law"),
    ("Chile",2023,"L","domestic value-add via state JV"),
    ("Bolivia",2007,"N","ICSID withdrawal"),
    ("Bolivia",2008,"N","hydrocarbon/mining nationalizations"),
    ("Bolivia",2012,"N","BIT abrogation; sector state control"),
    ("Bolivia",2017,"L","YLB lithium state monopoly"),
    ("Mexico",2022,"N","lithium declared national property (LitioMex)"),
    ("Mexico",2023,"N","LitioMex concession revocations"),
    ("Indonesia",2009,"L","Law 4/2009 domestic-processing requirement"),
    ("Indonesia",2014,"E","nickel ore export ban (partial)"),
    ("Indonesia",2020,"E","nickel ore export ban (full)"),
    ("Indonesia",2023,"E","bauxite export ban"),
    ("Zimbabwe",2022,"E","SI 213/2022 raw-lithium export ban"),
    ("Zimbabwe",2022,"L","beneficiation mandate"),
    ("Zimbabwe",2023,"E","SI 5/2023 base-minerals extension"),
    ("Zimbabwe",2025,"T","10% lithium-concentrate export tax"),
    ("Zimbabwe",2025,"L","concentrate export ban announced (2027)"),
    ("Zimbabwe",2026,"E","raw-mineral/concentrate export suspension"),
    ("Namibia",2023,"E","unprocessed critical-minerals export ban"),
    ("Namibia",2023,"L","domestic-refining requirement"),
    ("Ghana",2023,"E","Green Minerals Policy raw-lithium ban"),
    ("Ghana",2023,"L","local-processing mandate"),
    ("Nigeria",2022,"E","raw-ore export ban"),
    ("Nigeria",2022,"L","local-processing push"),
    ("Zambia",2008,"T","windfall tax"),
    ("Zambia",2019,"N","KCM/Vedanta forced liquidation"),
    ("Zambia",2019,"T","mineral royalty increase"),
    ("Mali",2023,"T","new mining code; state stakes"),
    ("Mali",2024,"N","Barrick/Resolute detentions; asset seizure"),
    ("Niger",2024,"N","Orano uranium permit revocation"),
    ("Guinea",2025,"N","mining-licence revocations"),
    ("DRC",2025,"E","cobalt export suspension"),
    ("Venezuela",2011,"N","gold-sector nationalization"),
    ("Australia",2012,"T","Minerals Resource Rent Tax (MRRT)"),
    ("Chile",2010,"T","royalty increase (post-earthquake reconstruction)"),
    ("Peru",2011,"T","gravamen especial a la mineria"),
]

BOOM_YEARS = set(range(2021, 2024)) | {2010, 2011}

# --------------------------------------------------------------------------
# 3. EXTERNAL INDICATOR: Freedom House regime (0 not free, 1 partly, 2 free)
#    A-M pulled live from Our World in Data (Freedom House / V-Dem panel,
#    political-regime-fh, retrieved 2026); N-Z from FH published classes.
#    Encoded compactly as (y0, y1, score) because FH scores are persistent.
# --------------------------------------------------------------------------
FH_REGIME = {
    "Argentina": [(2007,2026,2)],                                   # live
    "Australia": [(2007,2026,2)],                                   # live
    "Bolivia":   [(2007,2024,1),(2025,2026,2)],                     # live
    "Canada":    [(2007,2026,2)],                                   # live
    "Chile":     [(2007,2026,2)],                                   # live
    "DRC":       [(2007,2026,0)],                                   # live
    "Ghana":     [(2007,2026,2)],                                   # live
    "Guinea":    [(2007,2009,0),(2010,2020,1),(2021,2026,0)],       # live
    "Indonesia": [(2007,2012,2),(2013,2026,1)],                     # live
    "Mali":      [(2007,2011,2),(2012,2012,0),(2013,2019,1),(2020,2026,0)],  # live
    "Mexico":    [(2007,2026,1)],                                   # FH class
    "Namibia":   [(2007,2026,2)],                                   # FH class
    "Niger":     [(2007,2022,1),(2023,2026,0)],                     # FH class
    "Nigeria":   [(2007,2026,1)],                                   # FH class
    "Peru":      [(2007,2021,2),(2022,2026,1)],                     # FH class
    "Venezuela": [(2007,2016,1),(2017,2026,0)],                     # FH class
    "Zambia":    [(2007,2026,1)],                                   # FH class
    "Zimbabwe":  [(2007,2026,0)],                                   # FH class
}

# --------------------------------------------------------------------------
# 4. ESTIMATION
# --------------------------------------------------------------------------
def regime_of(country, year):
    for y0, y1, r in REGIME_PERIODS[country]:
        if y0 <= year <= y1:
            return r
    return None

def fh_of(country, year):
    for y0, y1, s in FH_REGIME.get(country, []):
        if y0 <= year <= y1:
            return s
    return None

def exposure_by_regime(exclude=None):
    exclude = exclude or set()
    exp = np.zeros(3)
    for c, periods in REGIME_PERIODS.items():
        if c in exclude: continue
        for y0, y1, r in periods:
            exp[r] += (min(y1, YEAR1) - max(y0, YEAR0) + 1)
    return exp

def event_counts(exclude=None):
    exclude = exclude or set()
    counts = {r: {L: 0 for L in LEVERS} for r in range(3)}
    seen = set()
    for c, y, L, _ in EVENTS:
        if c in exclude: continue
        r = regime_of(c, y)
        key = (c, y, L)
        if r is not None and key not in seen:
            seen.add(key); counts[r][L] += 1
    return counts

def estimate(exclude=None, a0=0.5, b0=20.0, horizon=5):
    exp = exposure_by_regime(exclude); counts = event_counts(exclude)
    p_hor = np.zeros((3, 4)); alpha = np.zeros((3, 4))
    for r in range(3):
        for j, L in enumerate(LEVERS):
            k, n = counts[r][L], exp[r]
            p = (k + a0) / (n + a0 + b0)
            ph = 1 - (1 - p) ** horizon
            p_hor[r, j] = ph
            alpha[r, j] = norm.ppf(min(max(ph, 1e-4), 1 - 1e-4))
    return exp, counts, p_hor, alpha

def boom_effect():
    eb = en = 0
    for c, periods in REGIME_PERIODS.items():
        for y0, y1, r in periods:
            for y in range(max(y0, YEAR0), min(y1, YEAR1) + 1):
                if y in BOOM_YEARS: eb += 1
                else: en += 1
    evb = sum(1 for c, y, L, _ in EVENTS if y in BOOM_YEARS)
    evn = sum(1 for c, y, L, _ in EVENTS if y not in BOOM_YEARS)
    rb, rn = evb / max(eb, 1), evn / max(en, 1)
    return rb, rn, rb / max(rn, 1e-9)

# module-level estimates (the drop-in for stage_a_engine) --------------------
_exp, _counts, _phor, _ALPHA = estimate()
ESTIMATED_ALPHA = {L: _ALPHA[:, i].copy() for i, L in enumerate(LEVERS)}
# Panel exposure shares answer "how common is each regime among producers".
# They are NOT the probability of Argentina's regime. Export both; the engine
# must use the country-specific prior for a country-specific evaluation.
PANEL_EXPOSURE_SHARES = _exp / _exp.sum()
def country_regime_prior(country):
    """Own-history regime shares for one country (Stage-A stand-in for the HMM)."""
    sh = np.zeros(3)
    for y0, y1, r in REGIME_PERIODS[country]:
        sh[r] += (min(y1, YEAR1) - max(y0, YEAR0) + 1)
    return sh / sh.sum()
ESTIMATED_REGIME_PRIOR = country_regime_prior("Argentina")

# --------------------------------------------------------------------------
# 5. ROBUSTNESS
# --------------------------------------------------------------------------
def leave_one_country_out():
    """Per (regime,lever) 5-yr prob: spread when each country is dropped.
    Large spread => that cell is driven by one country (fragile)."""
    base = estimate()[2]
    cells = {}
    for c in REGIME_PERIODS:
        ph = estimate(exclude={c})[2]
        for r in range(3):
            for j, L in enumerate(LEVERS):
                cells.setdefault((r, j), []).append(ph[r, j])
    spread = {}
    for (r, j), vals in cells.items():
        spread[(r, j)] = (base[r, j], min(vals), max(vals))
    return spread

def prior_sweep(b0s=(10, 20, 40)):
    return {b0: estimate(b0=b0)[2] for b0 in b0s}

def validate_against_fh():
    """Cross-tabulate hand regime tags vs Freedom House score over country-years."""
    tab = {r: {0: 0, 1: 0, 2: 0} for r in range(3)}
    mismatches = []
    for c, periods in REGIME_PERIODS.items():
        for y0, y1, r in periods:
            for y in range(max(y0, YEAR0), min(y1, YEAR1) + 1):
                s = fh_of(c, y)
                if s is None: continue
                tab[r][s] += 1
                if r == CRIS and s == 2:            # crisis tag but "free"
                    mismatches.append((c, y))
    return tab, sorted(set((c,) for c, y in mismatches))

# --------------------------------------------------------------------------
# 6. REPORT
# --------------------------------------------------------------------------
def main():
    exp, counts, p_hor, alpha = estimate()
    print("=" * 78)
    print("ACTIVATION PANEL — regime-conditional lever activation (first-pass)")
    print("=" * 78)
    print(f"\nPanel: {len(REGIME_PERIODS)} countries x {YEAR0}-{YEAR1} = "
          f"{int(exp.sum())} country-years; {len(EVENTS)} coded events.")

    print("\n[ estimated P(lever within 5-yr window | regime) ]")
    print(f"  {'regime':<16}" + "".join(f"{L:>8}" for L in LEVERS))
    for r in range(3):
        print(f"  {REGIME_NAME[r]:<16}" + "".join(f"{p_hor[r,j]:>8.2f}" for j in range(4)))

    print("\n[ implied probit intercepts alpha (drop-in default for the engine) ]")
    for j, L in enumerate(LEVERS):
        print(f'    "{L}": [{", ".join(f"{alpha[r,j]:+.2f}" for r in range(3))}]')

    rb, rn, mult = boom_effect()
    print(f"\n[ price-boom effect (beta) ]  boom={rb:.3f}/cy  normal={rn:.3f}/cy  "
          f"-> {mult:.1f}x more likely in booms")

    print("\n[ robustness: leave-one-country-out 5-yr-prob spread (fragile cells) ]")
    spread = leave_one_country_out()
    flagged = sorted(spread.items(), key=lambda kv: kv[1][2]-kv[1][1], reverse=True)[:5]
    for (r, j), (base, lo, hi) in flagged:
        print(f"  {REGIME_NAME[r]:<15} {LEVERS[j]}:  base={base:.2f}  range=[{lo:.2f},{hi:.2f}]"
              f"  span={hi-lo:.2f}")

    print("\n[ robustness: prior-strength (b0) sweep on N activation ]")
    sweep = prior_sweep()
    print(f"  {'b0':>4}" + "".join(f"{REGIME_NAME[r][:5]:>8}" for r in range(3)))
    for b0, ph in sweep.items():
        jN = LEVERS.index("N")
        print(f"  {b0:>4}" + "".join(f"{ph[r,jN]:>8.2f}" for r in range(3)))

    print("\n[ external validation: hand regime tag vs Freedom House score ]")
    tab, cris_free = validate_against_fh()
    print(f"  {'tag':<16}{'NF(0)':>7}{'PF(1)':>7}{'Free(2)':>8}   (country-years)")
    for r in range(3):
        print(f"  {REGIME_NAME[r]:<16}{tab[r][0]:>7}{tab[r][1]:>7}{tab[r][2]:>8}")
    crisis_nf = tab[CRIS][0] / max(sum(tab[CRIS].values()), 1)
    print(f"  -> crisis tags that FH also flags 'not free': {crisis_nf:.0%}")
    print(f"  -> crisis tags FH calls 'free' (economic, not political, crisis): "
          f"{[c for (c,) in cris_free]}")
    print("     FH validates crisis-as-autocratization (coups) but misses macro")
    print("     crises like Argentina 2018 -> full latent regime needs IMF macro too.")
    print("=" * 78)

if __name__ == "__main__":
    main()
