"""
Stage-A causal engine  ·  Argentina country-risk SCM  ·  Mineral-X × TTC
========================================================================
Implements the DAG from Causal_Engine_Spec.docx Figure 1 as a runnable
structural causal model, so the dashboards' stub-do(·) sliders can be
replaced with genuine intervention semantics.

Scope (Stage A, per spec §5): DAG + downstream mechanisms + ELICITED
priors for the R->lever activation coefficients. Stage B replaces the
elicited priors with cross-country panel estimates; the interface here
(regime state + logP -> lever probabilities -> mechanism shocks -> NPV)
does not change when that happens.

Three Pearl rungs, all via the same structural equations:
  rung 1  observational   E[NPV]                     -> npv_observational()
  rung 2  interventional  E[NPV | do(lever=v)]       -> npv_do()
  rung 3  counterfactual  "had RIGI existed in 2019" -> counterfactual_rigi()

TIMING SEMANTICS (stated, not hidden): panel activations are 5-yr-window
probabilities. The engine converts them to a per-year hazard,
h = 1 - (1 - p5)^(1/5), samples a FIRST-PASSAGE firing year for each lever
(geometric), and applies each mechanism from its firing year onward. do()
pins firing at year 1. A single regime is drawn per trial and held for the
horizon (Stage-A simplification; the HMM supplies transitions in Stage B).
A fired mandate (L) adds its plant capital in the firing year and, as a
documented simplification, does not interrupt production.

STATUS: proof-of-concept engine. Structural COEFFICIENTS are elicited
placeholders, clearly tabled below and meant for Jef's review, NOT panel
estimates. Every magnitude that drives a number is in ELICITED_PRIORS.
NPV uses a compact stub DCF (ArrowHead-compatible p(t),q(t) interface).
"""

import numpy as np
from scipy.stats import norm, truncnorm

# --------------------------------------------------------------------------
# 0. DAG  (declared explicitly so graph surgery is auditable, not implicit)
# --------------------------------------------------------------------------
# Parents of each node. R is latent; P is exogenous observed. do() removes
# a node's incoming edges (graph surgery) and pins its value.
DAG_PARENTS = {
    "R":  [],                 # latent regime, HMM prior (Stage A: yearly prior)
    "P":  [],                 # world Li price, exogenous (price-taker)
    "E":  ["R", "P"],         # export restrictions
    "N":  ["R", "P"],         # nationalization pressure
    "L":  ["R", "P"],         # local value-added mandate
    "T":  ["R", "P"],         # royalty/tax change (level + jump)
    "NPV": ["E", "N", "L", "T", "P", "RIGI"],   # via 4 mechanisms + revenue
}

REGIMES = ["orthodox", "interventionist", "crisis"]

# --------------------------------------------------------------------------
# 1. ELICITED PRIORS  (Stage A stand-in for panel estimates — REVIEW THESE)
# --------------------------------------------------------------------------
# Probit activation:  lever = 1[ alpha_R[state] + beta * (logP - logP_ref) - tau > eps_neg ]
# equivalently P(lever=1) = Phi( alpha_R[state] + beta*(logP-logP_ref) - tau ).
# alpha_R shifts the latent index by regime; beta is the resource-nationalism
# boom effect (higher price -> more restriction). tau is the lever threshold.
ELICITED_PRIORS = {
    # regime prior for a "single evaluation year" (Stage A collapses the HMM
    # to a marginal; Stage B supplies the year-by-year transition matrix)
    "regime_prior": np.array([0.45, 0.40, 0.15]),   # orthodox / interv / crisis

    # per-lever latent-index intercept by regime  (rows align to REGIMES)
    #                 orthodox  interv  crisis
    "alpha_R": {
        "E": np.array([-1.60,  -0.25,   0.90]),
        "N": np.array([-2.30,  -1.10,   0.10]),   # nationalization: rarer
        "L": np.array([-1.20,   0.10,   0.70]),   # mandates: interventionist-driven
        "T": np.array([-1.00,   0.35,   1.10]),   # fiscal grabs: common in stress
    },
    "beta":  {"E": 0.55, "N": 0.30, "L": 0.40, "T": 0.45},   # boom effect on logP
    "tau":   {"E": 0.00, "N": 0.00, "L": 0.00, "T": 0.00},   # abs. into alpha_R

    "logP_ref": np.log(15000.0),   # ref benchmark Li2CO3 (USD/t), boom-effect pivot
}

# Preserve the elicited placeholders so the two can be compared side by side.
_PLACEHOLDER_ALPHA = {L: v.copy() for L, v in ELICITED_PRIORS["alpha_R"].items()}
_PLACEHOLDER_REGIME_PRIOR = ELICITED_PRIORS["regime_prior"].copy()

# Stage B: DEFAULT to the panel-estimated activation intercepts (coded cross-
# country lever-event panel; see activation_panel.py), falling back to the
# elicited placeholders above if the panel module is unavailable. This is the
# "make the estimates the engine default" wiring — the numbers now come from
# resource-nationalism history, not elicitation.
try:
    import activation_panel as _ap
    ELICITED_PRIORS["alpha_R"] = {L: _ap.ESTIMATED_ALPHA[L].copy() for L in _ap.LEVERS}
    ELICITED_PRIORS["regime_prior"] = _ap.ESTIMATED_REGIME_PRIOR.copy()
    ACTIVATION_SOURCE = "panel-estimated (activation_panel.py)"
except Exception as _e:                                        # pragma: no cover
    ACTIVATION_SOURCE = f"elicited-placeholder (panel unavailable: {_e})"

# --------------------------------------------------------------------------
# 2. MECHANISM MAGNITUDES  (lever -> DCF input shock; spec §2 rows 4-8)
# --------------------------------------------------------------------------
# Downstream mechanisms are estimable from project data alone (spec's point:
# "mechanism, not politics"), so Stage A treats these as tighter than the
# activation priors. Still elicited placeholders here.
MECH = {
    # EXTRACTED (E edge): net export-duty incidence on lithium-carbonate sales.
    # Source: Cauchari-Olaroz NI 43-101 Operational Technical Report, §22.3.2
    # (Lithium Americas (Argentina) Corp., SEC EDGAR): 4.31% duty - 1.44% national
    # incentive refund = 2.87% net. Statutory duty is currently 0 (Decree 563/2025,
    # Aug-2025), so this is the do(E=1) reimposition magnitude, not today's rate.
    "E_duty_incidence":   0.0287,  # [EXTRACTED] net export duty, frac of sales
    # DERIVED (L edge): forced carbonate->hydroxide conversion plant scope delta.
    # Anchor: Naraha LiOH plant (Allkem-TTC JV, Olaroz feed) US$77.6M gross (ex-VAT)
    # for 10 ktpa battery-grade hydroxide -> ~$7,760/t. Matched-throughput plant
    # scaled to project size (0.65 cost-capacity exponent) vs upstream carbonate
    # capex: ~0.49x at Olaroz-S1 scale ($229M/17.5kt), ~0.34x at Cauchari scale
    # ($565M/40kt). Central 0.42 (Argentina siting premium vs scale economies).
    # NOTE: this is the MILDEST mandate (hydroxide). A cathode/precursor or cell
    # mandate would push the multiplier > 1.0.
    "L_capex_multiplier": 0.42,   # [DERIVED] +42% CapEx (hydroxide-conversion mandate)
    "L_schedule_delay_y": 1.5,    # [DERIVED] Naraha build ~2yr, partly parallel -> ~1.5yr
    # ELICITED (N edge, prior-dominated by design — sparse events, no Argentine
    # lithium precedent). Anchors: YPF 2012 expropriation (Repsol claimed US$10.5B,
    # settled US$5.0B in discounted bonds -> ~50%+ haircut on fair value); Bolivia
    # YLB / Mexico LitioMex (state control, minimal compensation). Premium anchored
    # to Argentina's calm->crisis EMBI+ swing (~400 -> ~1900-2800 bps) and to
    # 3-8pp practitioner expropriation-risk premia. Regime-conditional; labeled.
    "N_risk_premium_pp":  0.05,   # [ELICITED] +5pp on discount rate if N fires
    "N_exit_haircut":     0.45,   # [ELICITED] ~45% exit-value haircut (YPF-anchored)
    # T edge: incremental fiscal grab if the tax/royalty lever fires. Anchored to
    # real statutory headroom (royalty 1.6% effective -> 3% cap = +1.4pp; plus a
    # plausible new levy). RIGI's 30-yr stability BLOCKS this path; RIGI also cuts
    # CIT 35%->25% (a level benefit, handled in the DCF, not here).
    "T_extra_take":       0.03,   # [elicited, cap-anchored] extra take, frac revenue
}

# --------------------------------------------------------------------------
# 3. STUB DCF  (ArrowHead-compatible; same economics family as the dashboards)
# --------------------------------------------------------------------------
# Fiscal rates below are EXTRACTED from the Cauchari-Olaroz NI 43-101 Operational
# Technical Report §22.3 (Lithium Americas (Argentina) Corp., SEC EDGAR) unless
# marked. royalty 1.6% effective (2-3% statutory, 3% cap under Law 24.196);
# debit/credit tax 0.996% effective; CIT 35% standard / 25% under RIGI.
DCF = {
    "capex_base":     229.0,   # USD MM (Olaroz Stage-1 anchor; see reconciliation note)
    "opex_per_t":     6800.0,  # USD/t LCE (all-in; 2024 TR revised ~US$6,543/t)
    "prod_peak_tpa":  25000.0, # t LCE/yr at plateau
    "ramp_years":     3,       # linear ramp to plateau
    "life_years":     20,
    "base_royalty":   0.016,   # [EXTRACTED] effective provincial royalty (frac sales)
    "debit_credit":   0.00996, # [EXTRACTED] net debit/credit tax (frac sales)
    "cit_standard":   0.35,    # [EXTRACTED] corporate income tax, standard regime
    "cit_rigi":       0.25,    # [EXTRACTED] corporate income tax under RIGI
    "wacc":           0.10,
    "price_usd_t":    14000.0, # flat p(t) stub; ArrowHead replaces with a path
}

# per-draw noise, so NPV is continuous (fixes the deterministic-spike tail).
# CapEx multiplier = the Olaroz reference-class posterior (P10/P90 = 0.87/1.97,
# handoff §5). OpEx lognormal sigma=0.18. Demand-CAGR jitter is open-item #1:
# a per-draw price drift so within-scenario price risk enters CVaR.
NOISE = {
    "capex_p10": 0.87, "capex_p90": 1.97,   # -> lognormal for RC multiplier
    "opex_sigma": 0.18,
    "cagr_sigma": 0.015,                     # demand-CAGR jitter, ~1.5pp (slider)
}
_CAPEX_MU = np.log(np.sqrt(NOISE["capex_p10"] * NOISE["capex_p90"]))   # ln(P50)
_CAPEX_SIG = np.log(NOISE["capex_p90"] / NOISE["capex_p10"]) / (2 * 1.2816)

def draw_dcf_noise(rng, cagr_sigma=None):
    """Sample the cost- and price-side noise for one Monte-Carlo trial.
    Returned dict is held FIXED across a counterfactual's factual/CF pair
    (abduction: noise-term stability)."""
    cs = NOISE["cagr_sigma"] if cagr_sigma is None else cagr_sigma
    return {
        "capex_mult": float(np.exp(rng.normal(_CAPEX_MU, _CAPEX_SIG))),
        "opex_mult":  float(np.exp(rng.normal(0.0, NOISE["opex_sigma"]))),
        "cagr":       float(rng.normal(0.0, cs)),   # per-draw price drift
    }

def _npv(price0, capex, opex_per_t, export_duty, extra_take, cit_rate,
         disc, exit_haircut, delay_y, cagr):
    """Compact after-tax DCF. Returns NPV in USD MM. price0 grows at `cagr`/yr.
    Revenue-based charges: royalty + debit/credit tax + export duty (E-gated) +
    incremental grab (T-gated). Then CIT on positive pre-tax profit."""
    d = DCF
    years = int(d["life_years"] + round(delay_y))
    ramp = d["ramp_years"]
    rev_rate = d["base_royalty"] + d["debit_credit"] + export_duty + extra_take
    cfs = [-capex]  # t=0 outflow
    for y in range(1, years + 1):
        producing_year = y - round(delay_y)
        if producing_year <= 0:
            cfs.append(0.0)
            continue
        price = price0 * (1 + cagr) ** producing_year          # demand-CAGR jitter
        q = d["prod_peak_tpa"] * min(1.0, producing_year / ramp) / 1e6  # MM t
        revenue = price * q                        # USD MM (price USD/t * MM t)
        opex = opex_per_t * q                      # USD MM
        pretax = revenue - opex - rev_rate * revenue
        tax = cit_rate * max(0.0, pretax)          # CIT on positive profit only
        cfs.append(pretax - tax)
    # terminal/exit value: small multiple of last CF, haircut applied
    cfs[-1] += cfs[-1] * 3.0 * (1.0 - exit_haircut)
    npv = sum(cf / (1 + disc) ** t for t, cf in enumerate(cfs))
    return npv

def _npv_timed(price0, capex0, opex_per_t, fire, rigi_cit, rigi_block,
               cagr, capex_mult):
    """After-tax DCF with lever FIRING YEARS. `fire` maps lever -> year (1-based)
    or None. Mechanisms apply from the firing year onward. Returns NPV, USD MM."""
    d = DCF
    years = d["life_years"]; ramp = d["ramp_years"]
    cit = d["cit_rigi"] if rigi_cit else d["cit_standard"]
    prem_from = fire.get("N")
    haircut = MECH["N_exit_haircut"] if fire.get("N") else 0.0
    cfs = [-capex0]
    for y in range(1, years + 1):
        price = price0 * (1 + cagr) ** y
        q = d["prod_peak_tpa"] * min(1.0, y / ramp) / 1e6
        revenue = price * q
        rate = d["base_royalty"] + d["debit_credit"]
        if fire.get("E") and y >= fire["E"]:
            rate += MECH["E_duty_incidence"]
        if fire.get("T") and y >= fire["T"] and not rigi_block:
            rate += MECH["T_extra_take"]
        pretax = revenue - opex_per_t * q - rate * revenue
        cf = pretax - cit * max(0.0, pretax)
        if fire.get("L") and y == fire["L"]:
            cf -= MECH["L_capex_multiplier"] * DCF["capex_base"] * capex_mult
        cfs.append(cf)
    cfs[-1] += cfs[-1] * 3.0 * (1.0 - haircut)
    # discount with an N premium from the firing year onward (step in the rate)
    npv, df = 0.0, 1.0
    for t, cf in enumerate(cfs):
        if t > 0:
            r = d["wacc"] + (MECH["N_risk_premium_pp"] if (prem_from and t >= prem_from) else 0.0)
            df *= (1.0 + r)
        npv += cf / df
    return npv

def hazard_from_window(p5, horizon=5):
    """Per-year hazard from a 5-yr-window activation probability."""
    return 1.0 - (1.0 - min(max(p5, 1e-6), 1 - 1e-6)) ** (1.0 / horizon)

# --------------------------------------------------------------------------
# 4. THE SCM
# --------------------------------------------------------------------------
class ArgentinaSCM:
    def __init__(self, rng=None, logP=None):
        self.rng = rng or np.random.default_rng(7)
        self.logP = ELICITED_PRIORS["logP_ref"] if logP is None else logP

    # ---- structural equation for a lever (probit activation) --------------
    def _lever_index(self, lever, regime_idx):
        p = ELICITED_PRIORS
        return (p["alpha_R"][lever][regime_idx]
                + p["beta"][lever] * (self.logP - p["logP_ref"])
                - p["tau"][lever])

    def lever_prob(self, lever, regime_idx):
        return norm.cdf(self._lever_index(lever, regime_idx))

    def _sample_lever(self, lever, regime_idx, eps=None):
        """lever = 1[index > eps_neg]; eps ~ N(0,1). Pass eps to fix the noise
        (needed for the counterfactual abduction step)."""
        idx = self._lever_index(lever, regime_idx)
        if eps is None:
            eps = self.rng.standard_normal()
        return int(idx > eps), eps

    def _sample_regime(self):
        return int(self.rng.choice(3, p=ELICITED_PRIORS["regime_prior"]))

    # ---- mechanisms: levers -> DCF inputs ---------------------------------
    def _npv_from_levers(self, E, N, L, T, rigi, noise):
        d = DCF
        export_duty = MECH["E_duty_incidence"] * E                        # E: capture (duty)
        capex = d["capex_base"] * noise["capex_mult"] * (1 + MECH["L_capex_multiplier"] * L)  # L + RC
        delay = MECH["L_schedule_delay_y"] * L                            # L: schedule
        disc  = d["wacc"] + MECH["N_risk_premium_pp"] * N                  # N: premium
        haircut = MECH["N_exit_haircut"] * N                              # N: exit
        extra_take = MECH["T_extra_take"] * T * (0 if rigi else 1)        # T: fiscal (RIGI blocks)
        cit = d["cit_rigi"] if rigi else d["cit_standard"]                # RIGI: 25% vs 35% CIT
        return _npv(d["price_usd_t"], capex, d["opex_per_t"] * noise["opex_mult"],
                    export_duty, extra_take, cit, disc, haircut, delay, noise["cagr"])

    # ---- first-passage firing year (per-year hazard from the 5-yr window) ----
    def _sample_firing_year(self, lever, regime_idx):
        p5 = self.lever_prob(lever, regime_idx)
        h = hazard_from_window(p5)
        for y in range(1, DCF["life_years"] + 1):
            if self.rng.random() < h:
                return y
        return None

    # ---- one forward sample of the whole model ----------------------------
    def _forward(self, do=None, rigi=False, regime_idx=None):
        """One draw. `do` pins levers (graph surgery: cut R->lev, P->lev) with
        firing at year 1. Otherwise each lever gets a first-passage year from
        its per-year hazard, or never fires."""
        do = do or {}
        r = self._sample_regime() if regime_idx is None else regime_idx
        fire = {}
        for lev in ["E", "N", "L", "T"]:
            if lev in do:
                fire[lev] = 1 if do[lev] else None
            else:
                fire[lev] = self._sample_firing_year(lev, r)
        noise = draw_dcf_noise(self.rng)
        capex0 = DCF["capex_base"] * noise["capex_mult"]
        npv = _npv_timed(DCF["price_usd_t"], capex0,
                         DCF["opex_per_t"] * noise["opex_mult"], fire,
                         rigi_cit=rigi, rigi_block=rigi, cagr=noise["cagr"],
                         capex_mult=noise["capex_mult"])
        return npv, fire, r

    # ---- rung 1: observational --------------------------------------------
    def npv_observational(self, n=20000, rigi=False):
        s = np.array([self._forward(rigi=rigi)[0] for _ in range(n)])
        return s

    # ---- rung 2: atomic intervention  do(lever=v) -------------------------
    def npv_do(self, interventions, n=20000, rigi=False):
        s = np.array([self._forward(do=interventions, rigi=rigi)[0] for _ in range(n)])
        return s

    # ---- rung 3: counterfactual (abduction -> action -> prediction) -------
    def counterfactual_rigi(self, observed_year_regime="crisis", n=20000):
        """'What would NPV have been had RIGI existed in 2019?'
        Factual: crisis regime, and the fiscal package fired (T=1, E=1, year 1).
        Abduction on the pinned binary levers is vacuous (their noise cannot
        change the pinned value), so no lever-noise abduction is performed.
        The operative abduction holds fixed, across the factual/counterfactual
        pair: the DCF noise draw and the sampled N and L firing years.
        Returns (factual, cf_full, cf_block_only, cf_cit_only): the full RIGI
        counterfactual and its decomposition into the blocked fiscal path and
        the 35%->25% CIT cut."""
        r = REGIMES.index(observed_year_regime)
        fact = np.empty(n); cf_full = np.empty(n)
        cf_blk = np.empty(n); cf_cit = np.empty(n)
        for i in range(n):
            yN = self._sample_firing_year("N", r)
            yL = self._sample_firing_year("L", r)
            noise = draw_dcf_noise(self.rng)   # HELD FIXED across the pair
            capex0 = DCF["capex_base"] * noise["capex_mult"]
            opx = DCF["opex_per_t"] * noise["opex_mult"]
            fire = {"E": 1, "T": 1, "N": yN, "L": yL}
            args = dict(price0=DCF["price_usd_t"], capex0=capex0, opex_per_t=opx,
                        fire=fire, cagr=noise["cagr"], capex_mult=noise["capex_mult"])
            fact[i]    = _npv_timed(rigi_cit=False, rigi_block=False, **args)
            cf_full[i] = _npv_timed(rigi_cit=True,  rigi_block=True,  **args)
            cf_blk[i]  = _npv_timed(rigi_cit=False, rigi_block=True,  **args)
            cf_cit[i]  = _npv_timed(rigi_cit=True,  rigi_block=False, **args)
        return fact, cf_full, cf_blk, cf_cit

# --------------------------------------------------------------------------
# 5. DEMONSTRATION
# --------------------------------------------------------------------------
def _pctl(s):
    return (np.percentile(s, 10), np.percentile(s, 50), np.percentile(s, 90))

def _cvar(s, q=0.05):
    v = np.percentile(s, q * 100)
    return s[s <= v].mean()

def main():
    scm = ArgentinaSCM(rng=np.random.default_rng(7))

    print("=" * 74)
    print("STAGE-A CAUSAL ENGINE — Argentina country-risk SCM  (demonstration)")
    print("=" * 74)

    print(f"\nactivation source: {ACTIVATION_SOURCE}")

    print("\n[ activation probabilities by regime, at reference price ]")
    print(f"  {'lever':<6}" + "".join(f"{r:>16}" for r in REGIMES))
    for lev in ["E", "N", "L", "T"]:
        row = "".join(f"{scm.lever_prob(lev, i):>16.3f}" for i in range(3))
        print(f"  {lev:<6}{row}")

    # ---- placeholder vs estimated: effect on the observational baseline ----
    def _baseline():
        b = ArgentinaSCM(rng=np.random.default_rng(7)).npv_observational()
        return b.mean(), (b > 0).mean(), _pctl(b)[0]
    est = _baseline()
    _sa, _sp = ELICITED_PRIORS["alpha_R"], ELICITED_PRIORS["regime_prior"]
    ELICITED_PRIORS["alpha_R"], ELICITED_PRIORS["regime_prior"] = _PLACEHOLDER_ALPHA, _PLACEHOLDER_REGIME_PRIOR
    plc = _baseline()
    ELICITED_PRIORS["alpha_R"], ELICITED_PRIORS["regime_prior"] = _sa, _sp
    print("\n[ baseline shift: elicited placeholder -> panel-estimated activation ]")
    print(f"  placeholder :  E[NPV]={plc[0]:7.1f}  P(NPV>0)={plc[1]:.2f}  P10={plc[2]:7.1f}")
    print(f"  estimated   :  E[NPV]={est[0]:7.1f}  P(NPV>0)={est[1]:.2f}  P10={est[2]:7.1f}")
    print("  (estimated activations are less pessimistic on orthodox/interventionist L,T)")

    # ---- rung 1 ----
    base = scm.npv_observational()
    p10, p50, p90 = _pctl(base)
    print("\n[ rung 1 · observational  E[NPV] ]  (USD MM)")
    print(f"  E[NPV]={base.mean():7.1f}   P10/50/90 = {p10:6.1f}/{p50:6.1f}/{p90:6.1f}"
          f"   P(NPV>0)={ (base>0).mean():.2f}   CVaR5={_cvar(base):7.1f}")

    # ---- rung 2 · atomic interventions ----
    print("\n[ rung 2 · atomic interventions  do(lever=1)  vs  observational ]")
    for lev in ["L", "E", "T", "N"]:
        d = scm.npv_do({lev: 1})
        print(f"  do({lev}=1):  E[NPV]={d.mean():7.1f}   "
              f"ATE = {d.mean()-base.mean():+7.1f}   CVaR5={_cvar(d):7.1f}")
    # combined mandate+duty (correlated in reality via R — see rung-2b)
    combo = scm.npv_do({"L": 1, "E": 1})
    print(f"  do(L=1,E=1): E[NPV]={combo.mean():7.1f}   "
          f"ATE = {combo.mean()-base.mean():+7.1f}   CVaR5={_cvar(combo):7.1f}")
    # forward fiscal intervention: secure RIGI (25% vs 35% CIT + T-path stability)
    rigi = scm.npv_observational(rigi=True)
    print(f"  secure RIGI: E[NPV]={rigi.mean():7.1f}   "
          f"ATE = {rigi.mean()-base.mean():+7.1f}   CVaR5={_cvar(rigi):7.1f}   (25% CIT + stability)")

    # ---- rung 2b · the common-cause correction (spec's key point) ----
    print("\n[ common-cause check · joint-through-R  vs  independent levers ]")
    # independent: each lever draws its OWN regime (severs the shared cause),
    # then first-passage timing exactly as in the joint model.
    def _indep_draw():
        fire = {lev: scm._sample_firing_year(lev, scm._sample_regime())
                for lev in ["E", "N", "L", "T"]}
        noise = draw_dcf_noise(scm.rng)
        return _npv_timed(DCF["price_usd_t"], DCF["capex_base"]*noise["capex_mult"],
                          DCF["opex_per_t"]*noise["opex_mult"], fire,
                          rigi_cit=False, rigi_block=False,
                          cagr=noise["cagr"], capex_mult=noise["capex_mult"])
    indep = np.array([_indep_draw() for _ in range(20000)])
    print(f"  joint-through-R   CVaR5 = {_cvar(base):7.1f}   P10 = {_pctl(base)[0]:7.1f}")
    print(f"  independent       CVaR5 = {_cvar(indep):7.1f}   P10 = {_pctl(indep)[0]:7.1f}")
    print(f"  -> independence understates tail loss by "
          f"{_cvar(indep)-_cvar(base):+.1f} MM  (cost-side RC risk dominates this tail)")
    # co-firing within the horizon: one regime for all four vs one regime each
    def _p_atleast(k, joint):
        hits = 0
        for _ in range(30000):
            if joint:
                r = scm._sample_regime()
                fired = sum(scm._sample_firing_year(l, r) is not None
                            for l in ["E", "N", "L", "T"])
            else:
                fired = sum(scm._sample_firing_year(l, scm._sample_regime()) is not None
                            for l in ["E", "N", "L", "T"])
            hits += fired >= k
        return hits / 30000
    p3_j, p3_i = _p_atleast(3, True), _p_atleast(3, False)
    p4_j, p4_i = _p_atleast(4, True), _p_atleast(4, False)
    print(f"  P(>=3 levers fire in horizon): joint = {p3_j:.3f}  indep = {p3_i:.3f}  ({p3_j/max(p3_i,1e-9):.1f}x)")
    print(f"  P( all 4 fire in horizon)    : joint = {p4_j:.3f}  indep = {p4_i:.3f}  ({p4_j/max(p4_i,1e-9):.1f}x)")
    print("  -> co-firing is where geopolitics bites; independent levers miss it.")

    # ---- rung 3 · counterfactual ----
    fact, cf_full, cf_blk, cf_cit = scm.counterfactual_rigi("crisis")
    print("\n[ rung 3 · counterfactual · 'Cauchari NPV had RIGI existed in 2019' ]")
    print(f"  factual (no RIGI, crisis, duty package fired):   E[NPV]={fact.mean():7.1f}")
    print(f"  counterfactual, full RIGI:                       E[NPV]={cf_full.mean():7.1f}"
          f"   delta = {(cf_full-fact).mean():+6.1f} MM")
    print(f"    of which fiscal-path blocker alone:            delta = {(cf_blk-fact).mean():+6.1f} MM")
    print(f"    of which 35%->25% CIT cut alone:               delta = {(cf_cit-fact).mean():+6.1f} MM")
    print("  NOTE: least-robust tier (noise-term stability assumed). Label as such.")
    print("=" * 74)

if __name__ == "__main__":
    main()
