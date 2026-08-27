import os as _os
"""
Indicator panel for the latent-regime HMM (Stage B).

WHAT THIS IS. For each of the 18 producer countries x 20 years (2007-2026) we code
a small block of OBSERVABLE indicators that a regime emits. The HMM (built next, in
hmm_regime.py) infers a belief distribution over regimes from these indicators; it
must NOT be handed the regime labels. This module therefore produces indicators only.

ANTI-CIRCULARITY (the load-bearing discipline). The held-out validation showed the
static regime label is the problem. If we code indicators by peeking at the labels,
the HMM just relearns the labels and the test is meaningless. So:
  - Indicators are coded from macro/political FACTS (deficits, reserves, elections,
    resource-nationalism rhetoric), NOT from the REGIME_PERIODS table.
  - A leakage check (bottom) reports how well indicators alone separate the coded
    regimes. We WANT moderate, not perfect, separation. Perfect separation would mean
    we cheated; near-zero would mean the indicators carry no signal.

CONFIDENCE. These are structured expert codings on documented 0-2 / z-scored scales,
medium confidence, meant to be REPLACED cell-by-cell with V-Dem / ICRG / IMF WEO /
World Bank series in the curation pass. The register at the bottom lists what each
indicator maps to in those sources so the swap is mechanical.

INDICATORS (all observable at year end, none derived from the regime label):
  fisc   fiscal balance stress   0 surplus/low deficit .. 2 large/twin deficit
  fx     FX/reserve pressure     0 stable .. 2 controls/reserve crisis
  infl   inflation stress        0 <5% .. 2 >30%/accelerating
  elec   election/leadership     0 none .. 1 election yr .. 2 populist/outsider win
  rhet   resource-nationalism    0 none .. 1 rising rhetoric .. 2 active measures
  cpol   coalition/ideology tilt 0 market-orthodox .. 2 statist/left-populist
"""
import numpy as np
COUNTRIES = ['Argentina','Chile','Bolivia','Mexico','Indonesia','Zimbabwe','Namibia',
             'Ghana','Nigeria','Zambia','DRC','Mali','Niger','Guinea','Venezuela',
             'Peru','Australia','Canada']
YEAR0, YEAR1 = 2007, 2026
IND = ["fisc","fx","infl","elec","rhet","cpol"]

# ---- compact coder: per country, a list of (year_from, year_to, [fisc,fx,infl,elec,rhet,cpol]) ----
# Values are documented-history codings. Provenance tier per country in REGISTER below.
_SEG = {
 "Argentina":[  # macro crises 2012-14, 2018-19 (reserves/IMF), Kirchner->Macri->Fernandez->Milei
   (2007,2011,[1,1,2,1,1,2]),(2012,2013,[2,2,2,0,2,2]),(2014,2015,[2,2,2,1,1,2]),
   (2016,2017,[1,1,1,1,0,0]),(2018,2019,[2,2,2,1,1,1]),(2020,2023,[2,2,2,1,1,2]),
   (2024,2026,[1,1,2,2,0,0])],
 "Chile":[  # orthodox, then 2019 unrest -> 2021 convention -> Boric 2022
   (2007,2018,[0,0,0,1,0,0]),(2019,2019,[1,0,0,0,1,1]),(2020,2021,[1,1,1,1,1,1]),
   (2022,2023,[1,1,1,2,2,2]),(2024,2026,[1,0,1,0,1,1])],
 "Bolivia":[ (2007,2018,[1,1,1,1,2,2]),(2019,2020,[1,1,1,2,1,1]),(2021,2026,[1,2,1,1,2,2]) ],
 "Mexico":[ (2007,2017,[1,0,0,1,0,0]),(2018,2018,[1,0,0,2,1,1]),(2019,2026,[1,1,1,0,2,2]) ],
 "Indonesia":[ (2007,2013,[1,1,1,1,1,1]),(2014,2014,[1,1,1,2,2,1]),(2015,2019,[1,1,1,1,2,1]),
   (2020,2026,[1,1,1,1,2,2]) ],
 "Zimbabwe":[ (2007,2008,[2,2,2,1,2,2]),(2009,2016,[2,2,1,1,2,2]),(2017,2017,[2,2,2,2,2,2]),
   (2018,2026,[2,2,2,1,2,2]) ],
 "Namibia":[ (2007,2026,[1,0,0,1,1,1]) ],
 "Ghana":[ (2007,2013,[1,1,1,1,0,0]),(2014,2015,[2,2,2,1,1,1]),(2016,2021,[1,1,1,1,1,1]),
   (2022,2026,[2,2,2,1,1,1]) ],
 "Nigeria":[ (2007,2014,[1,1,1,1,1,1]),(2015,2016,[2,2,2,2,1,1]),(2017,2026,[2,2,2,1,1,1]) ],
 "Zambia":[ (2007,2010,[1,1,1,1,1,1]),(2011,2014,[1,1,1,2,2,2]),(2015,2020,[2,2,2,1,2,2]),
   (2021,2026,[1,1,2,2,1,1]) ],
 "DRC":[ (2007,2017,[2,1,1,1,2,2]),(2018,2018,[2,1,1,2,2,2]),(2019,2026,[1,1,1,1,2,2]) ],
 "Mali":[ (2007,2011,[1,1,1,1,1,1]),(2012,2012,[2,2,1,2,1,2]),(2013,2019,[2,1,1,1,1,1]),
   (2020,2026,[2,1,1,2,2,2]) ],
 "Niger":[ (2007,2009,[1,1,1,1,1,1]),(2010,2010,[2,1,1,2,2,2]),(2011,2022,[1,1,1,1,1,1]),
   (2023,2026,[2,2,1,2,2,2]) ],
 "Guinea":[ (2007,2020,[1,1,1,1,1,1]),(2021,2021,[2,1,1,2,2,2]),(2022,2026,[1,1,1,1,2,2]) ],
 "Venezuela":[ (2007,2012,[2,1,2,1,2,2]),(2013,2016,[2,2,2,2,2,2]),(2017,2026,[2,2,2,1,2,2]) ],
 "Peru":[ (2007,2010,[0,0,1,1,1,0]),(2011,2011,[0,0,1,2,2,1]),(2012,2020,[0,0,1,1,1,0]),
   (2021,2022,[1,1,2,2,2,2]),(2023,2026,[1,1,1,0,1,1]) ],
 "Australia":[ (2007,2009,[0,0,1,1,0,0]),(2010,2013,[0,0,1,2,1,1]),(2014,2026,[0,0,0,1,0,0]) ],
 "Canada":[ (2007,2026,[0,0,0,1,0,0]) ],
}

def panel():
    """Return X[(country,year)] = np.array of 6 indicators."""
    X = {}
    for c in COUNTRIES:
        for y0,y1,vec in _SEG[c]:
            for y in range(y0,y1+1):
                X[(c,y)] = np.array(vec,dtype=float)
    # fill any gaps defensively with country median
    for c in COUNTRIES:
        yrs=[y for (cc,y) in X if cc==c]
        for y in range(YEAR0,YEAR1+1):
            if (c,y) not in X:
                X[(c,y)] = np.median([X[(c,yy)] for yy in yrs],axis=0)
    return X

def matrix():
    """Stacked (n, 6) array + index list, z-scored per indicator for the HMM."""
    X=panel(); keys=sorted(X); M=np.array([X[k] for k in keys])
    Mz=(M-M.mean(0))/(M.std(0)+1e-9)
    return Mz, keys, M

# ------------------------------------------------------------------ leakage check
def anticircularity_report():
    """How separable are the HAND-CODED regimes from indicators ALONE?
    We want moderate accuracy. Perfect => indicators were copied from labels."""
    import sys; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import activation_panel as ap
    Mz,keys,M = matrix()
    y=np.array([ap.regime_of(c,yr) for (c,yr) in keys])
    # nearest-centroid classifier accuracy (crude separability proxy)
    cents=np.array([Mz[y==r].mean(0) for r in range(3)])
    pred=np.array([np.argmin(((Mz[i]-cents)**2).sum(1)) for i in range(len(Mz))])
    acc=(pred==y).mean()
    # per-indicator correlation with an orthodox(0)->crisis(2) regime scale
    corr={IND[j]:float(np.corrcoef(M[:,j],y)[0,1]) for j in range(6)}
    return acc, corr, y

if __name__=="__main__":
    Mz,keys,M=matrix()
    print(f"indicator panel: {len(keys)} country-years x {len(IND)} indicators")
    print("indicator means:",dict(zip(IND,np.round(M.mean(0),2))))
    acc,corr,y=anticircularity_report()
    print(f"\nANTI-CIRCULARITY CHECK")
    print(f"  nearest-centroid regime accuracy from indicators alone: {acc:.2f}")
    print(f"  (want ~0.55-0.80: signal present but not a copy of the labels)")
    print("  per-indicator corr with orthodox->crisis scale:")
    for k,v in corr.items(): print(f"    {k:5s} {v:+.2f}")
    print(f"  regime mix in panel: orthodox {int((y==0).sum())}, interventionist {int((y==1).sum())}, crisis {int((y==2).sum())}")

# ------------------------------------------------------------------ curation register
REGISTER = """
CURATION MAP (replace codings with real series, cell by cell):
  fisc  -> IMF WEO general-government net lending/borrowing %GDP (z-score, sign-flip)
  fx    -> World Bank/IMF reserves-in-months-imports + a capital-control dummy (Fernandez-Villaverde/ICRG)
  infl  -> IMF WEO CPI inflation %
  elec  -> national election calendar (year dummy) + outsider/populist-win flag (V-Party)
  rhet  -> resource-nationalism measures index (OECD export-restriction inventory + coded speeches)
  cpol  -> V-Dem economic-left / statism scale or V-Party econ-left
Provenance tiers by country (this draft): A(rich public record) Argentina, Chile, Peru,
  Australia, Canada, Indonesia, Venezuela; B Bolivia, Mexico, Ghana, Nigeria, Zambia,
  Zimbabwe; C (thin, verify) DRC, Mali, Niger, Guinea, Namibia.
"""
