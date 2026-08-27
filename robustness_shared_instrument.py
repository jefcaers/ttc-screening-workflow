"""Robustness: collapse shared-instrument event pairs to one event and re-estimate.
Each flagged instrument was coded as two levers (an export-restriction E row and a
local-content L row -- or, for Argentina 2018, an E row and a tax T row). We drop
the SECONDARY facet of each shared instrument and re-run the hazard estimation."""
import numpy as np
import activation_panel as ap

# (country, year, lever) rows to DROP -- the secondary facet of a shared instrument.
DROP = {
    ("Argentina", 2018, "T"),   # Decreto 793/2018 also coded E (export duty = the same decree)
    ("Chile",     2023, "L"),   # National Lithium Strategy also coded N (state control); royalty T is a separate law
    ("Ghana",     2023, "L"),   # Green Minerals Policy also coded E
    ("Namibia",   2023, "L"),   # June-2023 cabinet directive also coded E
    ("Zimbabwe",  2022, "L"),   # SI 213/2022 also coded E
}

BASE_EVENTS = list(ap.EVENTS)
COLLAPSED   = [e for e in BASE_EVENTS if (e[0], e[1], e[2]) not in DROP]

def run(events):
    ap.EVENTS = events
    exp, counts, p_hor, alpha = ap.estimate()
    rb, rn, mult = ap.boom_effect()
    tot_per_lever = {L: sum(counts[r][L] for r in range(3)) for L in ap.LEVERS}
    return dict(n=len(events), p_hor=p_hor, alpha=alpha, boom=(rb, rn, mult),
                per_lever=tot_per_lever, counts=counts)

base = run(BASE_EVENTS)
coll = run(COLLAPSED)
ap.EVENTS = BASE_EVENTS  # restore

L = ap.LEVERS; RN = ap.REGIME_NAME
print("="*72)
print("SHARED-INSTRUMENT ROBUSTNESS  (baseline 41 events  ->  collapsed %d)" % coll["n"])
print("="*72)
print("Dropped (secondary facet of a shared instrument):")
for c,y,l in sorted(DROP): print(f"   - {c} {y} [{l}]")

print("\nEvents per lever (E export, N nationalization, L local-content, T tax):")
print(f"   {'lever':<6}{'baseline':>10}{'collapsed':>11}")
for l in L:
    print(f"   {l:<6}{base['per_lever'][l]:>10}{coll['per_lever'][l]:>11}")

print("\nP(lever within 5-yr window | regime)  baseline -> collapsed:")
for r in range(3):
    print(f"  {RN[r]}:")
    for j,l in enumerate(L):
        b=base['p_hor'][r,j]; c=coll['p_hor'][r,j]
        flag = "   <-- change" if abs(b-c)>0.005 else ""
        print(f"     {l}:  {b:0.3f} -> {c:0.3f}  (d={c-b:+0.3f}){flag}")

print("\nProbit intercept alpha, largest shifts:")
diffs=[]
for r in range(3):
    for j,l in enumerate(L):
        diffs.append((abs(coll['alpha'][r,j]-base['alpha'][r,j]), RN[r], l,
                      base['alpha'][r,j], coll['alpha'][r,j]))
for d,rn,l,b,c in sorted(diffs, reverse=True)[:6]:
    print(f"   {rn:<15} {l}:  {b:+0.2f} -> {c:+0.2f}  (|d|={d:0.2f})")

rb,rn_,m=base['boom']; rb2,rn2,m2=coll['boom']
print(f"\nPrice-boom multiplier (beta):  baseline {m:0.2f}x  ->  collapsed {m2:0.2f}x")
print(f"   (boom rate {rb:0.3f}->{rb2:0.3f} /cy;  normal {rn_:0.3f}->{rn2:0.3f} /cy)")
print("="*72)
