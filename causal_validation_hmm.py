"""Re-run the held-out validation with HMM soft regime beliefs.
The Stage-A model scored held-out events under a HARD, retrospectively-coded regime
label and LOST to a base rate. Here the activation probability is belief-weighted:
  P(lever fires) = sum_r  belief_r(country, freeze_year) * P(fire | regime r)
using the HMM's PREDICTED (one-step-ahead, filtered) belief at the freeze year, which
carries the transition the static label missed. Same events, same frozen alphas, same
base rate and placebo, so the comparison is clean."""
import numpy as np, sys
_os = __import__('os')
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import activation_panel as ap
import hmm_regime as hr
import importlib; importlib.reload(hr)

LEV=ap.LEVERS; REG=ap.REGIME_NAME

def frozen_phor(cutoff, exclude_country):
    reg={}
    for c,periods in ap.REGIME_PERIODS.items():
        if c==exclude_country: continue
        for y0,y1,r in periods:
            for y in range(max(y0,ap.YEAR0),min(y1,cutoff)+1): reg[(c,y)]=r
    exp=np.zeros(3); cnt=np.zeros((3,4))
    for r in reg.values(): exp[r]+=1
    for c,y,L,_ in ap.EVENTS:
        if y<=cutoff and c!=exclude_country and (c,y) in reg: cnt[reg[(c,y)],LEV.index(L)]+=1
    p=(cnt+0.5)/(exp[:,None]+20.5); return 1-(1-p)**5

def base_rate(cutoff,lever,excl):
    ny=sum(min(y1,cutoff)-max(y0,ap.YEAR0)+1 for c,ps in ap.REGIME_PERIODS.items()
           if c!=excl for y0,y1,r in ps)
    k=sum(1 for c,y,L,_ in ap.EVENTS if y<=cutoff and c!=excl and L==lever)
    return 1-(1-(k+0.5)/(ny+20.5))**5

def logs(p): return np.log(max(min(p,1-1e-6),1e-6))

print("="*82)
print("HELD-OUT VALIDATION, RE-RUN WITH HMM SOFT REGIME BELIEFS")
print("="*82)
m=hr._label_states(hr.train())

TESTS=[("T1 freeze 2018 -> Argentina 2019",2018,"Argentina",[("T",2019)]),
       ("T2 freeze 2021 -> Chile 2023",2021,"Chile",[("N",2023),("T",2023),("L",2023)])]

rows=[]
for title,cut,ctry,fired in TESTS:
    phor=frozen_phor(cut,ctry)
    belief=m["predicted"][(ctry,cut)]                 # HMM one-step-ahead belief
    hard=ap.regime_of(ctry,cut)
    print(f"\n{title}")
    print(f"  static label at freeze: {REG[hard]}   |   HMM belief: "
          f"ortho {belief[0]:.2f} interv {belief[1]:.2f} crisis {belief[2]:.2f}")
    for lev,yr in fired:
        j=LEV.index(lev)
        p_static=phor[hard,j]
        p_hmm=float(belief@phor[:,j])
        p_base=base_rate(cut,lev,ctry)
        print(f"    {lev} {yr}:  static {p_static:.3f}  HMM {p_hmm:.3f}  base {p_base:.3f}"
              f"   | HMM lift vs base x{p_hmm/p_base:.2f}   vs static x{p_hmm/max(p_static,1e-6):.2f}")
        rows.append((p_static,p_hmm,p_base))
ps,ph,pb=map(np.array,zip(*rows))
print("\n"+"="*82)
print("AGGREGATE (4 held-out events)")
print(f"  mean prob:   static {ps.mean():.3f}   HMM {ph.mean():.3f}   base {pb.mean():.3f}")
print(f"  HMM beats base on {int((ph>pb).sum())}/4   (Stage-A static beat base on 0/4)")
print(f"  HMM beats static on {int((ph>ps).sum())}/4")
print(f"  mean log-score gain over base: static {np.mean([logs(a)-logs(b) for a,b in zip(ps,pb)]):+.2f}"
      f"   HMM {np.mean([logs(a)-logs(b) for a,b in zip(ph,pb)]):+.2f} nats/event")
print("\nNegative control (Australia orthodox, N):")
au=m["predicted"][("Australia",2018)]; ph_au=float(au@frozen_phor(2018,"Australia")[:,LEV.index("N")])
print(f"  HMM P(N)={ph_au:.3f}  (should stay low; Australia had no nationalization)")
print("\nStill 4 held-out events: directional, pre-registered, with placebo+control.")
