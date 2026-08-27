import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, numpy as np
from contextlib import redirect_stdout
src=open(_os.path.join(_HERE,"backtest_validation.py")).read()
g={}; buf=io.StringIO()
with redirect_stdout(buf): exec(src,g)
rows=g["rows"]; rng=np.random.default_rng(5)
rns=np.array([r["rn"] for r in rows]); verd=[r["sys"]["verdict"] for r in rows]
def loss(vlist,es,chi):
    E={"enter":1.0,"stage":es,"walk":0.0}; L=0.0
    for v,npv in zip(vlist,rns):
        e=E[v]; L+=e*max(0,-npv)+(1-e)*chi*max(0,npv)
    return L
print("Sensitivity of the verdict-loss result to the two assumed weights")
print("cells: system loss | %% of always-enter->oracle improvement captured | beats all 3 baselines?")
print()
hdr="stage exp \\ chi |"+ "".join("   %.2f        "%c for c in [0.25,0.5,0.75,1.0])
print(hdr); print("-"*len(hdr))
worst=100; best=0
for es in [0.1,0.2,0.3,0.4,0.5]:
    cells=[]
    for chi in [0.25,0.5,0.75,1.0]:
        Ls=loss(verd,es,chi); Le=loss(["enter"]*10,es,chi); Lw=loss(["walk"]*10,es,chi)
        Lo=loss(["enter" if x>0 else "walk" for x in rns],es,chi)
        Lr=np.mean([loss(rng.choice(["enter","stage","walk"],10),es,chi) for _ in range(1000)])
        pct=100*(Le-Ls)/max(1e-9,Le-Lo)
        beats=Ls<min(Le,Lw,Lr)
        worst=min(worst,pct); best=max(best,pct)
        cells.append("%5.0f |%3.0f%% |%s" % (Ls,pct,"Y" if beats else "N"))
    print("   %.1f          |"%es + "  ".join(cells))
print()
print("captured-improvement range across the whole grid: %.0f%% to %.0f%%"%(worst,best))
# where does the system stop beating always-walk? scan chi upward at es=0.3
print()
print("crossover scan (stage exp = 0.3): chi at which always-walk becomes competitive with the system")
for chi in np.arange(0.1,1.05,0.05):
    Ls=loss(verd,0.3,chi); Lw=loss(["walk"]*10,0.3,chi)
    if Lw<Ls: print("   always-walk overtakes system at chi = %.2f (walk %0.f vs system %.0f)"%(chi,Lw,Ls)); break
else: print("   system beats always-walk for all chi in [0.10, 1.00]")
for chi in np.arange(1.0,0.05,-0.05):
    Ls=loss(verd,0.3,chi); Le=loss(["enter"]*10,0.3,chi)
    if Le<Ls: print("   always-ENTER overtakes system at chi = %.2f"%chi); break
else: print("   system beats always-enter for all chi in [0.10, 1.00]")
