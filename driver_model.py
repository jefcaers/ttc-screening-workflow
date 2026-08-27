"""Conditional reference class: predict CapEx overrun from sanction-date observables.
Drivers (all codeable before FID, from public disclosures):
  HW    hardware novelty 0/1/2: count-based flag for unit operations whose delivering
        vendor had fewer than three prior commercial installations at similar scale
        ("vendor maturity rule" - the TTC-facing operationalization)
  CHEM  1 if the sanctioned scope includes an integrated on-site chemical plant
  REM   1 if site logistics are extreme (>3500 m altitude, >150 km new linear
        infrastructure, or fly-in remote)
  BOOM  1 if sanctioned inside the commodity's price boom (Li: 2016-18, 2021-22;
        Cu: 2010-12, 2021+) - contractor-market tightness and input escalation
Model: log(realized/sanctioned) = b0 + b.X, OLS with bootstrap CIs, LOO CV,
and a conditional-vs-unconditional band comparison under the SAME leakage rule
(each case predicted from coefficients fit on the other 20).
Driver codings are the authors' judgment from public project histories; flag med
confidence, curate before client use."""
import numpy as np
from scipy.stats import t as t_dist
rng = np.random.default_rng(3)

#            name              ratio  HW CHEM REM BOOM  note
D = [
 ("Olaroz '12",          1.75, 2,1,1,0, "first new brine pond+carbonation train in a decade"),
 ("MtCattlin '10",       1.30, 0,0,0,0, "conventional DMS/flotation, established vendors"),
 ("MtMarion '15",        1.15, 0,0,0,0, "conventional"),
 ("Pilgangoora '17",     1.17, 0,0,0,1, "conventional; boom-window sanction"),
 ("Altura '17",          1.30, 0,0,0,1, "conventional; boom"),
 ("Mibra '17",           1.05, 0,0,0,1, "brownfield restart, mature kit"),
 ("Nemaska '18",         1.45, 2,1,1,1, "bespoke electrochemical plant, no prior vendor install"),
 ("BaldHill '18",        1.25, 0,0,0,1, "conventional; boom"),
 ("Wodgina '18",         1.10, 0,0,0,1, "conventional; boom"),
 ("Cauchari '19",        1.73, 1,1,1,0, "second-generation brine train, one prior analog"),
 ("Caserones '10",       2.10, 0,0,1,1, "conventional SX-EW+conc; 4000m, new corridor"),
 ("MinistroHales '10",   1.30, 1,0,0,1, "high-arsenic roaster, specialized hardware"),
 ("SierraGorda '11",     1.40, 1,0,1,1, "raw-seawater processing at scale, moly circuit"),
 ("TenienteNML '11",     1.60, 1,0,1,1, "deep block-cave at unprecedented scale"),
 ("EscondidaOGP1 '12",   1.10, 0,0,0,1, "conventional concentrator"),
 ("ChuquiUG '12",        1.31, 1,0,0,1, "open-pit to mega block-cave conversion"),
 ("Antucoya '12",        1.12, 0,0,0,1, "conventional heap leach cathode"),
 ("Encuentro '15",       1.01, 0,0,0,0, "conventional oxides"),
 ("SpenceSGO '17",       1.02, 0,0,0,0, "conventional; near budget"),
 ("QB2 '18",             1.67, 0,0,1,0, "conventional flotation; 165km lines, coastal port, tailings"),
 ("RajoInca '19",        1.38, 0,0,0,0, "district conversion; execution-driven"),
]
names=[d[0] for d in D]; y=np.log([d[1] for d in D])
X=np.column_stack([np.ones(len(D))]+[[d[k] for d in D] for k in (2,3,4,5)]).astype(float)
labels=["intercept","HW novelty (per level)","CHEM plant scope","REM logistics","BOOM sanction"]

def ols(Xm,ym): return np.linalg.lstsq(Xm,ym,rcond=None)[0]
b=ols(X,y); resid=y-X@b; s2=resid@resid/(len(y)-X.shape[1])
# bootstrap CIs (case resampling)
bs=np.array([ols(X[i],y[i]) for i in (rng.integers(0,len(y),(4000,len(y))))])
lo,hi=np.percentile(bs,[5,95],axis=0)

print("="*88)
print("DRIVER MODEL  log(overrun) ~ drivers   (n=21, combined cohorts)")
print("="*88)
for k,l in enumerate(labels):
    star="*" if lo[k]*hi[k]>0 else " "
    print(f"  {l:26s}  b={b[k]:+.3f}  (90% CI {lo[k]:+.3f},{hi[k]:+.3f}) {star}"
          f"   multiplier x{np.exp(b[k]):.2f}")
print(f"  residual sd (log): {np.sqrt(s2):.3f}   R^2: {1-resid@resid/((y-y.mean())@(y-y.mean())):.2f}")

# ---- marginal effects (univariate): the "factors" view before adjustment ----
print("\nMARGINAL EFFECTS (mean overrun with flag vs without; unadjusted)")
for k,l in [(1,"HW novelty >=1"),(2,"CHEM plant scope"),(3,"REM logistics"),(4,"BOOM sanction")]:
    on=np.exp(y[X[:,k]>=1]).mean(); off=np.exp(y[X[:,k]==0]).mean()
    print(f"  {l:18s}  with x{on:.2f}   without x{off:.2f}   marginal multiplier x{on/off:.2f}"
          f"   (n_with={int((X[:,k]>=1).sum())})")
print("  NOTE: in this sample HW, CHEM and REM travel together (all big-overrun")
print("  projects carry two or three of them); n=21 cannot separate their shares.")

# ---- leakage-honest predictive test: leave-one-out ----
Z=1.2816
print("\nLEAVE-ONE-OUT: conditional prior vs unconditional reference class")
print(f"{'case':18s} {'ratio':>5s}  {'uncond band':>13s} in?  {'cond band':>13s} in?  width shrink")
u_cov=c_cov=0; shrinks=[]; u_ls=[]; c_ls=[]
mu_u=y.mean(); s_u=y.std(ddof=1)
for i in range(len(D)):
    m=np.ones(len(D),bool); m[i]=False
    bu_,su_=y[m].mean(), y[m].std(ddof=1)
    bi=ols(X[m],y[m]); ri=y[m]-X[m]@bi
    si=np.sqrt(ri@ri/(m.sum()-X.shape[1]))
    XtXi=np.linalg.pinv(X[m].T@X[m])
    lev=float(X[i]@XtXi@X[i])
    si_pred=si*np.sqrt(1.0+lev)               # predictive sd, new-point leverage
    tq=t_dist.ppf(0.90, m.sum()-X.shape[1])   # t quantile, small-sample honest
    su_pred=su_*np.sqrt(1.0+1.0/m.sum())
    tqu=t_dist.ppf(0.90, m.sum()-1)
    mc=X[i]@bi
    ub=(np.exp(bu_-tqu*su_pred),np.exp(bu_+tqu*su_pred))
    cb=(np.exp(mc-tq*si_pred),np.exp(mc+tq*si_pred))
    r=np.exp(y[i])
    ui=ub[0]<=r<=ub[1]; ci=cb[0]<=r<=cb[1]
    u_cov+=ui; c_cov+=ci
    shrinks.append((cb[1]-cb[0])/(ub[1]-ub[0]))
    u_ls.append(-0.5*((y[i]-bu_)/su_)**2-np.log(su_))
    c_ls.append(-0.5*((y[i]-mc)/si)**2-np.log(si))
    print(f"{names[i]:18s} {r:5.2f}  {ub[0]:5.2f}-{ub[1]:5.2f}   {'Y' if ui else 'n'}   "
          f"{cb[0]:5.2f}-{cb[1]:5.2f}   {'Y' if ci else 'n'}     x{shrinks[-1]:.2f}")
print(f"\ncoverage: unconditional {u_cov}/21, conditional {c_cov}/21")
print(f"mean 80% band width shrink under conditioning: x{np.mean(shrinks):.2f}")
print(f"mean log-score gain from conditioning: {np.mean(c_ls)-np.mean(u_ls):+.2f} nats/case")

# ---- worked example: what the model tells a hardware-sensitive buyer ----
print("\nWORKED PREDICTIONS (full-sample fit, 80% bands):")
for tag,x in [("mature-vendor, concentrate-only, easy site, off-boom",[1,0,0,0,0]),
              ("one novel unit op, chemical plant, remote, off-boom",[1,1,1,1,0]),
              ("bespoke hardware (HW=2), chemical plant, remote, boom",[1,2,1,1,1])]:
    mhat=np.array(x)@b; sd=np.sqrt(s2)
    print(f"  {tag:52s} P50 x{np.exp(mhat):.2f}  band x{np.exp(mhat-Z*sd):.2f}-x{np.exp(mhat+Z*sd):.2f}")
print("\nCodings are authors' judgment (med confidence); curate before client use.")
