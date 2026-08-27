"""
Latent-regime HMM (Stage B, step 2).

Fits a 3-state Gaussian-emission HMM per country over the indicator panel, then
produces, for each country-year, a BELIEF distribution over {orthodox, interventionist,
crisis} using only indicators observable up to that year (filtered belief) or the
whole series (smoothed belief). These soft beliefs replace the hard regime label in
the activation model.

Key design choices that address the held-out failure:
  1. FILTERED belief (forward pass only) uses information up to year t, so a frozen
     evaluation at end-2021 sees Chile's DRIFT, not a retrospective label.
  2. One-step-ahead PREDICTED belief (filtered belief pushed through the transition
     matrix) is what a 5-year activation window should integrate over: it anticipates
     the transition instead of waiting for it.
  3. Emissions and transitions are initialized from the hand-coded regimes (weak
     supervision) but then re-fit by EM to the indicators, so the labels seed the
     model without dictating the beliefs.

Honesty: 3 states, 6 indicators, 360 country-years. This is a small HMM with weak
supervision; beliefs are estimates. The point is not a definitive regime call but a
CALIBRATED SOFTENING that carries transition uncertainty forward.
"""
import numpy as np, sys
_os = __import__('os')
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import indicator_panel as ip
import activation_panel as ap
rng = np.random.default_rng(0)
K = 3  # regimes

def _gauss_ll(x, mu, var):
    return -0.5*(np.log(2*np.pi*var).sum() + (((x-mu)**2)/var).sum())

def fit_country(seq, mu, var, A, pi, n_iter=25):
    """Baum-Welch for one country's indicator sequence. Returns filtered & smoothed
    beliefs and updated sufficient stats (accumulated by caller for pooling)."""
    T=len(seq)
    logB=np.array([[_gauss_ll(seq[t],mu[k],var[k]) for k in range(K)] for t in range(T)])
    B=np.exp(logB-logB.max(1,keepdims=True))+1e-300
    # forward (filtered)
    al=np.zeros((T,K)); c=np.zeros(T)
    al[0]=pi*B[0]; c[0]=al[0].sum(); al[0]/=c[0]
    for t in range(1,T):
        al[t]=(al[t-1]@A)*B[t]; c[t]=al[t].sum(); al[t]/=c[t]
    # backward
    be=np.zeros((T,K)); be[-1]=1.0
    for t in range(T-2,-1,-1):
        be[t]=(A@(B[t+1]*be[t+1])); be[t]/=be[t].sum()+1e-300
    ga=al*be; ga/=ga.sum(1,keepdims=True)+1e-300           # smoothed
    # predicted (one-step-ahead from filtered): what a forward-looking window sees
    pred=np.vstack([pi]+[al[t]@A for t in range(T-1)])
    return al, ga, pred, B

def train(n_outer=12):
    X=ip.panel(); Mz,keys,M=ip.matrix()
    # init from hand labels (weak supervision)
    y0=np.array([ap.regime_of(c,yr) for (c,yr) in keys])
    mu=np.array([Mz[y0==k].mean(0) for k in range(K)])
    var=np.array([Mz[y0==k].var(0)+0.8 for k in range(K)])
    A=np.full((K,K),0.06); np.fill_diagonal(A,0.88)         # sticky regimes
    A/=A.sum(1,keepdims=True)
    pi=np.array([0.5,0.35,0.15])
    countries=ip.COUNTRIES
    seqs={c:np.array([Mz[keys.index((c,y))] for y in range(ip.YEAR0,ip.YEAR1+1)]) for c in countries}
    for it in range(n_outer):
        num_mu=np.zeros((K,6)); den=np.zeros(K); num_var=np.zeros((K,6))
        Atot=np.zeros((K,K))
        for c in countries:
            al,ga,pred,B=fit_country(seqs[c],mu,var,A,pi)
            for k in range(K):
                num_mu[k]+=(ga[:,k:k+1]*seqs[c]).sum(0); den[k]+=ga[:,k].sum()
            # transition counts via xi
            T=len(seqs[c])
            for t in range(T-1):
                xi=(al[t][:,None]*A*(B[t+1]*1.0)[None,:]); xi/=xi.sum()+1e-300
                Atot+=xi
        mu=num_mu/den[:,None]
        for c in countries:
            al,ga,pred,B=fit_country(seqs[c],mu,var,A,pi)
            for k in range(K):
                num_var[k]+=(ga[:,k:k+1]*(seqs[c]-mu[k])**2).sum(0)
        var=np.maximum(num_var/den[:,None],0.6)
        A=Atot/Atot.sum(1,keepdims=True)
    # produce belief tables
    filt={}; smooth={}; predb={}
    for c in countries:
        al,ga,pred,B=fit_country(seqs[c],mu,var,A,pi)
        for i,y in enumerate(range(ip.YEAR0,ip.YEAR1+1)):
            filt[(c,y)]=al[i]; smooth[(c,y)]=ga[i]; predb[(c,y)]=pred[i]
    return dict(mu=mu,var=var,A=A,filtered=filt,smoothed=smooth,predicted=predb)

# map fitted states to canonical {orthodox=0, interv=1, crisis=2} by stress ordering
def _label_states(m):
    stress = m["mu"][:, :3].sum(1)          # fiscal+fx+infl; low=orthodox, high=crisis
    order = np.argsort(stress)              # order[0]=lowest-stress fitted state
    # perm[new_canonical] = old_fitted_index
    perm = [int(order[0]), int(order[1]), int(order[2])]
    # belief vectors are indexed by fitted state; reorder each to canonical
    for tbl in ["filtered","smoothed","predicted"]:
        m[tbl] = {k: v[perm] for k, v in m[tbl].items()}
    m["mu"] = m["mu"][perm]
    m["var"] = m["var"][perm]
    m["A"] = m["A"][np.ix_(perm, perm)]
    return m

if __name__=="__main__":
    m=train(); m=_label_states(m)
    print("fitted transition matrix (rows from-state, ortho/interv/crisis):")
    print(np.round(m["A"],2))
    print("\nregime emission means (z-scored indicators):")
    for k,nm in enumerate(["orthodox","interv","crisis"]):
        print(f"  {nm:9s}", dict(zip(ip.IND,np.round(m['mu'][k],2))))
    # the two cases the held-out test turned on
    print("\nFILTERED belief (info up to that year only):")
    for c,ys in [("Chile",[2019,2020,2021,2022,2023]),("Argentina",[2016,2017,2018,2019])]:
        for y in ys:
            b=m["filtered"][(c,y)]
            print(f"  {c:9s} {y}: ortho {b[0]:.2f}  interv {b[1]:.2f}  crisis {b[2]:.2f}")
    print("\nPREDICTED (one-step-ahead) belief at the freeze years:")
    for c,y in [("Chile",2021),("Argentina",2018)]:
        b=m["predicted"][(c,y)]
        print(f"  {c:9s} {y}->next: ortho {b[0]:.2f}  interv {b[1]:.2f}  crisis {b[2]:.2f}")
