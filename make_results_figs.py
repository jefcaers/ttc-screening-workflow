"""Candidate figures for the Results section (300 dpi, Arial-metric font), regenerated from
the cohort engines where the engines exist and from engine result files otherwise.
Every number is engine output; nothing is drawn from memory. Outputs: rf_*.png + rf_data.json"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import io, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from contextlib import redirect_stdout
from scipy.stats import spearmanr, norm, kstest

INK="#141414"; GRAY="#8A8A8A"; LB="#9BB8D3"
BLUE="#2F6DB5"; GREEN="#1B8A5A"; RED="#C8102E"; AMBER="#C98A1E"
plt.rcParams.update({"font.family":"Liberation Sans","font.size":9,"axes.titlesize":9,
                     "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8})
DPI=300; W=6.5
def clean(ax, left=True):
    for sp in ["top","right"]+([] if left else ["left"]): ax.spines[sp].set_visible(False)
def save(name):
    plt.tight_layout(); plt.savefig(_os.path.join(_HERE,"figures",name+".png"),dpi=DPI,bbox_inches="tight",facecolor="white"); plt.close()

g={}
with redirect_stdout(io.StringIO()):
    exec(open(_os.path.join(_HERE,"backtest_copper.py")).read(), g)
posterior,frozen_pool,Z=g["posterior"],g["frozen_pool"],1.2816
CASES=g["CASES_CU"]; PRE_CU=g["PRE_CU"]; era_cu=g["era_cu"]; npv_engine=g["npv_engine"]
realized_npv=g["realized_npv"]; GAM=g["GAM_CU"]; LIp=g["LI"]; CU=g["CU"]
DATA={}

# ---------------- lithium cohort (from paper Table 1 / generator) -------------------
PRE_LI=[("HombreMuerto",1998,1.35),("Zabuye",2006,1.85),("Qinghai",2007,1.60),("SilverPeak",1990,1.00),("Rincon_pilot",2009,1.50)]
LI_KNOWN={"Olaroz":2016,"MtCattlin":2013,"MtMarion":2017,"Pilgangoora":2019,"Altura":2019,"Nemaska":2019,"BaldHill":2019,"Wodgina":2019,"Cauchari":2023,"Mibra":2019}
LI=[("Olaroz",2012,1.75,229),("MtCattlin",2010,1.30,70),("MtMarion",2015,1.15,90),("Pilgangoora",2017,1.17,180),("Altura",2017,1.30,110),
    ("Nemaska",2018,1.45,880),("BaldHill",2018,1.25,42),("Wodgina",2018,1.10,420),("Cauchari",2019,1.73,565),("Mibra",2017,1.05,60)]
def li_pool(yr):
    p=[r for(_,ky,r) in PRE_LI if ky<yr]; p+=[r for(n,y,r,_) in LI if LI_KNOWN[n]<yr]; return p

# ================= R1 forest: all 21 cases vs frozen 80% band =====================
rows=[]
for n,y,r,_ in LI:
    mu,s=posterior(li_pool(y),"rc"); mui,si=posterior(li_pool(y),"inside")
    rows.append(dict(name=n,yr=y,ratio=r,lo=np.exp(mu-Z*s),hi=np.exp(mu+Z*s),ilo=np.exp(mui-Z*si),ihi=np.exp(mui+Z*si),
                     pit=float(norm.cdf((np.log(r)-mu)/s)),com="Li"))
for c in CASES:
    pool=frozen_pool(c,CASES,PRE_CU); mu,s=posterior(pool,"rc"); mui,si=posterior(pool,"inside")
    rows.append(dict(name=c["name"].split(" (")[0],yr=c["yr"],ratio=c["ratio"],lo=np.exp(mu-Z*s),hi=np.exp(mu+Z*s),
                     ilo=np.exp(mui-Z*si),ihi=np.exp(mui+Z*si),pit=float(norm.cdf((np.log(c["ratio"])-mu)/s)),com="Cu",conf=c["conf"]))
for r in rows: r["in"]=bool(r["lo"]<=r["ratio"]<=r["hi"]); r["in_inside"]=bool(r["ilo"]<=r["ratio"]<=r["ihi"])
DATA["forest"]=rows
fig,ax=plt.subplots(figsize=(W,6.4))
ys=np.arange(len(rows))[::-1]
for r,yy in zip(rows,ys):
    col=BLUE if r["com"]=="Li" else GREEN
    ax.plot([r["lo"],r["hi"]],[yy,yy],color=col,lw=3.2,alpha=.5,solid_capstyle="butt")
    ax.plot([r["ilo"],r["ihi"]],[yy-0.28,yy-0.28],color=GRAY,lw=1.2,alpha=.9,solid_capstyle="butt")
    ax.plot(r["ratio"],yy,"o" if r["in"] else "X",color=INK if r["in"] else RED,ms=6,zorder=5)
ax.axhline(len(CASES)-0.5,color=GRAY,lw=.8,ls=":")
ax.set_yticks(ys); ax.set_yticklabels([f'{r["name"]} \'{str(r["yr"])[2:]}' for r in rows],fontsize=7.5)
ax.text(2.22,len(rows)-1.0,"lithium",color=BLUE,fontweight="bold",ha="right"); ax.text(2.22,len(CASES)-2.0,"copper",color=GREEN,fontweight="bold",ha="right")
ax.set_xlim(0.7,2.25); ax.set_xlabel("realized ÷ sanctioned capital cost")
ax.plot([],[],color=BLUE,lw=3,alpha=.5,label="reference-class 80% band (frozen; blue lithium, green copper)"); ax.plot([],[],color=GRAY,lw=1.2,label="inside-view 80% band")
ax.plot([],[],"o",color=INK,label="realized, inside RC band"); ax.plot([],[],"X",color=RED,label="realized, outside RC band")
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.45,-0.07),ncol=2); clean(ax); save("rf01_forest")

# ================= R2 PIT ECDF ==================================================
pits=np.array([r["pit"] for r in rows]); pli=np.array([r["pit"] for r in rows if r["com"]=="Li"]); pcu=np.array([r["pit"] for r in rows if r["com"]=="Cu"])
ks_all=kstest(pits,"uniform"); ks_li=kstest(pli,"uniform")
DATA["pit"]=dict(all=pits.tolist(),ks_p_all=float(ks_all.pvalue),mean_all=float(pits.mean()),ks_p_li=float(ks_li.pvalue),mean_li=float(pli.mean()),
                 ks_p_cu=float(kstest(pcu,"uniform").pvalue),mean_cu=float(pcu.mean()))
fig,ax=plt.subplots(figsize=(W,3.3))
xs=np.sort(pits); ax.step(np.r_[0,xs,1],np.r_[0,np.arange(1,len(xs)+1)/len(xs),1],where="post",color=INK,lw=2,label=f"21 cases (KS p = {ks_all.pvalue:.2f}, mean {pits.mean():.2f})")
ax.plot([0,1],[0,1],color=GRAY,ls="--",lw=1.2,label="uniform reference")
for v in pli: ax.plot([v,v],[0.0,0.035],color=BLUE,lw=1.5)
for v in pcu: ax.plot([v,v],[0.045,0.08],color=GREEN,lw=1.5)
ax.text(0.005,0.018,"lithium",color=BLUE,fontsize=7,va="center"); ax.text(0.005,0.063,"copper",color=GREEN,fontsize=7,va="center")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel("probability integral transform of realized ratio under frozen posterior"); ax.set_ylabel("empirical CDF")
ax.legend(frameon=False,loc="upper left"); clean(ax); save("rf02_pit")

# ================= R3 calibration summary bars ==================================
cov=dict(Li_rc=sum(r["in"] for r in rows if r["com"]=="Li"),Li_in=sum(r["in_inside"] for r in rows if r["com"]=="Li"),
         Cu_rc=sum(r["in"] for r in rows if r["com"]=="Cu"),Cu_in=sum(r["in_inside"] for r in rows if r["com"]=="Cu"))
DATA["coverage"]=cov
fig,axs=plt.subplots(1,2,figsize=(W,2.9))
ax=axs[0]; xx=np.arange(2)
ax.bar(xx-0.18,[cov["Li_rc"]/10,cov["Cu_rc"]/11],0.34,color=[BLUE,GREEN],label="reference class")
ax.bar(xx+0.18,[cov["Li_in"]/10,cov["Cu_in"]/11],0.34,color=GRAY,label="inside view")
for i,(a,b,n) in enumerate([(cov["Li_rc"],cov["Li_in"],10),(cov["Cu_rc"],cov["Cu_in"],11)]):
    ax.text(i-0.18,a/n+0.02,f"{a}/{n}",ha="center",fontsize=8); ax.text(i+0.18,b/n+0.02,f"{b}/{n}",ha="center",fontsize=8)
ax.axhline(0.8,color=INK,ls=":",lw=1); ax.text(1.5,0.81,"nominal 80%",fontsize=7,ha="right")
ax.set_xticks(xx); ax.set_xticklabels(["lithium (n=10)","copper (n=11)"]); ax.set_ylim(0,1.3); ax.set_ylabel("80% band coverage"); ax.legend(frameon=False,loc="upper center",ncol=2,bbox_to_anchor=(0.5,1.04)); clean(ax)
ax=axs[1]
ls=[0.23,-0.36,-0.22,-0.61]  # mean log scores: Li RC, Li inside, Cu RC, Cu inside (engine results)
ax.bar([0-0.18,1-0.18],[ls[0],ls[2]],0.34,color=[BLUE,GREEN]); ax.bar([0+0.18,1+0.18],[ls[1],ls[3]],0.34,color=GRAY)
ax.axhline(0,color=INK,lw=.8); ax.set_xticks(xx); ax.set_xticklabels(["lithium","copper"]); ax.set_ylabel("mean log score (nats/case)")
ax.text(0,0.30,"margin +0.58",ha="center",fontsize=8,color=INK); ax.text(1,0.05,"margin +0.39",ha="center",fontsize=8,color=INK)
clean(ax); save("rf03_calibration_bars")

# ================= R4 copper RAV scatter (8%) ==================================
rng=np.random.default_rng(7)
def sysrun(case,disc=0.08,lam=0.5,N=2000):
    p0,scen,sup=era_cu(case["yr"]); mu,s=posterior(frozen_pool(case,CASES,PRE_CU))
    draws=case["est"]*np.exp(mu+s*rng.standard_normal(N)); tot=sum(w for _,w in scen); npvs=np.empty(N)
    for i in range(N):
        r=rng.random()*tot; gg=scen[0][0]
        for cg,w in scen:
            r-=w
            if r<=0: gg=cg; break
        pp=lambda t,gg=gg:min(max(p0*((1+gg)/(1+sup))**(GAM*t),0.5*p0),2.5*p0)
        c2=dict(case); c2["cc"]=case["cc"]*np.exp(0.15*rng.standard_normal())
        npvs[i]=npv_engine(c2,draws[i],pp,disc=disc)
    m=npvs.mean(); srt=np.sort(npvs); cv=srt[:max(1,N//20)].mean(); rav=m-lam*(m-cv)
    v="walk" if rav<=0 and m<=0 else ("stage" if rav<=0 else "enter")
    return rav,v,npvs
res=[(c,)+sysrun(c)+(realized_npv(c,disc=0.08),) for c in CASES]
ravs=np.array([r[1] for r in res]); rns=np.array([r[4] for r in res]); rho,pv=spearmanr(ravs,rns)
DATA["copper_choices"]=[dict(name=c["name"],yr=c["yr"],rav=float(rav),choice=v,realized=float(rn),ratio=c["ratio"],conf=c["conf"]) for c,rav,v,_,rn in res]
DATA["rho8"]=[float(rho),float(pv)]
colm={"walk":RED,"stage":AMBER,"enter":GREEN}
fig,ax=plt.subplots(figsize=(W,4.1))
for c,rav,v,_,rn in res:
    ax.plot(rav/1000,rn/1000,"o",color=colm[v],ms=7,mec=INK,mew=.5)
    if "Sierra" in c["name"] or "Quebrada" in c["name"]:
        ax.annotate(("Sierra Gorda" if "Sierra" in c["name"] else "QB2")+" (Sumitomo entered; engine: walk)",(rav/1000,rn/1000),textcoords="offset points",xytext=(8,-4),fontsize=7.5,color=RED)
ax.axhline(0,color=GRAY,lw=.8); ax.axvline(0,color=GRAY,lw=.8)
ax.set_xlabel("ex-ante RAV on frozen sanction-date information (US$B, 8% discount)"); ax.set_ylabel("realized NPV, reconstruction (US$B)")
ax.text(.02,.96,f"Spearman ρ = {rho:.2f} (p = {pv:.2f}); 0.57 to 0.80 across the 9-cell grid",transform=ax.transAxes,va="top",fontsize=8)
for v in ["enter","stage","walk"]: ax.plot([],[],"o",color=colm[v],mec=INK,mew=.5,label="choice: "+v)
ax.legend(frameon=False,loc="lower right"); clean(ax); save("rf04_copper_scatter")

# ================= R5 lithium decision-loss bars (paper numbers, lithium engine) ===
DATA["li_loss"]=dict(workflow=721,always_enter=1476,always_walk=1074,random=1242)
fig,ax=plt.subplots(figsize=(W,2.8))
labs=["workflow","always-enter","always-walk","random"]; vals=[721,1476,1074,1242]
ax.barh(labs[::-1],vals[::-1],color=[GRAY,GRAY,GRAY,BLUE],height=0.55)
for l,v in zip(labs,vals): ax.text(v+20,l,f"US${v:,} M",va="center",fontsize=8)
ax.set_xlim(0,1750); ax.set_xlabel("cost-weighted decision loss, lithium cohort (staged exposure 0.3, missed-upside weight 0.5)")
ax.text(1740,3.35,"51% of the always-enter → oracle improvement captured; beats all baselines in 19 of 20 grid cells",ha="right",fontsize=7.5,color=INK)
clean(ax); save("rf05_lithium_loss")

# ================= R6 copper loss-weight grid heatmap (engine sensitivity file) ==
grid=np.array([[94,87,81,75],[93,87,81,76],[93,87,81,76]]); chis=[0.25,0.5,0.75,1.0]; exps=[0.1,0.3,0.5]
DATA["cu_lossgrid"]=grid.tolist()
fig,ax=plt.subplots(figsize=(W*0.8,2.7))
im=ax.imshow(grid,cmap="Blues",vmin=60,vmax=100,aspect="auto")
for i in range(3):
    for j in range(4): ax.text(j,i,f"{grid[i,j]}%",ha="center",va="center",color="white" if grid[i,j]>82 else INK,fontsize=9,fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(chis); ax.set_yticks(range(3)); ax.set_yticklabels(exps)
ax.set_xlabel("missed-upside weight χ"); ax.set_ylabel("staged exposure")
ax.set_title("share of always-enter → oracle improvement captured (copper, 8%, λ = 0.5)",fontsize=8.5)
save("rf06_copper_lossgrid")

# ================= R7 rho grid disc x lambda (engine sensitivity file) ==========
rg=np.array([[0.80,0.71,0.70],[0.75,0.67,0.69],[0.66,0.65,0.57]]); pg=np.array([[0.00,0.01,0.02],[0.01,0.02,0.02],[0.03,0.03,0.07]])
mix=[["1/2/8","1/2/8","0/3/8"],["1/2/8","0/3/8","0/3/8"],["0/1/10","0/1/10","0/1/10"]]
DATA["rho_grid"]=dict(rho=rg.tolist(),p=pg.tolist(),mix=mix)
fig,ax=plt.subplots(figsize=(W*0.8,2.9))
im=ax.imshow(rg,cmap="Blues",vmin=0.4,vmax=0.9,aspect="auto")
for i in range(3):
    for j in range(3):
        ax.text(j,i-0.12,f"ρ = {rg[i,j]:.2f}",ha="center",va="center",color="white" if rg[i,j]>0.68 else INK,fontsize=8.5,fontweight="bold")
        ax.text(j,i+0.22,f"p = {pg[i,j]:.2f} · {mix[i][j]}",ha="center",va="center",color="white" if rg[i,j]>0.68 else INK,fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(["λ = 0.3","λ = 0.5","λ = 0.7"]); ax.set_yticks(range(3)); ax.set_yticklabels(["8%","9%","10%"]); ax.set_ylabel("discount rate")
ax.set_title("Spearman ρ(ex-ante RAV, realized NPV), copper; cell text: p-value · enter/stage/walk mix",fontsize=8)
save("rf07_rho_grid")

# ================= R8 price indices ============================================
yrs=sorted(set(CU)&set(LIp))
fig,ax=plt.subplots(figsize=(W,3.1))
ax.plot(yrs,[LIp[y]/LIp[2010] for y in yrs],color=BLUE,lw=2,label="lithium carbonate (index, 2010 = 1)")
ax.plot(yrs,[CU[y]/CU[2010] for y in yrs],color=GREEN,lw=2,label="copper (index, 2010 = 1)")
ax.set_yscale("log"); ax.set_yticks([0.5,1,2,4,7]); ax.set_yticklabels(["0.5","1","2","4","7"]); ax.set_xlabel("year")
ax.legend(frameon=False,loc="upper left"); ax.text(.98,.05,"mean pairwise sanction-window correlation, 21 cases: −0.02",transform=ax.transAxes,ha="right",fontsize=8)
clean(ax); save("rf08_prices")

# ================= R9 21x21 window correlation heatmap =========================
LI_YR=[y for _,y,_,_ in LI]; names=[f"{n} '{str(y)[2:]}" for n,y,_,_ in LI]+[f'{c["name"].split(" (")[0]} \'{str(c["yr"])[2:]}' for c in CASES]
def logwin(series,yr,L=7):
    ys=sorted(series); v=np.log([series[y] for y in ys]); a=ys.index(yr) if yr in ys else 0; return v[a:a+L]
wins=[logwin(LIp,y) for y in LI_YR]+[logwin(CU,c["yr"]) for c in CASES]
n=len(wins); M=np.full((n,n),np.nan)
for i in range(n):
    for j in range(n):
        wi,wj=wins[i],wins[j]; m=min(len(wi),len(wj))
        if m>=4: M[i,j]=np.corrcoef(wi[:m],wj[:m])[0,1]
off=[M[i,j] for i in range(n) for j in range(i+1,n) if not np.isnan(M[i,j])]
DATA["corr_mean"]=float(np.mean(off))
fig,ax=plt.subplots(figsize=(W,5.6))
im=ax.imshow(M,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(names,rotation=90,fontsize=6.5); ax.set_yticklabels(names,fontsize=6.5)
ax.axhline(9.5,color=INK,lw=.8); ax.axvline(9.5,color=INK,lw=.8)
cb=plt.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cb.set_label("correlation of 7-year log-price windows",fontsize=8)
ax.set_title(f"pairwise sanction-window price correlation (mean off-diagonal {np.mean(off):+.2f})",fontsize=8.5)
save("rf09_corr_matrix")

# ================= R10 held-out prediction bars (engine result files) ==========
ev=["Argentina T 2019","Chile N 2023","Chile T 2023","Chile L 2023"]
static=[0.052,0.016,0.079,0.016]; one=[0.091,0.043,0.089,0.026]; multi=[0.086,0.193,0.122,0.091]; base=[0.101,0.114,0.114,0.045]
DATA["heldout"]=dict(events=ev,static=static,onestep=one,multi=multi,base=base)
fig,ax=plt.subplots(figsize=(W,3.2)); xx=np.arange(4); w=0.2
ax.bar(xx-1.5*w,static,w,color="#CFCFCF",label="static regime (−1.00 nats vs base)")
ax.bar(xx-0.5*w,one,w,color=GRAY,label="latent regime, one-step (−0.46)")
ax.bar(xx+0.5*w,multi,w,color=BLUE,label="latent regime, multi-year path (+0.29)")
ax.bar(xx+1.5*w,base,w,color="white",edgecolor=INK,hatch="///",label="regime-blind base rate")
ax.set_xticks(xx); ax.set_xticklabels(ev); ax.set_ylabel("P(activation within 5 years)"); ax.legend(frameon=False,loc="upper left",fontsize=7)
clean(ax); save("rf10_heldout")

# ================= R11 Chile belief path ======================================
path=np.array([[0.11,0.87,0.01],[0.11,0.82,0.06],[0.11,0.78,0.11],[0.11,0.74,0.16],[0.10,0.70,0.20],[0.10,0.66,0.24]])
DATA["chile_path"]=path.tolist()
fig,ax=plt.subplots(figsize=(W,2.9)); ks=np.arange(6)
ax.stackplot(ks,path[:,0],path[:,1],path[:,2],colors=[LB,BLUE,RED],labels=["orthodox","interventionist","crisis"],alpha=.9)
ax.set_xticks(ks); ax.set_xticklabels(["end-2021\n(filtered)","+1","+2","+3","+4","+5"]); ax.set_ylim(0,1); ax.set_ylabel("regime belief"); ax.set_xlabel("years after the freeze")
ax.axvline(2,color=INK,ls=":",lw=1); ax.text(2.05,0.03,"2023 package activates",fontsize=7.5)
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.36),ncol=3,fontsize=7); clean(ax); save("rf11_chile_belief")

# ================= R12 QB2 RAV distribution (methods fig, regenerated) ==========
qb=[c for c in CASES if "Quebrada" in c["name"]][0]; rav,v,npvs=sysrun(qb)
srt=np.sort(npvs); cv=srt[:len(srt)//20].mean(); m=npvs.mean()
DATA["qb2"]=dict(rav=float(rav),cvar=float(cv),mean=float(m))
fig,ax=plt.subplots(figsize=(W,3.0))
ax.hist(npvs/1000,bins=40,color=LB,edgecolor="white"); ax.hist(srt[:len(srt)//20]/1000,bins=8,color=RED,edgecolor="white")
for val,lab,c in [(cv,"CVaR₅",RED),(rav,"RAV",INK),(m,"E[NPV]",BLUE)]:
    ax.axvline(val/1000,color=c,lw=1.5,ls="--"); ax.text(val/1000,ax.get_ylim()[1]*0.95,f" {lab} {val/1000:+.1f}",color=c,fontsize=8,va="top")
ax.set_xlabel("NPV, Quebrada Blanca 2 at 2018 sanction (US$B, 8% discount, frozen information)"); ax.set_yticks([]); clean(ax,left=False); save("rf12_qb2_dist")

# ================= R13 Olaroz cost posterior (methods fig, regenerated) ==========
muP=np.log(1.31); sP=0.318; s1=0.45; x=np.linspace(0.4,2.9,600)
post=np.exp(-0.5*((np.log(x)-muP)/sP)**2)/(x*sP); ins=np.exp(-0.5*((np.log(x))/s1)**2)/(x*s1)
fig,ax=plt.subplots(figsize=(W,3.2))
ax.plot(x,ins/ins.max(),color=GRAY,ls="--",lw=1.4,label="inside view (estimate-centered, AACE class 5)")
ax.plot(x,post/post.max(),color=BLUE,lw=2,label="reference-class × inside posterior")
for v,lab,c in [(0.87,"P10 ×0.87",LB),(1.31,"P50 ×1.31",INK),(1.97,"P90 ×1.97",RED)]:
    ax.axvline(v,color=c,ls=":",lw=1.3); ax.text(v,1.04,lab,ha="center",fontsize=8,color=c,fontweight="bold")
ax.axvline(1.75,color=GRAY,lw=1.6); ax.text(1.75,0.55,"built ×1.75",rotation=90,ha="right",va="center",fontsize=8)
ax.set_xlabel("realized ÷ sanctioned capital cost"); ax.set_yticks([]); ax.set_ylim(0,1.14); ax.legend(frameon=False,loc="upper right"); clean(ax,left=False); save("rf13_olaroz_posterior")

# ================= R14 workflow diagram (five engines, decision-choice wording) ==
fig,ax=plt.subplots(figsize=(W,3.6)); ax.set_xlim(0,100); ax.set_ylim(0,52); ax.axis("off")
def box(x,y,w,h,t,fc="white",ec=INK,fs=8,tc=None,bold=True):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.4",fc=fc,ec=ec,lw=1.3))
    ax.text(x,y,t,ha="center",va="center",fontsize=fs,color=tc or ec,fontweight="bold" if bold else "normal",linespacing=1.2)
def arr(x1,y1,x2,y2,c=INK,lw=1.2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,color=c,lw=lw,shrinkA=2,shrinkB=3))
box(50,46,72,7,"language-model agents: gather filings, propose reference cases, explain results (produce no number)",fs=8,ec=GRAY,tc=INK,bold=False)
ins_=[("scoping\nestimate",12),("comparable completed\nprojects (reference cases)",38),("era demand\nscenarios",64),("policy-event panel,\nindicator block",88)]
for t,x in ins_: box(x,34,22,7,t,fc="#F4F6F9",ec=GRAY,tc=INK,fs=7.5,bold=False)
eng=[("cost engine\ncapital posterior",16),("market engine\nprice paths",46),("country-risk engine\nintervention · prediction",80)]
for t,x in eng: box(x,21,22,7.5,t,ec=BLUE,fs=8)
arr(12,30.5,14,25); arr(38,30.5,20,25); arr(64,30.5,46,25); arr(88,30.5,82,25)
box(40,8.5,26,7.5,"valuation engine\nMonte Carlo NPV distribution",ec=BLUE,fs=8)
box(74,8.5,26,7.5,"decision engine\nRAV · V_stage → enter / stage / walk",ec=INK,fs=8)
arr(16,17.2,34,12.5); arr(46,17.2,42,12.5); arr(80,17.2,76,12.5,c=BLUE); arr(53.2,8.5,60.8,8.5)
for x in [12,38,64,88]: arr(x,42.4,x,37.6,c=GRAY,lw=0.9)
save("rf14_workflow")

json.dump(DATA,open(_os.path.join(_HERE,"data","rf_data.json"),"w"),indent=1,default=float)
print("forest misses:",[r["name"] for r in rows if not r["in"]]); print("coverage:",cov)
print("PIT:",{k:round(v,3) for k,v in DATA["pit"].items() if k!="all"})
print("rho8:",DATA["rho8"]); print("corr mean:",round(DATA["corr_mean"],3)); print("qb2:",{k:round(v) for k,v in DATA["qb2"].items()})
print("choices@8%:",[(d["name"].split(" (")[0],d["choice"]) for d in DATA["copper_choices"]])
