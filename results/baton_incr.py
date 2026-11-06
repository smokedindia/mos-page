import json, os, numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
RES="/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/results/baton_aqa"
HA="/home/jhkim/.claude/jobs/61b803a8/tmp/Baton/data/HA"
AX={"Integrity (2label)":("2label_Integrity_HD_Audio","2label_HA.json"),
    "Temporal (3label)":("3label_Temporal_HD_Audio","3label_HA.json")}
def au(s,y):
    p,n=s[y==1],s[y==0]
    return stats.mannwhitneyu(p,n).statistic/(len(p)*len(n))
for ax,(lb,haf) in AX.items():
    aqa={r["output_file"]:r["aqa"] for r in json.load(open(f"{RES}/{lb}_aqa.json"))["results"]}
    clap={r["output_file"]:r["clap"] for r in json.load(open(f"{RES}/{lb}_clap.json"))["results"]}
    ha=[json.loads(l) for l in open(f"{HA}/{haf}") if l.strip()]
    lab={os.path.basename(x["location"]):int(x["feedback"]) for x in ha}
    f=[b for b in aqa if b in clap and b in lab]
    y=np.array([lab[b] for b in f]); A=np.array([aqa[b] for b in f]); C=np.array([clap[b] for b in f])
    X=np.c_[(A-A.mean())/A.std(),(C-C.mean())/C.std()]
    oof=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        oof[te]=LogisticRegression().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    lr=LogisticRegression().fit(X,y)
    print(f"\n{ax}: AUROC  AQA={au(A,y):.4f}  CLAP={au(C,y):.4f}  AQA+CLAP(5-fold OOF)={au(oof,y):.4f}")
    print(f"  logistic coefs (z-scored): AQA={lr.coef_[0][0]:+.3f}  CLAP={lr.coef_[0][1]:+.3f}")
    # partial: does AQA add signal beyond CLAP? residualize AQA on CLAP, correlate with label
    res=A-np.polyval(np.polyfit(C,A,1),C)
    pr=stats.pointbiserialr(y,res)
    print(f"  AQA residualized on CLAP vs human label: r={pr.statistic:+.4f} (p={pr.pvalue:.2e})")
    res2=C-np.polyval(np.polyfit(A,C,1),A)
    pr2=stats.pointbiserialr(y,res2)
    print(f"  CLAP residualized on AQA vs human label: r={pr2.statistic:+.4f} (p={pr2.pvalue:.2e})")
