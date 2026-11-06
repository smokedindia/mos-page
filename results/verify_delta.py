import json, os, numpy as np
from collections import defaultdict
from scipy import stats
RESDIR="/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/results/baton_aqa"
HA_DIR="/home/jhkim/.claude/jobs/61b803a8/tmp/Baton/data/HA"
AXES={"Integrity":("2label_Integrity_HD_Audio","2label_HA.json"),
      "Temporal":("3label_Temporal_HD_Audio","3label_HA.json")}
def auroc(s,y):
    p,n=s[y==1],s[y==0]
    return stats.mannwhitneyu(p,n,alternative="two-sided").statistic/(len(p)*len(n))
for ax,(lab_,haf) in AXES.items():
    aqa={r["output_file"]:r["aqa"] for r in json.load(open(f"{RESDIR}/{lab_}_aqa.json"))["results"]}
    clap={r["output_file"]:r["clap"] for r in json.load(open(f"{RESDIR}/{lab_}_clap.json"))["results"]}
    ha=[json.loads(l) for l in open(os.path.join(HA_DIR,haf),encoding="utf-8") if l.strip()]
    lab={os.path.basename(x["location"]):int(x["feedback"]) for x in ha}
    cap={os.path.basename(x["location"]):x["captions"] for x in ha}
    files=[b for b in aqa if b in clap and b in lab]
    y=np.array([lab[b] for b in files]); caps=[cap[b] for b in files]
    sa=np.array([aqa[b] for b in files]); sc=np.array([clap[b] for b in files])
    d=auroc(sa,y)-auroc(sc,y)
    g=defaultdict(list)
    for k,c in enumerate(caps): g[c].append(k)
    idxs=[np.array(v) for v in g.values()]
    for tag,mode in (("iid","iid"),("clustered","cl")):
        rng=np.random.default_rng(1); out=[]
        for _ in range(10000):
            if mode=="iid": idx=rng.integers(0,len(y),len(y))
            else:
                pick=rng.integers(0,len(idxs),len(idxs)); idx=np.concatenate([idxs[p] for p in pick])
            yy=y[idx]
            if 0<yy.sum()<len(yy): out.append(auroc(sa[idx],yy)-auroc(sc[idx],yy))
        lo,hi=np.percentile(out,[2.5,97.5])
        print(f"{ax:10} dAUROC(AQA-CLAP)={d:+.4f}  {tag:9} 95%CI[{lo:+.4f},{hi:+.4f}]  -> {'SIGNIFICANT' if (lo>0 or hi<0) else 'not significant'}")
