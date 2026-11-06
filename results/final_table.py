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
for ax,(lb,haf) in AXES.items():
    aqa={r["output_file"]:r["aqa"] for r in json.load(open(f"{RESDIR}/{lb}_aqa.json"))["results"]}
    ha=[json.loads(l) for l in open(os.path.join(HA_DIR,haf),encoding="utf-8") if l.strip()]
    lab={os.path.basename(x["location"]):int(x["feedback"]) for x in ha}
    cap={os.path.basename(x["location"]):x["captions"] for x in ha}
    files=[b for b in aqa if b in lab]
    y=np.array([lab[b] for b in files]); s=np.array([aqa[b] for b in files])
    caps=[cap[b] for b in files]
    g=defaultdict(list)
    for k,c in enumerate(caps): g[c].append(k)
    idxs=[np.array(v) for v in g.values()]
    rng=np.random.default_rng(0); out=[]
    for _ in range(10000):
        pick=rng.integers(0,len(idxs),len(idxs)); idx=np.concatenate([idxs[p] for p in pick])
        yy=y[idx]
        if 0<yy.sum()<len(yy): out.append(auroc(s[idx],yy))
    lo,hi=np.percentile(out,[2.5,97.5])
    pb=stats.pointbiserialr(y,s)
    mw=stats.mannwhitneyu(s[y==1],s[y==0],alternative="greater")
    print(f"{ax}: n={len(y)} (pref={int(y.sum())}, rej={int((y==0).sum())}, {len(idxs)} prompts)")
    print(f"  mean preferred={s[y==1].mean():.4f}  mean rejected={s[y==0].mean():.4f}  gap={s[y==1].mean()-s[y==0].mean():+.4f}")
    print(f"  AUROC={auroc(s,y):.4f}  clustered 95%CI[{lo:.4f},{hi:.4f}]")
    print(f"  point-biserial r={pb.statistic:+.4f} (p={pb.pvalue:.2e})   MW one-sided p={mw.pvalue:.2e}")
