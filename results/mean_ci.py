import json, os, numpy as np
from collections import defaultdict
from scipy import stats
RESDIR="/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/results/baton_aqa"
HA_DIR="/home/jhkim/.claude/jobs/61b803a8/tmp/Baton/data/HA"
AXES={"Integrity":("2label_Integrity_HD_Audio","2label_HA.json"),
      "Temporal":("3label_Temporal_HD_Audio","3label_HA.json")}
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
    rng=np.random.default_rng(0); mp=[]; mr=[]; gp=[]
    for _ in range(10000):
        pick=rng.integers(0,len(idxs),len(idxs)); idx=np.concatenate([idxs[p] for p in pick])
        yy=y[idx]; ss=s[idx]
        if 0<yy.sum()<len(yy):
            a,b=ss[yy==1].mean(), ss[yy==0].mean()
            mp.append(a); mr.append(b); gp.append(a-b)
    q=lambda v: np.percentile(v,[2.5,97.5])
    sp, sr_ = s[y==1], s[y==0]
    print(f"\n=== {ax} (n={len(y)}, {len(idxs)} prompts) ===")
    print(f"  preferred: mean={sp.mean():.4f} sd={sp.std(ddof=1):.4f} n={len(sp)}  clustered 95%CI[{q(mp)[0]:.4f},{q(mp)[1]:.4f}]  (naive 1.96SE=+/-{1.96*sp.std(ddof=1)/np.sqrt(len(sp)):.4f})")
    print(f"  rejected : mean={sr_.mean():.4f} sd={sr_.std(ddof=1):.4f} n={len(sr_)}  clustered 95%CI[{q(mr)[0]:.4f},{q(mr)[1]:.4f}]  (naive 1.96SE=+/-{1.96*sr_.std(ddof=1)/np.sqrt(len(sr_)):.4f})")
    lo,hi=q(gp)
    print(f"  GAP      : {sp.mean()-sr_.mean():+.4f}  clustered 95%CI[{lo:+.4f},{hi:+.4f}]  -> {'excludes 0' if lo>0 else 'includes 0'}")
