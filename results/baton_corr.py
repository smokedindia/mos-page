#!/usr/bin/env python3
"""AQAScore vs Baton human labels — Integrity (and Temporal if available).

Human feedback: 1 = audio matches caption (good), 0 = not.
AQAScore = P("Yes" | audio, caption) in [0,1]; higher should mean "matches".

VQAScore's binary-match protocol: AUROC of the continuous metric vs the binary
human label (primary), plus point-biserial r, Mann-Whitney, and best-threshold
accuracy, all with bootstrap CIs. Also a caption-grouped (within-prompt) pairwise
accuracy: for each caption, over all good/bad clip pairs, does AQA rank good>bad?
"""
import json, os, numpy as np
from collections import defaultdict
from scipy import stats

RESDIR = "/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/results/baton_aqa"
HA_DIR = "/home/jhkim/.claude/jobs/61b803a8/tmp/Baton/data/HA"
rng = np.random.default_rng(0)

AXES = {
    "Integrity (2label)": ("2label_Integrity_HD_Audio_aqa.json", "2label_HA.json"),
    "Temporal (3label)":  ("3label_Temporal_HD_Audio_aqa.json",  "3label_HA.json"),
}


def loadjl(f):
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


def boot(fn, s, y, n=10000):
    out = []
    m = len(y)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = fn(s[idx], y[idx])
        if np.isfinite(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5])


def auroc(s, yy):
    pos, neg = s[yy == 1], s[yy == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos)*len(neg))


for axis, (aqf, haf) in AXES.items():
    aqp = os.path.join(RESDIR, aqf)
    if not os.path.exists(aqp):
        print(f"\n### {axis}: AQA output not found ({aqf}) — skipping")
        continue
    res = json.load(open(aqp))["results"]
    aqa = {r["output_file"]: r["aqa"] for r in res}
    ha = loadjl(os.path.join(HA_DIR, haf))
    lab = {os.path.basename(x["location"]): int(x["feedback"]) for x in ha}
    cap = {os.path.basename(x["location"]): x["captions"] for x in ha}

    score, y, caps = [], [], []
    for b, a in aqa.items():
        if b in lab:
            score.append(a); y.append(lab[b]); caps.append(cap[b])
    score, y = np.array(score), np.array(y)
    n = len(y)
    print(f"\n########## {axis}  (n={n}, pos={int(y.sum())}, neg={int((y==0).sum())}) ##########")
    print(f"  AQA overall: mean={score.mean():.4f} sd={score.std():.4f} "
          f"[{score.min():.3f},{score.max():.3f}]")
    print(f"  AQA|good={score[y==1].mean():.4f}  AQA|bad={score[y==0].mean():.4f}  "
          f"gap={score[y==1].mean()-score[y==0].mean():+.4f}")

    A, aci = auroc(score, y), boot(auroc, score, y)
    pb = stats.pointbiserialr(y, score)
    mw = stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="greater")
    ths = np.unique(score)
    accs = [((score >= t).astype(int) == y).mean() for t in ths]
    base = max(y.mean(), 1 - y.mean())
    print(f"  AUROC            = {A:.4f}  95%CI[{aci[0]:.4f},{aci[1]:.4f}]  (0.5=chance)")
    print(f"  point-biserial r = {pb.statistic:+.4f} (p={pb.pvalue:.2e})")
    print(f"  Mann-Whitney one-sided (good>bad) p = {mw.pvalue:.2e}")
    print(f"  best-threshold acc = {max(accs):.4f}  (base rate {base:.4f})")

    # within-caption pairwise accuracy (rank good>bad among clips of same prompt)
    byc = defaultdict(list)
    for a, yy, c in zip(score, y, caps):
        byc[c].append((a, yy))
    ncorr = nties = ntot = 0
    for c, lst in byc.items():
        for i in range(len(lst)):
            for j in range(len(lst)):
                ai, yi = lst[i]; aj, yj = lst[j]
                if yi == 1 and yj == 0:  # i good, j bad
                    ntot += 1
                    if ai > aj:
                        ncorr += 1
                    elif ai == aj:
                        nties += 1
    if ntot:
        pa = (ncorr + 0.5 * nties) / ntot
        print(f"  within-caption pairwise acc (good>bad) = {pa:.4f}  "
              f"over {ntot} good/bad pairs in {sum(1 for c in byc if any(t[1]==1 for t in byc[c]) and any(t[1]==0 for t in byc[c]))} captions")
