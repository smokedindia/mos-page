#!/usr/bin/env python3
"""AQAScore vs CLAP score against BATON human preference labels.

Same protocol for both metrics (VQAScore's binary-match test): AUROC of the
continuous metric vs the binary human label (primary), point-biserial r,
one-sided Mann-Whitney, within-caption pairwise accuracy; bootstrap 95% CIs.
"""
import json, os, numpy as np
from collections import defaultdict
from scipy import stats

RESDIR = "/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/results/baton_aqa"
HA_DIR = "/home/jhkim/.claude/jobs/61b803a8/tmp/Baton/data/HA"

AXES = {
    "Integrity (2label)": ("2label_Integrity_HD_Audio", "2label_HA.json"),
    "Temporal (3label)":  ("3label_Temporal_HD_Audio",  "3label_HA.json"),
}


def loadjl(f):
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


def auroc(s, yy):
    pos, neg = s[yy == 1], s[yy == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg))


def boot(fn, s, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    out, m = [], len(y)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = fn(s[idx], y[idx])
        if np.isfinite(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5])


def pairwise(score, y, caps):
    byc = defaultdict(list)
    for a, yy, c in zip(score, y, caps):
        byc[c].append((a, yy))
    ncorr = nties = ntot = 0
    for lst in byc.values():
        for ai, yi in lst:
            for aj, yj in lst:
                if yi == 1 and yj == 0:
                    ntot += 1
                    ncorr += ai > aj
                    nties += ai == aj
    return ((ncorr + 0.5 * nties) / ntot if ntot else np.nan), ntot


def delta_auroc(sa, sc, y, n=10000, seed=1):
    """Bootstrap CI on AUROC(AQA) - AUROC(CLAP), paired on the same clips."""
    rng = np.random.default_rng(seed)
    out, m = [], len(y)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        va, vc = auroc(sa[idx], y[idx]), auroc(sc[idx], y[idx])
        if np.isfinite(va) and np.isfinite(vc):
            out.append(va - vc)
    return np.percentile(out, [2.5, 97.5])


rows = []
for axis, (label, haf) in AXES.items():
    aqp = os.path.join(RESDIR, f"{label}_aqa.json")
    clp = os.path.join(RESDIR, f"{label}_clap.json")
    if not (os.path.exists(aqp) and os.path.exists(clp)):
        print(f"### {axis}: missing output — skipping")
        continue
    aqa = {r["output_file"]: r["aqa"] for r in json.load(open(aqp))["results"]}
    clap = {r["output_file"]: r["clap"] for r in json.load(open(clp))["results"]}
    ha = loadjl(os.path.join(HA_DIR, haf))
    lab = {os.path.basename(x["location"]): int(x["feedback"]) for x in ha}
    cap = {os.path.basename(x["location"]): x["captions"] for x in ha}

    files = [b for b in aqa if b in clap and b in lab]
    y = np.array([lab[b] for b in files])
    caps = [cap[b] for b in files]
    S = {"AQAScore": np.array([aqa[b] for b in files]),
         "CLAP score": np.array([clap[b] for b in files])}

    print(f"\n########## {axis}  (n={len(y)}, good={int(y.sum())}, bad={int((y==0).sum())}) ##########")
    for name, s in S.items():
        A, ci = auroc(s, y), boot(auroc, s, y)
        pb = stats.pointbiserialr(y, s)
        mw = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="greater")
        pa, npair = pairwise(s, y, caps)
        print(f"  {name:11s} mean={s.mean():.4f} sd={s.std():.4f} "
              f"| good={s[y==1].mean():.4f} bad={s[y==0].mean():.4f} "
              f"gap={s[y==1].mean()-s[y==0].mean():+.4f}")
        print(f"  {'':11s} AUROC={A:.4f} [{ci[0]:.4f},{ci[1]:.4f}]  r={pb.statistic:+.4f} "
              f"(p={pb.pvalue:.2e})  MW p={mw.pvalue:.2e}  pairwise={pa:.4f} ({npair} pairs)")
        rows.append({"axis": axis, "metric": name, "n": int(len(y)), "auroc": float(A),
                     "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                     "r": float(pb.statistic), "p": float(pb.pvalue),
                     "pairwise": float(pa), "n_pairs": int(npair),
                     "mean_good": float(s[y == 1].mean()), "mean_bad": float(s[y == 0].mean())})

    d = auroc(S["AQAScore"], y) - auroc(S["CLAP score"], y)
    dci = delta_auroc(S["AQAScore"], S["CLAP score"], y)
    print(f"  ΔAUROC (AQA - CLAP) = {d:+.4f}  95%CI[{dci[0]:+.4f},{dci[1]:+.4f}]"
          f"  -> {'differ' if (dci[0] > 0 or dci[1] < 0) else 'no significant difference'}")
    print(f"  Spearman(AQA, CLAP) = {stats.spearmanr(S['AQAScore'], S['CLAP score']).statistic:+.4f}")
    rows.append({"axis": axis, "metric": "delta(AQA-CLAP)", "auroc": float(d),
                 "ci_lo": float(dci[0]), "ci_hi": float(dci[1])})

json.dump(rows, open(os.path.join(RESDIR, "baton_metric_comparison.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(RESDIR, 'baton_metric_comparison.json')}")
