#!/usr/bin/env python3
"""Audit: is our BATON evaluation identical to VQAScore's stated protocol?

Checks
  1. AUROC  : ours (Mann-Whitney U / n_pos*n_neg) vs sklearn.roc_auc_score
  2. r      : point-biserial vs Pearson (should be identical by definition)
  3. pairwise: ours (within-caption, human-ties dropped, 0.5 for exact metric tie)
              vs global concordance (human-ties dropped)
              vs Deutsch et al. 2023 tie-CALIBRATED pairwise accuracy (all pairs)
  4. bootstrap: i.i.d.-clip resampling vs caption-clustered resampling
"""
import json, os, numpy as np
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score

RESDIR = "/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/results/baton_aqa"
HA_DIR = "/home/jhkim/.claude/jobs/61b803a8/tmp/Baton/data/HA"
AXES = {
    "Integrity (2label)": ("2label_Integrity_HD_Audio", "2label_HA.json"),
    "Temporal (3label)":  ("3label_Temporal_HD_Audio",  "3label_HA.json"),
}


def loadjl(f):
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


def auroc_ours(s, yy):
    pos, neg = s[yy == 1], s[yy == 0]
    return stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg))


def pairwise_ours(score, y, caps):
    """What we reported: within-caption, only good/bad pairs, 0.5 for exact tie."""
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
    return (ncorr + 0.5 * nties) / ntot, ntot


def pairwise_global(score, y):
    """Global (not caption-grouped) concordance over discordant-human pairs only."""
    pos, neg = score[y == 1], score[y == 0]
    d = pos[:, None] - neg[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size, d.size


def pairwise_deutsch(score, y):
    """Deutsch et al. 2023 pairwise accuracy WITH tie calibration.

    Over all M(M-1)/2 pairs: human relation in {>,<,=} vs metric relation where
    |dm| <= eps counts as a metric tie. eps is chosen to maximise accuracy
    (tie calibration). Human ties are INCLUDED and require a metric tie for credit.
    """
    M = len(y)
    i, j = np.triu_indices(M, k=1)
    dh = np.sign(y[i] - y[j])           # -1 / 0 / +1
    dm = score[i] - score[j]
    a = np.abs(dm)
    # candidate eps: 0 plus every observed |dm| (accuracy is piecewise constant)
    cands = np.unique(np.concatenate([[0.0], a]))
    if len(cands) > 4000:                # subsample grid for tractability
        cands = np.quantile(a, np.linspace(0, 1, 4000))
        cands = np.unique(np.concatenate([[0.0], cands]))
    best = (-1.0, None)
    n = len(dh)
    for eps in cands:
        sm = np.where(a <= eps, 0, np.sign(dm))
        acc = (sm == dh).sum() / n
        if acc > best[0]:
            best = (acc, eps)
    sm0 = np.where(a <= 0, 0, np.sign(dm))
    return best[0], best[1], (sm0 == dh).sum() / n, n, (dh == 0).mean()


def boot_iid(s, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    out, m = [], len(y)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if 0 < y[idx].sum() < m:
            out.append(auroc_ours(s[idx], y[idx]))
    return np.percentile(out, [2.5, 97.5])


def boot_cluster(s, y, caps, n=10000, seed=0):
    """Resample CAPTIONS (clusters), not clips — clips are nested in captions."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for k, c in enumerate(caps):
        groups[c].append(k)
    keys = list(groups)
    idxs = [np.array(groups[k]) for k in keys]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idxs[p] for p in pick])
        if 0 < y[idx].sum() < len(idx):
            out.append(auroc_ours(s[idx], y[idx]))
    return np.percentile(out, [2.5, 97.5])


for axis, (label, haf) in AXES.items():
    aqa = {r["output_file"]: r["aqa"] for r in json.load(open(f"{RESDIR}/{label}_aqa.json"))["results"]}
    clap = {r["output_file"]: r["clap"] for r in json.load(open(f"{RESDIR}/{label}_clap.json"))["results"]}
    ha = loadjl(os.path.join(HA_DIR, haf))
    lab = {os.path.basename(x["location"]): int(x["feedback"]) for x in ha}
    cap = {os.path.basename(x["location"]): x["captions"] for x in ha}
    files = [b for b in aqa if b in clap and b in lab]
    y = np.array([lab[b] for b in files])
    caps = [cap[b] for b in files]

    ncap = len(set(caps))
    print(f"\n{'='*78}\n{axis}: n={len(y)} clips, {ncap} distinct captions "
          f"({len(y)/ncap:.2f} clips/caption), pos={int(y.sum())}")

    for name, d in (("AQAScore", aqa), ("CLAP score", clap)):
        s = np.array([d[b] for b in files])
        a_ours = auroc_ours(s, y)
        a_sk = roc_auc_score(y, s)
        pb = stats.pointbiserialr(y, s)
        pr = stats.pearsonr(y.astype(float), s)
        p_ours, n_ours = pairwise_ours(s, y, caps)
        p_glob, n_glob = pairwise_global(s, y)
        p_deu, eps, p_noeps, n_all, frac_htie = pairwise_deutsch(s, y)
        ci_i = boot_iid(s, y)
        ci_c = boot_cluster(s, y, caps)

        print(f"\n  --- {name} ---")
        print(f"  [1] AUROC ours = {a_ours:.6f}   sklearn = {a_sk:.6f}   "
              f"|diff| = {abs(a_ours-a_sk):.2e}  -> {'IDENTICAL' if abs(a_ours-a_sk)<1e-9 else 'MISMATCH'}")
        print(f"  [2] point-biserial r = {pb.statistic:+.6f}   Pearson r = {pr.statistic:+.6f}   "
              f"-> {'IDENTICAL' if abs(pb.statistic-pr.statistic)<1e-9 else 'MISMATCH'}")
        print(f"  [3] pairwise, ours (within-caption, {n_ours} pairs)      = {p_ours:.4f}")
        print(f"      pairwise, global concordance ({n_glob} pairs)     = {p_glob:.4f}")
        print(f"      Deutsch tie-CALIBRATED, all {n_all} pairs         = {p_deu:.4f}  (eps*={eps:.4g})")
        print(f"      same but eps=0 (no calibration)                  = {p_noeps:.4f}")
        print(f"      fraction of pairs that are HUMAN TIES            = {frac_htie:.3f}")
        print(f"  [4] AUROC 95%CI  i.i.d.-clip bootstrap    = [{ci_i[0]:.4f}, {ci_i[1]:.4f}]  "
              f"(width {ci_i[1]-ci_i[0]:.4f})")
        print(f"      AUROC 95%CI  caption-clustered boot   = [{ci_c[0]:.4f}, {ci_c[1]:.4f}]  "
              f"(width {ci_c[1]-ci_c[0]:.4f})")
