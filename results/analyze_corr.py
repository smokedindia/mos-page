#!/usr/bin/env python3
"""AQAScore vs CMOS — clean recompute with scipy + system-level aggregate.

delta = AQA(candidate) - AQA(SCORE ref); CMOS positive => candidate preferred.
Positive correlation would mean AQAScore agrees with human preference.
"""
import json, sqlite3, numpy as np
from collections import defaultdict
from scipy import stats

RES = "/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/results"
AQA_DIR = f"{RES}/aqa_output"
DB = "/home/jhkim/projects/2026_SPL_ITS/icassp_round/mos-page/instance/2026_icassp_its.db"
rng = np.random.default_rng(0)


def load_aqa(label):
    with open(f"{AQA_DIR}/{label}_aqa.json") as f:
        return {r["output_file"]: r["aqa"] for r in json.load(f)["results"]}


aqa = {l: load_aqa(l) for l in ("BON", "rank", "SCORE")}

con = sqlite3.connect(DB)
users = {u: (n, t) for u, n, t in con.execute("select id,name,task_type from user")}
rows = list(con.execute("select user_id,model_name,file_name,score from score"))
AQ_EXCLUDE, TA_KEEP = {8}, {4}


def cmos(task, filtered):
    agg = defaultdict(list)
    for uid, model, fn, sc in rows:
        name, tt = users[uid]
        if tt != task:
            continue
        if filtered:
            if task == "Audio Quality" and uid in AQ_EXCLUDE:
                continue
            if task == "Text Alignment" and uid not in TA_KEEP:
                continue
        agg[(model, fn)].append(sc)
    return {k: float(np.mean(v)) for k, v in agg.items()}


def tie_cal_acc(delta, human):
    a = 0.0
    for d, h in zip(delta, human):
        sd, sh = np.sign(d), np.sign(h)
        a += 0.5 if (sd == 0 or sh == 0) else (1.0 if sd == sh else 0.0)
    return a / len(delta)


def boot(delta, human, stat, n=10000):
    d, h = np.array(delta), np.array(human)
    m = len(d)
    out = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            v = stat(d[idx], h[idx])
        except Exception:
            v = np.nan
        if np.isfinite(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def analyze(task, filtered, tag):
    means = cmos(task, filtered)
    delta, human = [], []
    for (model, fn), ms in sorted(means.items()):
        if fn in aqa[model] and fn in aqa["SCORE"]:
            delta.append(aqa[model][fn] - aqa["SCORE"][fn])
            human.append(ms)
    d, h = np.array(delta), np.array(human)
    n = len(d)
    pr = stats.pearsonr(d, h)
    sr = stats.spearmanr(d, h)
    acc = tie_cal_acc(d, h)
    prci = boot(d, h, lambda a, b: stats.pearsonr(a, b)[0])
    accci = boot(d, h, tie_cal_acc)
    conc = int(np.sum((np.sign(d) == np.sign(h)) & (d != 0) & (h != 0)))
    disc = int(np.sum((np.sign(d) != np.sign(h)) & (d != 0) & (h != 0)))
    print(f"\n--- {task} | {tag} (n={n}) ---")
    print(f"  Pearson  r   = {pr.statistic:+.3f} (p={pr.pvalue:.2f})  95%CI[{prci[0]:+.3f},{prci[1]:+.3f}]")
    print(f"  Spearman rho = {sr.statistic:+.3f} (p={sr.pvalue:.2f})")
    print(f"  tie-cal acc  = {acc:.3f}  95%CI[{accci[0]:.3f},{accci[1]:.3f}]  (0.5=chance)")
    print(f"  concordant={conc} discordant={disc} ties={n-conc-disc}")


print("=== SYSTEM-LEVEL (aggregate) ===")
print("Full 944-clip AQA means:  SCORE=%.4f  rank=%.4f  BON=%.4f"
      % (np.mean(list(aqa['SCORE'].values())), np.mean(list(aqa['rank'].values())),
         np.mean(list(aqa['BON'].values()))))
for task in ("Text Alignment", "Audio Quality"):
    for filt in (True, False):
        means = cmos(task, filt)
        for model in ("BON", "rank"):
            hs = [v for (m, f), v in means.items() if m == model]
            das = [aqa[model][f] - aqa["SCORE"][f] for (m, f), v in means.items()
                   if m == model and f in aqa[model]]
            tag = "filtered" if filt else "raw"
            print(f"  {task:15} {model:4} {tag:8}: human CMOS mean={np.mean(hs):+.3f} (n={len(hs)}) | "
                  f"AQA delta(cand-SCORE) mean={np.mean(das):+.4f}")

print("\n=== PER-ITEM CORRELATION ===")
for task in ("Text Alignment", "Audio Quality"):
    analyze(task, True, "submission-filtered")
    analyze(task, False, "all raters (raw)")
