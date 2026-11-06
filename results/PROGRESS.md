# AQAScore human-validation — progress log

Context: a reviewer asked for human validation of AQAScore (Audio Flamingo 3
VQAScore, `audio-flamingo/scripts/eval_aqa.py`) against human preference. Plan is
to follow VQAScore's own validation methodology (arXiv 2404.01291) and correlate
AQAScore against the project's CMOS survey.

## 1. Located the CMOS survey

App: `2026_SPL_ITS/icassp_round/mos-page` (Flask + SQLite), separate from the unrelated
videogen MOS app of the same name elsewhere in the filesystem. Two task types,
both scored on a −3..+3 scale against a `SCORE` reference sample:
- **Audio Quality** — naturalness only
- **Text Alignment** — content correctness only, ignore naturalness

Raw DB: `instance/2026_icassp_its.db` (12 users, 220 scores; 1 user incomplete).

## 2. Reverse-engineered the published (filtered) table

The submitted paper table doesn't match the raw DB average — it excludes some
raters. Found the exact exclusion by brute-force search over rater subsets,
reproducing the published numbers exactly:

- **Audio Quality**: exclude rater **"woo"** (uid 8) — gave identical scores to
  both candidates on every item, i.e. no discrimination.
- **Text Alignment**: keep **only** rater **정재욱** (uid 4) — exclude 남기현,
  배시우, 정채영. Reduces Text Alignment to n=10 (one rater), vs n=60 for Audio
  Quality (6 raters).

Saved to `results/submission_results.json` (full detail + exclusion rationale)
and `results/submission_results_per_file.csv` (flat, for joining with automatic
metrics).

Published/reproduced table:

| Axis | vs. Naive sampling (BON) | vs. Rank aggregation (rank) |
|---|---|---|
| Text-Consistency | −0.60 ± 0.63 (n=10) | −0.10 ± 0.65 (n=10) |
| Audio-Quality | −0.47 ± 0.36 (n=60) | −0.22 ± 0.29 (n=60) |

## 3. Reviewed VQAScore's validation methodology

Read arXiv 2404.01291 in full (main text + Appendix F). Key takeaways applied to
our plan:
- Primary stat is **pairwise accuracy with tie calibration** (Deutsch et al.,
  EMNLP'23), not raw Pearson/Kendall — handles ties in human ratings, which
  Kendall ignores. Pearson/Kendall reported secondarily for completeness.
- Protocol varies by human-label type: Likert-scale benchmarks (TIFA160) →
  pairwise accuracy; pairwise A/B preference (Pick-a-Pic) → binary accuracy;
  binary match/no-match (DrawBench etc.) → AUROC.
- Our CMOS setup (signed candidate-vs-reference score) maps most closely onto
  the **Pick-a-Pic** protocol once ternarized into ABX preference (prefer
  candidate / tie / prefer reference), with continuous Pearson/Spearman against
  the raw signed score as the secondary check — mirrors VQAScore's own reporting
  of both a primary and secondary stat.
- Flagged for the write-up: our N is much smaller than VQAScore's benchmarks
  (n=60 / n=10 vs. hundreds–thousands), and Text Alignment's n=10 rests on a
  single rater — plan to report bootstrap CIs on the correlation/accuracy
  itself to preempt "is this even significant?".

## 4. Investigated a rater-inversion hypothesis for Text Alignment (open)

Hypothesis: the 3 excluded Text-Alignment raters (남기현, 배시우, 정채영) may have
misunderstood the comparison direction (UI has the score buttons sandwiched
between the Reference player and the "Sample N" player — plausible source of
confusion) rather than being genuinely low-quality, and negating their scores
might recover a valid larger sample.

Checked via item-level Pearson correlation against the trusted rater (정재욱) on
each rater's overlapping files (not just comparing aggregate means):

| Rater | n overlap | corr. with 정재욱 |
|---|---|---|
| 남기현 | 14 | −0.208 |
| 배시우 | 12 | +0.287 |
| 정채영 | 12 | −0.063 |

None are close to the ~−1 a genuine sign-flip should produce. 배시우 is mildly
*positively* correlated (contradicts inversion for that rater); inverting
배시우 would also flip their BON/rank scores to opposite signs from each other,
which a consistently-backwards rater shouldn't do. Only 정채영 fits the
"inverted" story at all, and weakly.

**Status: unresolved / awaiting decision.** Options on the table, not yet run:
(a) invert all three anyway, (b) invert only 정채영, (c) compute both and
compare. Recommendation so far: the data doesn't strongly support blanket
inversion — leaning toward NOT using this to inflate the Text-Alignment N
without a stronger signal.

## 5. Prepared samples for AQAScore scoring

`BON`, `rank`, `SCORE` under `static/samples/2026_icassp_its/` are symlinks into
a collaborator's (jjm) read-only directories on `/mnt/bear2` — couldn't write
`rewards.json` there directly. Staged a local mirror instead:

```
results/aqa_input/{BON,rank,SCORE}/
  ├── *.wav          (symlinks to the real files, 944 each)
  └── rewards.json   ({index, prompt, output_file}, prompt = caption text
                       from static/samples/2026_icassp_its/captions/*.txt)
```

Verified all 944 captions have a matching wav in every one of BON/rank/SCORE/GT
(no gaps). This covers the full AudioCaps-test-derived set, not just the ~16
file-ids actually rated in the Text-Alignment survey — scoring the full set is
cheap (~3.5 min/dir per `AQA_EVAL.md`) and leaves room to use more file-ids
later; will filter down to the surveyed files when computing the correlation.

## Next steps

1. Resolve the rater-inversion question (§4) — decide whether/how to adjust the
   Text-Alignment filtered result before it feeds into the correlation.
2. Run `eval_aqa.py` on `results/aqa_input/{BON,rank,SCORE}` — **blocked on GPU
   reservation** (`gpu_reserve.py --status-only` needs `GPU_MMAI_PASSWORD` set;
   not available in this session).
3. Join per-file AQAScore(candidate) − AQAScore(SCORE) delta against
   `submission_results_per_file.csv` (Text Alignment rows) and compute:
   - tie-calibrated pairwise accuracy (ABX-style, per VQAScore's Pick-a-Pic
     protocol)
   - Pearson/Spearman on the continuous delta vs. mean CMOS score
   - bootstrap CI on both, given small N

## 6. Ran AQAScore and computed the correlation (2026-07-21) — DONE

Reserved gn21 GPU2 (4090), ran `eval_aqa.py` on `aqa_input/{BON,rank,SCORE}`
(944 clips each). Verified the calculation first: `scoring.py` implements
`AQA = exp(-CrossEntropy(answer_logits,"Yes")) = P("Yes")`, no system prompt,
question `'Does this audio contain "{}"? Please answer yes or no.'` — matches
AQA_EVAL.md / VQAScore. Outputs in `aqa_output/{BON,rank,SCORE}_aqa.json`;
joined per-file table in `aqa_cmos_joined.csv`; analysis in `analyze_corr.py`.

**System-level (aggregate) — AQAScore AGREES with humans:**
Full 944-clip AQA means: **SCORE 0.9086 > rank 0.9076 > BON 0.8933**. AQAScore
ranks the proposed SCORE method above naive best-of-N (BON), ties rank agg —
matches the paper's claim and the filtered CMOS (humans rated BON strongly
negative vs SCORE: TA −0.60, AQ −0.54; AQA(BON)−AQA(SCORE) also negative).

**Per-item — NO correlation (but underpowered, not a clean negative):**

| Axis | filter | n | Pearson r (p) | Spearman | tie-cal acc |
|---|---|---|---|---|---|
| Text Alignment | filtered | 20 | −0.03 (.90) | +0.09 | 0.55 |
| Text Alignment | raw | 32 | +0.07 (.72) | −0.02 | 0.44 |
| Audio Quality | filtered | 30 | +0.06 (.77) | +0.08 | 0.58 |
| Audio Quality | raw | 30 | +0.03 (.89) | +0.04 | 0.55 |

All |r|≤0.07, all p>0.68, tie-cal accuracy ≈ 0.5 (chance); every bootstrap CI
straddles chance. Two reasons per-item validation fails here, both about power
not metric validity: (a) AQAScore is **ceiling-saturated** (~0.89–0.91) so
per-file deltas are tiny (~0.01–0.03 abs) — almost no dynamic range to
discriminate; (b) human data is thin/noisy — TA filtered is a single rater
(n=10/condition), ~half the joined items are ties.

**CMOS survey issue (resolves §4):** filtering flips the Text-Alignment sign.
Filtered (정재욱 only) → BON −0.60 / rank −0.10 (SCORE preferred, matches paper).
Raw all-4-raters → BON **+0.16** / rank **+0.62** (candidates preferred,
opposite). The submission conclusion on Text Alignment rests entirely on keeping
1 of 4 raters. The §4 inversion hypothesis stays unsupported (per-item corr with
정재욱 near 0, not −1) — so the 3 excluded raters are best read as genuine
disagreement/noise, not sign-flipped. Recommend NOT inflating TA via inversion;
report Text-Alignment human validation as single-rater/underpowered and lean on
the Audio-Quality axis + system-level AQA agreement.

**Bottom line for the reviewer:** AQAScore agrees with human preference at the
**system level** (its intended use — ranking methods), reproducing SCORE > BON.
A **per-item** human-vs-AQA correlation is inconclusive here due to metric
ceiling saturation and a tiny/noisy CMOS sample, not a demonstrated failure of
the metric.

## 7. BATON external validation — per-sample AQA↔human correlation IS significant (2026-07-22) — DONE

The ICASSP CMOS set is too small and ceiling-saturated to test per-sample
correspondence (the exact test VQAScore uses). Validated AQAScore instead on
**BATON** (Liao et al., github.com/hannieliao/Baton), a public text-to-audio
dataset with per-clip human preference labels — Tango-generated AudioCaps clips,
binary human `feedback` (1 = audio faithfully contains the caption, 0 = not),
which is exactly what AQAScore measures.

Data at `/mnt/lynx3/users/jhkim/data/Baton/` (audio scraped from the authors'
Google Drive via `embeddedfolderview` — gdown's folder mode caps at 50/folder).
Ran `eval_aqa.py` on both human-annotated (HD) axes on gn28 GPU2. Outputs in
`results/baton_aqa/{2label_Integrity,3label_Temporal}_HD_Audio_aqa.json`; joined
per-clip table `results/baton_aqa/baton_aqa_human_joined.csv`; analysis
`results/baton_corr.py`. Protocol = VQAScore's binary-match test: AUROC of the
continuous AQAScore vs the binary human label (primary), plus point-biserial r,
Mann-Whitney, within-caption pairwise accuracy; bootstrap 95% CIs.

| Axis | n | pos/neg | AQA good vs bad | AUROC (95% CI) | point-biserial r (p) | within-caption pairwise acc |
|---|---|---|---|---|---|---|
| Integrity (2label) | 1049 | 656/393 | 0.822 vs 0.740 | **0.624 [0.589, 0.659]** | +0.192 (3.4e-10) | 0.624 (869 pairs) |
| Temporal (3label) | 928 | 413/515 | 0.765 vs 0.676 | **0.598 [0.561, 0.634]** | +0.159 (1.2e-06) | 0.618 (918 pairs) |

**Both axes: AQAScore agrees with human preference at the per-sample level, and
both 95% CIs exclude chance (0.5).**

> **Correction (2026-07-30, protocol audit — see §9).** An earlier version of this
> paragraph claimed our magnitudes "sit squarely in VQAScore's own reported range
> (GenAI-Bench pairwise acc 63.3, TIFA160 71.2)". **That comparison is invalid and
> is withdrawn.** VQAScore's pairwise numbers are Deutsch et al. tie-*calibrated*
> accuracies on 1–5 Likert benchmarks; our "within-caption pairwise acc" column is
> an AUROC-equivalent concordance statistic on binary labels. Do not compare the
> two in either direction. The **AUROC** column *is* protocol-matched to VQAScore
> (they report AUROC for exactly this label type) and stands unchanged.

**Why BATON works where ICASSP-CMOS didn't (confirms §6's power/saturation read,
not a metric fix):** BATON has (a) a real negative class (393/515 human-"bad"
clips) and (b) genuine AQA dynamic range (mean 0.79/0.72, sd 0.21/0.28, spanning
0.05–1.00), versus the ICASSP set's ceiling (~0.91, no bad class, deltas ~0.02).
Given a discriminable signal and n~1000, the per-sample correlation the reviewer
asked for is clearly present and significant.

**Reviewer response:** report BATON per-sample AUROC/pairwise (properly-powered,
VQAScore's own protocol) as the primary evidence of correspondence, with the
ICASSP system-level SCORE>BON agreement as secondary. Do NOT claim per-sample
correspondence from the ICASSP CMOS data alone — it is underpowered for that.

## 8. CLAP-score baseline on BATON (2026-07-27) — DONE

§7 established AQAScore beats *chance*, but had no competing metric, so no
comparative claim was supported. Ran the standard **CLAP score** (cosine
similarity of LAION-CLAP `laion/clap-htsat-unfused` text vs. audio embeddings —
the metric AudioLDM/Tango report) on the exact same BATON clips, gn30 GPU1.
Scripts `clap_score.py`, `baton_compare.py`, `baton_incr.py`; outputs
`baton_aqa/{...}_clap.json`, `baton_aqa/baton_metric_comparison.json`.

| Axis | metric | mean good / bad | AUROC (95% CI) | r (p) | within-caption pairwise |
|---|---|---|---|---|---|
| Integrity (n=1049) | AQAScore | 0.822 / 0.740 | 0.6237 [0.589, 0.659] | +0.192 (3.4e-10) | 0.624 |
| Integrity | **CLAP** | 0.569 / 0.508 | **0.6676 [0.634, 0.700]** | +0.288 (1.8e-21) | 0.671 |
| Temporal (n=928) | **AQAScore** | 0.765 / 0.676 | **0.5977 [0.561, 0.634]** | +0.159 (1.2e-06) | 0.618 |
| Temporal | CLAP | 0.573 / 0.546 | 0.5756 [0.538, 0.612] | +0.132 (5.6e-05) | 0.621 |

**Honest read — CLAP is NOT worse; on Integrity it is better.**
ΔAUROC (AQA − CLAP): Integrity **−0.044, 95% CI [−0.086, −0.002] → CLAP
significantly better**; Temporal **+0.022, CI [−0.021, +0.065] → tie**. Any
claim that AQAScore outperforms CLAP on faithfulness is unsupported.

**What does hold — the two metrics are largely independent and complementary:**
Spearman(AQA, CLAP) = +0.25 (Integrity) / +0.35 (Temporal), i.e. they mostly
disagree. Residualizing each on the other, both retain significant signal about
the human label (AQA|CLAP r=+0.134 p=1.4e-05 and +0.122 p=1.9e-04; CLAP|AQA
r=+0.253 and +0.085). Combining them (logistic reg., 5-fold OOF) lifts Integrity
AUROC to **0.681** vs 0.668 CLAP-alone / 0.624 AQA-alone; Temporal is flat
(0.599 vs 0.598 AQA-alone).

**Implication for the rebuttal:** the defensible claim is *validity*, not
superiority — AQAScore tracks human preference significantly above chance, at a
level comparable to (Temporal) or below (Integrity) the standard CLAP baseline,
while carrying signal CLAP does not. If the paper's method needs a
best-available faithfulness metric, this is an argument for CLAP or for an
AQA+CLAP combination, and that gap should be acknowledged rather than hidden —
a reviewer can run CLAP on BATON in an afternoon.

## 9. Protocol audit — is our evaluation identical to VQAScore's? (2026-07-30) — DONE

Read VQAScore (arXiv 2404.01291v2) primary text §"image evaluation benchmarks"
and Appendix F to pin down exactly which statistic they use for which label type,
then re-ran every statistic against a reference implementation.
Script `verify_protocol.py` (+ `verify_delta.py`) in the job tmp dir.

**What VQAScore actually does (verbatim):**
- *"We follow SeeTrue to report AUROC on DrawBench, EditBench, and COCO-T2I."*
  SeeTrue = **binary match-or-not labels** — the same label type as BATON.
  *"We follow their original evaluation protocols to report the AUROC ... taking
  into account all possible classification thresholds."*
- *"For TIFA160 and Flickr8K, we evaluate pairwise accuracy as advocated by
  Deutsch et al."* — used **only on 1–5 Likert benchmarks**, with tie calibration.
- Pearson/Kendall are reported for completeness in Appendix F, on the Likert sets.
- **No confidence intervals and no significance tests anywhere in the paper.**

**Audit verdict, statistic by statistic:**

| Statistic | VQAScore | Ours | Verdict |
|---|---|---|---|
| AUROC (primary) | yes, for binary labels | `U/(n_pos·n_neg)` | **MATCH.** = `sklearn.roc_auc_score` to 1e-16 on all 4 axis×metric cells. Right statistic for BATON's label type. |
| Pearson r | Appendix, Likert sets | point-biserial | **Same number** (point-biserial ≡ Pearson with a binary variable; verified identical to 1e-9). Supplementary only. |
| Pairwise accuracy | Deutsch tie-calibrated, all pairs, Likert | within-caption, human-ties dropped, 0.5 for exact tie | **NOT the same statistic.** See below. |
| Bootstrap 95% CI | not used | percentile bootstrap, 10k | **Addition** (stricter than them), but resampling unit was wrong — see below. |
| best-threshold acc | not used (their Pick-a-Pic "binary accuracy" is 2AFC) | tuned on the same data | **Do not report** — optimistically biased. |

**Finding 1 — our "pairwise accuracy" is not VQAScore's pairwise accuracy.**
Ours drops human-tie pairs and gives 0.5 credit for exact metric ties, which is
just the AUROC estimator restricted within caption: Integrity gives 0.6237,
*identical to our AUROC* 0.6237. Computing Deutsch et al.'s real tie-calibrated
accuracy over all M(M−1)/2 pairs gives a completely different number:

| Axis | metric | ours (reported) | true Deutsch tie-calibrated | all-ties baseline |
|---|---|---|---|---|
| Integrity | AQAScore | 0.6237 | 0.5323 (ε*=0.82) | 0.531 |
| Integrity | CLAP | 0.6709 | 0.5358 (ε*=0.27) | 0.531 |
| Temporal | AQAScore | 0.6176 | 0.5055 (ε*=0.95) | 0.506 |
| Temporal | CLAP | 0.6209 | 0.5055 (ε*=0.69) | 0.506 |

With binary labels ~51–53% of all pairs are human ties, so tie calibration
degenerates: the optimal ε makes the metric declare (nearly) everything a tie and
the statistic collapses onto the trivial all-ties baseline. This is a property of
the *statistic*, not a failure of AQAScore — and it is precisely why VQAScore uses
AUROC rather than pairwise accuracy on binary benchmarks. **Action: rename the
column to "within-caption concordance" or drop it (it is redundant with AUROC),
and never juxtapose it with VQAScore's published pairwise numbers.**

**Finding 2 — bootstrap resampling unit was wrong (conclusions survive).**
BATON has ~5 clips per caption (211 captions / 1049 clips; 189 / 928), so clips
are not independent. Switching to a caption-clustered bootstrap widens the CIs:

| Axis | metric | i.i.d.-clip CI | caption-clustered CI |
|---|---|---|---|
| Integrity | AQAScore | [0.5888, 0.6586] | [0.5852, 0.6635] |
| Integrity | CLAP | [0.6338, 0.7005] | [0.6286, 0.7047] |
| Temporal | AQAScore | [0.5609, 0.6338] | [0.5629, 0.6342] |
| Temporal | CLAP | [0.5380, 0.6122] | [0.5363, 0.6154] |

All four still **exclude 0.5**, so §7's validity claim is unaffected — but the
clustered CIs are the ones to quote.

**Finding 3 — §8's CLAP-superiority claim does NOT survive clustering.**
ΔAUROC (AQA − CLAP) on Integrity = −0.0439: i.i.d. CI [−0.0862, −0.0015]
(significant) but **clustered CI [−0.0927, +0.0047] — straddles 0, not
significant.** Temporal is not significant either way. So the correct statement is
*CLAP is numerically ahead on Integrity but the difference is not statistically
established*; §8's "CLAP significantly better" is too strong. The rebuttal-relevant
conclusion is unchanged in direction: still do **not** claim AQAScore beats CLAP.

**Finding 4 — the ICASSP-CMOS "tie-cal acc" column (§6) is also mislabelled.**
`analyze_corr.py:tie_cal_acc` awards 0.5 whenever either sign is exactly zero and
otherwise scores sign agreement. It performs **no tie calibration** (no ε search),
so it is a sign-agreement accuracy, not Deutsch et al.'s statistic. §6's conclusion
(≈chance, underpowered) is unaffected — the column just needs renaming to
"sign-agreement acc". Note also that VQAScore *removed* all tied labels for its
Pick-a-Pic binary accuracy rather than giving them half credit.

**Overall:** the one statistic the rebuttal actually leans on — per-sample AUROC
with a bootstrap CI excluding 0.5 — is protocol-matched to VQAScore and verified
correct. The pairwise-accuracy column is not VQAScore's statistic and should be
dropped; CIs should use the caption-clustered bootstrap; and §8's CLAP
significance claim must be softened.

**Finding 5 — the group means need CIs too, and they must also be clustered.**
The reported means (0.822 / 0.740) carried no uncertainty. Prompt-clustered
bootstrap (`mean_ci.py`, 10k resamples):

| Axis | preferred (mean, 95% CI) | rejected (mean, 95% CI) | Δ (95% CI) |
|---|---|---|---|
| Integrity | 0.822 [0.799, 0.844] (sd 0.189, n=656) | 0.740 [0.703, 0.773] (sd 0.228, n=393) | **+0.083 [+0.052, +0.116]** |
| Temporal | 0.765 [0.729, 0.800] (sd 0.257, n=413) | 0.676 [0.638, 0.715] (sd 0.290, n=515) | **+0.089 [+0.056, +0.124]** |

Both Δ CIs exclude 0. Note the naive i.i.d. interval is ~1.5x too narrow
(Integrity preferred ±0.015 naive vs ±0.022 clustered; rejected ±0.023 vs ±0.035)
— same clustering issue as Finding 2. Quote Δ with its CI rather than arguing from
non-overlapping per-group CIs.
