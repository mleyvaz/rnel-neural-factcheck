# Experiment 23 — results (PREREGISTRATION.md v2, commit 6178466)

## Status of the analyses
- **Corrected confirmatory analysis (primary):** hyperparameters and thresholds are tuned inside each
  outer fold, on that fold's held-out training groups (`run.py --nested`, `analyze.py --nested`, files
  `*_nested.*`).
- **Why a correction was needed:** the preregistered procedure tuned the neural hyperparameters on
  fold 0 only and reused them in folds 1–4. The adversarial review (`review/codex_results_review.md`)
  showed that fold 0's training data contain every claim tested in folds 1–4, so 80% of the
  out-of-fold predictions were not out of sample for model selection. This is a design error in the
  preregistration, corrected after the fact.
- **The preregistered analysis** (`preds_*.npz`, `summary.json`) is kept and reported below as
  contaminated. Its verdicts are not used.

All performance estimates and intervals below come from `results/summary_nested.json`, except where
the contaminated analysis is named explicitly. Set-up:
- grouped 5-fold cross-validation: 3,442 claim groups in AVeriTeC and 1,489 in Climate-FEVER;
- 5 training seeds;
- 95% intervals from a cluster-and-seed bootstrap (2,000 draws, paired).

## Confirmatory hypotheses (corrected analysis)

**R1, non-inferiority to flat EDL (macro-F1, margin 0.02 fixed a priori).**

| | AVeriTeC | Climate-FEVER |
|---|---|---|
| `rnel` (NEE → gap) | −0.005 [−0.016, 0.005], **holds** | −0.006 [−0.022, 0.011], fails narrowly |
| `rnel_neeU` (NEE → ambiguity) | −0.008 [−0.021, 0.003], fails narrowly | +0.001 [−0.014, 0.015], **holds** |

**R2, main hypothesis: not supported.** RNEL did not demonstrate a superior F1 on C to a softmax in
either dataset or either mapping; the intervals remain compatible with a modest benefit or harm.
Differences in F1 on C, with 95% intervals:

| | AVeriTeC | Climate-FEVER |
|---|---|---|
| `rnel` − softmax | +0.015 [−0.008, 0.037] | −0.009 [−0.051, 0.034] |
| `rnel` − rule | **+0.123 [0.071, 0.176]** | **+0.111 [0.052, 0.174]** |
| `rnel_neeU` − softmax | +0.005 [−0.018, 0.028] | +0.017 [−0.019, 0.052] |
| `rnel_neeU` − rule | **+0.113 [0.063, 0.161]** | **+0.137 [0.072, 0.202]** |

## Main table (corrected analysis, mean over seeds)

| | AVeriTeC macro-F1 | AVeriTeC F1_C | Climate macro-F1 | Climate F1_C |
|---|---|---|---|---|
| RNEL | 0.483 | 0.221 | 0.465 | 0.234 |
| RNEL (NEE → U) | 0.479 | 0.211 | 0.472 | 0.259 |
| flat EDL | 0.488 | 0.223 | 0.472 | 0.242 |
| softmax | 0.469 | 0.206 | 0.467 | 0.242 |
| logistic regression (pooled NLI) | 0.337 | 0.198 | 0.362 | 0.212 |
| Subjective Logic, cumulative | 0.352 | 0.126 | 0.282 | 0.153 |
| Subjective Logic, averaging | 0.345 | 0.126 | 0.275 | 0.140 |
| NLI rule | 0.334 | 0.098 | 0.288 | 0.123 |

RNEL − Subjective Logic (cumulative), difference in F1 on C:
- AVeriTeC: +0.094 [0.042, 0.146];
- Climate-FEVER: +0.080 [0.011, 0.148].

PR-AUC for C (prevalence 0.065 in AVeriTeC and 0.100 in Climate-FEVER):

| | softmax | EDL | RNEL |
|---|---|---|---|
| AVeriTeC | 0.137 | 0.142 | 0.140 |
| Climate-FEVER | 0.184 | 0.190 | 0.177 |

## Descriptive (corrected analysis)
- **D1, ChaosNLI (not confirmatory).** The U label is defined from the vote distribution, so this
  comparison is structurally favoured. Spearman correlation with human entropy:
  - RNEL I_amb 0.234; RNEL I_glut 0.204;
  - softmax p(U) 0.217; softmax entropy 0.230;
  - EDL p(U) 0.227.

  Macro-F1: softmax 0.559, EDL 0.574, RNEL 0.567.
- **D2, Definition 8.1 components by true class** (`paper_section/section_numbers.json`):
  - The contradiction share C is larger on conflicting claims than on supported claims:
    - AVeriTeC: 0.199 vs 0.082;
    - Climate-FEVER: 0.205 vs 0.139.
  - The neither share N is largest on "not enough evidence" claims:
    - AVeriTeC: 0.282;
    - Climate-FEVER: 0.284.
  - The ambiguity share is numerically near zero, because these datasets have no ambiguity label.
    Its class means range from 8e-5 to 1.1e-4 (AVeriTeC) and 2.4e-4 to 3.6e-4 (Climate-FEVER).
  - These are deterministic summaries of evidence that was trained on typed labels. They were not
    discovered by the network.
- **D3, each score ranks the errors of its own method** (AUROC, seed 0; corrected after the review):

  | | RNEL I_vac | EDL vacuity | 1 − max softmax | − evidence count (on RNEL's errors) |
  |---|---|---|---|---|
  | AVeriTeC | 0.701 | 0.694 | 0.713 | 0.387 |
  | Climate-FEVER | 0.557 | 0.617 | 0.661 | 0.500 |

- **D4, reduction to flat EDL.** Maximum error 1.2e-7 (float32). This is implementation validation.

## Contaminated preregistered analysis (for the record)
Here RNEL (NEE → gap) failed R1 on AVeriTeC (−0.015 [−0.025, −0.004]), and R2 was not supported, with
the same pattern against the rule. Full numbers are in `results/summary.json` (computed before the D3 fix; its D3 block
still scores every method against RNEL's errors).

## Interpretation
- **No predictive advantage was detected under this design.** With typed training labels, the
  refined head is directly supervised as a five-type Dirichlet classifier, and its reported components
  are deterministic summaries of that typed evidence. So this experiment does not test whether the
  components would emerge without typed supervision.
- **Learning from the evidence clearly beats the non-learned combinations** (the rule, and Subjective
  Logic with cumulative or averaging fusion): about +0.08 to +0.14 F1 on C.
- **Next test:** whether the refinement helps when the types are not given as labels. That needs
  source-level evidence combined by the network, trained with crisp labels only.

## Deviations
1. **Hyperparameter selection corrected to nested tuning** (see "Status of the analyses").
2. **Training restarts.** The first run was stopped to free CPU for encoding; the second was stopped
   by the host for low memory. The final runs used 4 threads and one dataset per process, with code,
   seeds and splits unchanged.
3. **Analysis fixes after inspection.**
   - The `rnel_neeU` ambiguous-type predictions are now mapped back to N.
   - D3 now uses each method's own errors.
   - The SL thresholds are now recorded in `runs_nested.json`.
4. **Feature extraction** started before the preregistration commit; this was disclosed there.
5. **Encoder sanity check.** NLI argmax agrees with the ChaosNLI human majority on 60.9% of items
   (computed separately from `features/chaos.npz`).
