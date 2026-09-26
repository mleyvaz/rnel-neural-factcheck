# Experiment 23 — the refined evidential (RNEL) head of Smarandache's Definition 8.1 on real data (preregistration v2)

Written 2026-09-26 and committed to git BEFORE any model is trained. CPU only, no API, no new human
annotation (existing dataset annotations only). v1 (`PREREGISTRATION_DRAFT.md`, kept) was reviewed
adversarially by Codex (`review/codex_prereg_review.md`); v2 answers that review.

**What was run before this preregistration (disclosure).** The raw data were downloaded and class
counts computed (`data/README.md`). Feature extraction with the frozen encoder (`features.py`) was
started before the review and is still running; it produces only encoder outputs (NLI logits and
[CLS] vectors), no model was trained and no performance was computed or inspected. A leakage audit
(duplicate normalised claims: 67 groups in AVeriTeC train+dev, 46 in Climate-FEVER, 16 with
inconsistent labels) was run on the raw labels.

## Question
Does the typed head separate contradictory evidence (glut) from the other evidential situations better
than a flat classifier or a simple rule, on real fact-verification data? The labels are operational
dataset labels, not ground-truth neutrosophic states.

## Data and label mapping
| Dataset | T | F | glut (C) | gap (N) |
|---|---|---|---|---|
| AVeriTeC train+dev (3,568 claims) | Supported | Refuted | Conflicting Evidence/Cherrypicking (233) | Not Enough Evidence (317) |
| Climate-FEVER (1,535) | SUPPORTS | REFUTES | DISPUTED (154) | NOT_ENOUGH_INFO (474) |

- **Primary dataset for R2: AVeriTeC.** Its C class mixes conflict with cherry-picking; the claim is
  narrowed to that combined label. **Climate-FEVER is a rule-recovery benchmark**: DISPUTED is defined
  as "both supporting and refuting evidence", so a result there is not independent construct validation.
- **NEE mapping is co-primary:** RNEL is trained with NEE → gap (`rnel`) and with NEE → ambiguity
  (`rnel_neeU`); both are reported for every hypothesis. Flat methods are unaffected.

## Representation, splits, training (fully specified)
- Encoder: frozen `cross-encoder/nli-deberta-v3-base` (SNLI+MNLI). Per (evidence, claim) pair: [CLS]
  (768), 3 NLI logits, 3 NLI probabilities; per claim: mean and max over its evidence (1,548 features),
  standardised on the training part. AVeriTeC evidence = "Q: … A: … [boolean explanation]".
- Grouped, stratified 5-fold cross-validation (StratifiedGroupKFold, random_state 0); groups =
  lower-cased claim text with non-alphanumerics removed, so duplicates never cross folds. In each fold
  15% of training groups are held out for validation (GroupShuffleSplit, random_state = fold).
- MLP: 1,548 → 256 → 256 (ReLU, dropout 0.2), AdamW, batch 64, ≤ 80 epochs, early stopping on
  validation macro-F1 (patience 10; ties keep the earlier epoch). Inverse-frequency class weights for
  the trained networks and the logistic regression (no calibration claims are made).
- Grid (chosen on fold 0 validation, reused in all folds): lr {1e-3, 3e-4}, weight decay {1e-4, 1e-2};
  RNEL also W {2, 5} and regulariser {0, 0.1}; flat EDL also regulariser {0, 0.1}. Seeds 0–4.
- **Prediction (all neural methods):** argmax of the posterior (softmax, or α/S for EDL/RNEL) over the
  labels present in the dataset (for RNEL: T, F, glut, gap; for `rnel_neeU`: T, F, glut, amb).

## Methods
1. `rnel`, `rnel_neeU`: NeutroEvidentialHead (K = 2: T, F, glut, amb, gap), loss = neutro_edl_loss
   (class-weighted).
2. `softmax`: 4 flat classes, cross-entropy.
3. `edl`: 4 flat classes, same Dirichlet loss with W = 4 (Sensoy-style, the reduction of 1).
4. `logreg`: logistic regression on the 6 pooled NLI probabilities (capacity-light learned baseline).
5. `rule`: some evidence with p(entail) > t_e and some with p(contradict) > t_c → C; only entail → T;
   only contradict → F; neither → N. (t_e, t_c) on a 14×14 grid in [0.3, 0.95], tuned per fold on
   validation macro-F1.
6. `sl` (cumulative fusion, Jøsang) and `slavg` (averaging fusion) of opinions b = p(entail),
   d = p(contradict), u = p(neutral); typed by thresholds on the largest pairwise conflict and on u
   (18×19 grid), tuned per fold on validation.

## Confirmatory hypotheses
Bootstrap: 2,000 draws resampling claim groups (cluster bootstrap) and, inside each draw, training
seeds with replacement; differences are paired. "95% CI" = percentile interval.
- **R1 (no cost, non-inferiority, margin 0.02 fixed a priori):** macro-F1(RNEL) − macro-F1(flat EDL)
  has lower 95% bound > −0.02, on AVeriTeC (primary) and Climate-FEVER.
- **R2 (main):** F1 on C of RNEL − softmax > 0 AND RNEL − rule > 0 (lower 95% bounds > 0) on
  AVeriTeC. The same conjunction on Climate-FEVER is reported as secondary (rule recovery). R2 is one
  conjunctive claim; no further multiplicity correction; R1 is a separate family.
Both are evaluated for `rnel` and `rnel_neeU` (co-primary; each reported).

## Descriptive only (no confirmatory claim; reasons from the review)
- **D1, ChaosNLI ambiguity:** trained on typed labels where "top vote < 60%" → amb; this threshold is
  close to the evaluation target, so the comparison is structurally favoured. Reported: Spearman with
  human entropy of RNEL I_amb and its α_amb/S, softmax p(U) and its entropy, EDL p(U), RNEL I_glut;
  macro-F1. Task contamination: the encoder was trained on MNLI, whose development items ChaosNLI uses.
- **D2:** RNEL components by true class; share with T + F > 1 (Belnap reading, which counts glut
  evidence in both T and F).
- **D3, vacuity:** the v1 out-of-domain test is withdrawn (it would measure dataset identity). Instead:
  AUROC for detecting the method's own errors, RNEL I_vac vs EDL vacuity, 1 − max softmax, and minus the
  number of evidence pieces.
- **D4:** the reduction RNEL → flat EDL is an algebraic property; it is checked as implementation
  validation (tests/test_neutro_edl.py and analyze.py), not as a hypothesis.
- F1 per class, confusion matrix, PR-AUC for C, comparisons with sl, slavg, logreg.

## Power and limits (stated in advance)
C has 233 examples (AVeriTeC) and 154 (Climate-FEVER) in the out-of-fold evaluation; small F1
differences will not be detectable. Hyperparameters chosen on fold 0 are reused, so fold 0's test
part was not used for selection but its training data were. All results are reported, including
failed hypotheses; deviations go to RESULTS.md.
