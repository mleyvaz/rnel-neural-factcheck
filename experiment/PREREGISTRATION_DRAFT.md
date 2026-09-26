# Experiment 23: The refined evidential (RNEL) head of Smarandache's Definition 8.1 on real data (preregistration DRAFT)

**Status: DRAFT (2026-09-26).** Only the data have been downloaded and counted (`data/README.md`).
No features have been computed and no model has been trained. This file becomes the preregistration only
when it is renamed to `PREREGISTRATION.md` and committed, and that happens only after the author
approves it.

- **Compute:** CPU only, no API calls.
- **Annotation:** no new human annotation. All labels are annotations that already exist in the datasets.

## Question
The technical note on the RNEL network sent to Florentin reported 97.7% type accuracy. That figure came
from five well-separated synthetic clusters, so it tests nothing. This experiment asks whether, on real
data, the typed head separates the four evidential situations better than a network with flat classes
or a simple rule:
- C: contradictory evidence (glut);
- N: no evidence (gap);
- U: ambiguity (human disagreement);
- G: vacuity (out of distribution).

The head is `nel/neutro_edl.py` (NeutroEvidentialHead). K = 2 classes (true, false) give 5 types: T, F,
glut(T,F), amb(T,F) and gap.

## Label mapping (fixed in advance)

| Dataset | T | F | Glut (C) | Gap (N) | Ambiguous (U) |
|---|---|---|---|---|---|
| AVeriTeC | Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence | — |
| Climate-FEVER | SUPPORTS | REFUTES | DISPUTED | NOT_ENOUGH_INFO | — |
| ChaosNLI-MNLI | entailment | contradiction | — | neutral | top label < 60% of the votes |

- **Why NEE maps to the gap:** "Not enough evidence" is mapped to the gap (Belnap's "neither", no
  information), as in JoLLI.
- **Sensitivity analysis:** the same analysis is reported with NEE mapped to U.
- **Known limitation 1:** the AVeriTeC glut class mixes real conflict with cherry-picking (true but
  misleading claims), so it is not a pure glut. It is reported separately.
- **Known limitation 2:** Climate-FEVER's DISPUTED is defined by a rule on the evidence labels (both
  support and refutation present). The rule baseline below therefore has a structural advantage there.

## Representation
- **Encoder:** frozen `cross-encoder/nli-deberta-v3-base`, trained on SNLI and MNLI. It was not trained
  on FEVER, AVeriTeC or Climate-FEVER.
- **Contamination:** ChaosNLI uses items from the MNLI development set, whose labels the encoder did not
  see in training. The texts may still have been seen, and this is declared as a possible contamination.
- **Per piece of evidence:** each (evidence, claim) pair gives its 3 NLI logits plus the pooled vector.
  - AVeriTeC evidence: "Q: … A: …".
  - Climate-FEVER evidence: the sentence.
  - ChaosNLI evidence: the premise, with the hypothesis as the claim.
- **Per claim:** the evidence set is combined with mean and max pooling (DeepSets), followed by a
  2-layer MLP.

## Methods (same encoder and MLP; only the output layer and loss change)
1. **RNEL:** the NeutroEvidentialHead with the neutro_edl_loss.
2. **Softmax, 4 or 5 flat classes:** cross-entropy.
3. **Flat EDL** (Sensoy et al., 2018), with 4 Dirichlet classes.
4. **Rule on NLI outputs**, with no training of the MLP:
   - some piece of evidence is entailment and some is contradiction → C;
   - only entailment → T;
   - only contradiction → F;
   - none → N;
   - the probability thresholds are tuned on validation.
5. **Subjective Logic:**
   - each piece of evidence gives an opinion b = P(entail), d = P(contradict), u = P(neutral);
   - the opinions are combined by cumulative fusion;
   - the result is typed by thresholds on the conflict degree and on u, tuned on validation.

## Splits and training
- **AVeriTeC:** train on train (15% held out for validation), test on dev.
- **Climate-FEVER:** a stratified 70/15/15 split, three times.
- **ChaosNLI:** a stratified 50/50 split (train U / test U).
- **Seeds:** 5 per method.
- **Tuning:** hyperparameters (learning rate, weight decay, W, the regulariser) are chosen from a
  predefined grid by macro-F1 on validation only.
- **Class imbalance:** class weights inverse to frequency, for every method.

## Hypotheses
- A hypothesis holds on a dataset if the 95% bootstrap interval over claims (2,000 resamples, median
  over seeds) excludes the null.
- The main hypothesis (R2) must hold on both datasets.

The hypotheses:
- **R1 (no cost):** RNEL macro-F1 ≥ flat EDL − 0.02 (non-inferiority), on both datasets.
- **R2 (main, contradiction):** RNEL's F1 on C > flat softmax and > the NLI rule, on AVeriTeC and on
  Climate-FEVER.
- **R3 (ambiguity):** on ChaosNLI test, Spearman(I_amb, human entropy) > Spearman(I_glut, human
  entropy), CI of the difference > 0. RNEL's I_amb must also beat the entropy of the softmax (method 2)
  as a predictor of human entropy.
- **R4 (vacuity):** trained on AVeriTeC and applied to Climate-FEVER (and the reverse), the AUROC of
  I_vac for separating out-of-domain from in-domain examples is > 0.60. This is expected to fail,
  because the technical note already showed G ≈ 0 far from the training data. It is reported whatever
  the result.
- **R5 (reduction, sanity check):** without the glut, ambiguity and gap types and with W = K, RNEL
  reproduces flat EDL (difference < 1e-6 on the same weights). The existing test is repeated on real
  features.

## Secondary measures (descriptive)
- Confusion matrices.
- F1 per class (T, F, C, N).
- Calibration (ECE) of T.
- Components T, F, I_glut, I_amb, I_gap, I_vac by true class.
- Share of examples with T + F > 1 under the Belnap reading. Reported for transparency: this share
  counts glut evidence in both T and F.

## Contingencies and reporting
- Every result is reported, including failed hypotheses.
- Deviations are listed in `RESULTS.md`.
- **If the rule (method 4) matches RNEL on C,** the conclusion will be that the refined structure adds
  nothing over the rule on these data. That result goes into the paper with Florentin.
- **Licences:** the raw data are not redistributed (CC BY-NC).
