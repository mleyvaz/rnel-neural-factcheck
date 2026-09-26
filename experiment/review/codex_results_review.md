Maikel, the numerical transcription is mostly accurate, and the R1/R2 Boolean logic matches the preregistered inequalities. However, the confirmatory analysis has a fatal cross-fold hyperparameter-selection leak. R1 and R2 should not be treated as valid confirmatory verdicts until the models are rerun with genuinely nested tuning.

## Fatal

### 1. Hyperparameters use the labels of later outer-test folds

The code selects each neural method’s hyperparameters using fold 0 once, then reuses them for all outer folds ([run.py](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/run.py:255)). But fold 0’s training/validation population contains every observation assigned to outer folds 1–4.

I reconstructed the overlap:

- AVeriTeC: all 2,854 observations tested in folds 1–4 had already appeared in fold 0’s fitting or hyperparameter-validation data.
- Climate-FEVER: all 1,228 observations tested in folds 1–4 were similarly exposed.
- Approximately 15% of each later test fold was used directly in fold 0 hyperparameter validation; the remainder influenced fitting during the hyperparameter search.

Thus 80% of the “out-of-fold” predictions are not out of sample for the complete model-selection procedure. This can change a verdict because the four methods are tuned independently and may benefit unequally.

As a sensitivity warning, on the only procedurally clean outer fold, AVeriTeC point differences were:

- RNEL − EDL macro-F1: −0.0347
- RNEL-NEE→U − EDL: −0.0238

Both are below the −0.02 margin, whereas the pooled analysis says the NEE→U version passes. A single fold cannot establish the corrected verdict, but it demonstrates that the reported pass is not robust to removing the contaminated folds.

This design was preregistered and partly acknowledged in [PREREGISTRATION.md](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/PREREGISTRATION.md:41) and its limitations section, so it is not a post hoc deviation. It is nevertheless a fatal design error for confirmatory OOF inference.

**Exact fix:** rerun nested grouped CV. Within each outer fold, choose hyperparameters using only that fold’s outer-training groups—either the stated 15% validation split or grouped inner CV—and then evaluate once on the untouched outer-test fold. Rerun the paired cluster-and-seed bootstrap and recompute R1/R2. Because this changes the preregistered procedure, label it a corrected confirmatory analysis or new preregistration, not the original confirmatory result.

## Major

### 2. “No better” and “no detection power” confuse failure to reject with equivalence

[RESULTS.md](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/RESULTS.md:17) says RNEL “is no better” than softmax. The test only failed to demonstrate superiority:

- AVeriTeC CI: −0.021 to +0.031
- Climate-FEVER CI: −0.032 to +0.037

Those intervals admit meaningful advantages and disadvantages. No equivalence or non-inferiority criterion against softmax was preregistered.

Likewise, the Interpretation says the components “add no detection power” ([RESULTS.md](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/RESULTS.md:80)). That conclusion is not supported by nonsignificant superiority tests and two datasets.

**Exact fix:** replace these claims with:

> RNEL did not demonstrate superior F1_C to softmax in either dataset; the confidence intervals remain compatible with modest benefit or harm.

And:

> No predictive advantage of the refined head was detected under this supervised typed-label design.

### 3. The components are not merely “a relabelling of the same posterior”

The categorical posterior used for prediction is `alpha/S`. But `I_vac = W/(N+W)`, the indeterminacy shares use evidence normalized by `N`, and the Belnap T/F components count glut evidence in both coordinates ([neutro_edl.py](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/nel/neutro_edl.py:72)). These are deterministic functions of the evidence vector and concentration, but they are not simply label renamings of the categorical posterior mean.

**Exact fix:** say:

> The head is directly supervised as a five-type Dirichlet classifier. Its reported components are deterministic summaries of that typed evidence vector, so this experiment does not test whether they emerge independently of typed supervision.

### 4. D3 does not evaluate each method’s “own errors”

`analyze.py` creates one error indicator from RNEL seed 0, then evaluates RNEL, EDL, softmax, and evidence-count scores against that same RNEL error target ([analyze.py](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/analyze.py:137)). Therefore the table does not support the heading “detecting the method’s own errors.”

Using each classifier’s own seed-0 errors gives:

| Dataset | RNEL I_vac | EDL vacuity | Softmax uncertainty |
|---|---:|---:|---:|
| AVeriTeC | 0.707 | 0.700 | 0.718 |
| Climate-FEVER | 0.616 | 0.637 | 0.678 |

The published values for EDL and softmax—0.729/0.736 and 0.631/0.648—are instead their ability to rank RNEL’s errors.

**Exact fix:** construct separate `rnel_err`, `edl_err`, and `softmax_err` targets. State explicitly which target is used for the method-independent evidence-count score. Regenerate D3 and revise its interpretation. This does not alter R1 or R2.

### 5. One co-primary R2 variant is not numerically reported

The preregistration requires both NEE mappings to be reported for every hypothesis ([PREREGISTRATION.md](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/PREREGISTRATION.md:67)). R1 reports both fully, but R2 gives estimates and intervals only for `rnel`, followed by “the `rnel_neeU` variant gives the same verdicts.”

The omitted results are:

- AVeriTeC versus softmax: +0.008 [−0.018, 0.034]
- AVeriTeC versus rule: +0.122 [0.069, 0.171]
- Climate versus softmax: +0.000 [−0.028, 0.031]
- Climate versus rule: +0.112 [0.046, 0.175]

**Exact fix:** report all four estimates and intervals alongside the ordinary RNEL results.

## Minor

### 6. “The ambiguity share is 0” is numerically false

The summary contains small but nonzero ambiguity shares, approximately `1.4e-6` to `4.1e-5` for the reported class means; individual values reach approximately 0.010. Softplus outputs cannot be exactly zero.

**Exact fix:** change “is 0” to “is numerically near zero at the displayed precision” and, preferably, give a range.

### 7. “All numbers come from summary.json” is literally inaccurate

The experimental table, intervals, descriptive metrics, counts, and rounded percentages agree with `summary.json`. I found no transcription error among those values.

Exceptions to the provenance statement include:

- 5 folds, 5 seeds and 2,000 draws come from code/protocol.
- The 60.9% encoder sanity check is absent from `summary.json`; I independently reproduced it as 60.913% from `features/chaos.npz`.
- The commit identifier and training-history numbers are naturally not summary outputs.
- The categorical “ambiguity share is 0” conflicts with the nonzero summary values.

**Exact fix:** say “All reported performance estimates and intervals come from `summary.json`,” and add the encoder sanity statistic to the summary or identify its separate calculation.

### 8. Final analysis code and artifacts lack immutable provenance

`run.py` and `analyze.py` are modified relative to preregistration commit `6178466`; `RESULTS.md` and the result artifacts are untracked. The two visible changes are the disclosed four-thread setting and the disclosed `rnel_neeU` mapping repair, so I found no hidden outcome-changing code change. Nevertheless, no commit uniquely binds the final code to the JSON and predictions.

`runs.json` also omits the selected SL/SL-averaging thresholds and early-stopping epochs.

**Exact fix:** commit the final code and immutable result manifests, including hashes of feature/prediction files, package/model revision, every selected threshold, and per-fold stopping epochs.

## Checks that passed

- Dataset label mappings in `features.py` match the preregistration.
- NLI logit order is handled consistently.
- Train/test grouping and feature-standardization masks are correct within each fold.
- Inverse-frequency class weights are computed only from the current training subset.
- Prediction masks restrict RNEL to labels present in the dataset.
- The final `rnel_neeU` U→N repair is correctly applied before all R1/R2 metrics.
- The cluster-and-seed bootstrap implements the preregistered resampling and pairing rule.
- Conditional on accepting the contaminated estimates, every R1/R2 Boolean verdict follows the exact preregistered strict inequalities.
- All displayed performance numbers otherwise round correctly from [summary.json](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/results/summary.json).

Overall: the textual numbers are reliable, but the confirmatory inferential claim is not. The appropriate disposition is “R1/R2 verdicts pending clean nested-CV rerun,” with the present outputs retained as the preregistered-but-methodologically contaminated analysis.