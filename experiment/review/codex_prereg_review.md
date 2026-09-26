# Adversarial methods review

Overall recommendation: **do not finalize either preregistration yet**. Experiment 22 has serious construct-validity and circular-gating problems. Experiment 23’s main contradiction claim is not supported by clean contradiction labels, while R3 and R5 are close to built-in successes rather than empirical tests.

I made no file changes.

## Experiment 22: direct indeterminacy

### Fatal issues

| Problem | Why it is fatal | Exact fix |
|---|---|---|
| **The bilateral gate does not require accuracy.** The draft gates only gap, glut, and abstention ([lines 74–79](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/22_direct_indeterminacy/PREREGISTRATION_DRAFT.md:74)). A model that systematically reverses every true and false claim can have gap = glut = abstention = 0 and pass. The cited theorem assumes correct answers to positive claims ([main.tex:181](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/paper_jolli/main.tex:181)). | Passing the gate does not establish the bridge needed to interpret later gaps. | Add preregistered minimum accuracy for both \(P\) and \(\neg P\), separately on T and F and per construction. For example, require lower one-sided confidence bounds above 0.80 or 0.90. Gate specifically on gap among F and glut among T, not only aggregate rates. |
| **The direct gate can be passed by abstaining on every known claim.** `I_dir` counts only “No” ([line 69](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/22_direct_indeterminacy/PREREGISTRATION_DRAFT.md:69)). Thus “Cannot tell” on every T/F item gives `I_dir=0`, satisfying the T/F part of the gate. | The gate does not measure direct-probe calibration. | On T/F require a high “Yes” rate and separately cap “Cannot tell” and unparsed responses. On IO require the intended response rate and cap the others. Predefine whether “Cannot tell” is invalid, epistemic abstention, or a substantive answer. |
| **D1 is selected on essentially the same outcome it tests.** The direct gate demands low `I_dir` on T/F and high `I_dir` on IO, while D1 then tests whether `I_dir` separates IO from T/F ([lines 77 and 89–90](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/22_direct_indeterminacy/PREREGISTRATION_DRAFT.md:77)). The general rule also says hypotheses are judged only among models passing both gates. | Conditioning on the desired separation makes D1 circular and biases its AUROC upward. | Do not apply the direct gate before testing D1. Use the direct-calibration contrast itself as D1. If a gate is needed for D2–D5, estimate it on a disjoint calibration item set and evaluate those hypotheses on untouched items. |
| **The probe does not isolate semantic determinacy from epistemic or computational answerability.** “Can the truth … be determined with current knowledge?” explicitly cues epistemic availability. A finite digit of \(\pi\) is mathematically determinate and computable in principle, but may not have been computed; an historical claim may be factual but poorly documented; a future claim depends on a disputed metaphysics of future contingents. | D2’s intended discovery—that the probe has an epistemic rather than semantic reading—is partly written into the prompt. IO and IU have no unambiguous gold response to this wording. | Split the construct into separate, independently tested prompts: “Does this proposition presently have a truth value?”, “Has its truth been publicly established as of [date]?”, and “Is it computable/answerable in principle?” Do not combine their responses into one primary endpoint. |
| **The calibration constructions do not adequately match the target constructions.** The paper warns that transfer fails across small wording differences and licenses readings only for covered constructions ([main.tex:210–218](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/paper_jolli/main.tex:210)). “1718 was before 1818” does not calibrate “first snowfall in Oslo in 2031”; small-number comparisons do not calibrate population forecasts; ordinary sums do not calibrate future lottery outcomes or enormous mathematical objects. | A model may pass calibration on surface-simple T/F statements and still fail because of future tense, named entities, exponents, or modality rather than indeterminacy. | Construct matched T/F/IO/IU quartets using the same syntax, domain, named-entity load, numerical scale, and negation. Add known-past snowfall/population/lottery controls, feasible and infeasible mathematical computations in the same notation, and held-out template-family evaluation. Gate on those matched controls. |

### Major issues

1. **Several item-status claims are not established by construction.** The assertion that statuses are “fixed by construction” ([lines 52–57](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/22_direct_indeterminacy/PREREGISTRATION_DRAFT.md:52)) is too strong.

   - “Napoleon was born before the tallest person alive in 1810” may be answerable from historical records, and “tallest” is vague when measurements, ties, and the time within 1810 are unspecified.
   - Being beyond the largest *published* computation does not prove that nobody knows the answer or that it is absent from private computations.
   - A finite digit or finite prime is computable in principle, so “cannot be determined” is arguably false even when the answer has not yet been calculated.
   - A future population threshold may already be strongly entailed or ruled out by demographic constraints, depending on what “determined with current knowledge” means.
   - Calling future contingents genuine truth-value gaps assumes the philosophical conclusion being tested.

   **Fix:** rename the operational groups to something defensible, such as “future unresolved as of cutoff” and “no public answer located by cutoff.” Archive full search sources and dates; use two independent adjudicators; mark disputed items; exclude vague historical superlatives. Treat semantic openness as a theoretical condition, not a gold label.

2. **Surface cues can make D1 and D4 pass trivially.** IO contains future tense and years such as 2031/2040; mathematical IU contains enormous exponents; T/F uses ordinary arithmetic and dates. Matching only length and digit count ([lines 109–112](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/22_direct_indeterminacy/PREREGISTRATION_DRAFT.md:109)) does not address these cues.

   **Fix:** use lexical/template matching and counterfactual controls. Report frozen bag-of-words, character n-gram, year-only, exponent-only, and claim-length classifiers. Use group-held-out template evaluation. The direct probe should count as successful only if it exceeds these superficial baselines.

3. **D3 has an invalid confirmatory logic.** “Correlation with I2 is not larger than correlation with I1” ([lines 93–95](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/22_direct_indeterminacy/PREREGISTRATION_DRAFT.md:93)) cannot be established by failing to reject a positive difference. It is also inconsistent with “the CI excludes the null.”

   **Fix:** either preregister a directional superiority hypothesis  
   \[
   \rho(I_{\rm dir},I_1)-\rho(I_{\rm dir},I_2)>0,
   \]
   or an equivalence/non-inferiority margin \(\delta\), justified before data collection, and require the appropriate one-sided bound. A random mock need not “fail” the present formulation.

4. **D2 correlations may be driven entirely by IO-versus-IU membership.** “On IO and on IU” is ambiguous: it could mean separately, jointly, or both. Pooling the groups can create large correlations even when there is no within-group relationship.

   **Fix:** make within-status correlations primary. If a pooled result is reported, use partial/rank regression controlling for status, construction, prompt order, and negation form.

5. **Direct-probe order is confounded with item.** Assigning one of two orders once per item does not distinguish an order effect from an item effect. “Cannot tell” also remains last in both orders.

   **Fix:** cross order within item across repeated samples, or use all six permutations in a balanced Latin-square allocation. Predefine an order-adjusted estimator.

6. **Determinability should be invariant under negation, but this is not tested.** If \(P\) is currently determinable, so should \(\neg P\) be.

   **Fix:** administer \(D(P)\) and \(D(\neg P)\) in separate conversations and preregister a symmetry/calibration measure. Large disagreement should invalidate interpretation of the direct probe.

7. **DO is underdefined and does not measure “confidence.”** Majority can tie, especially with even \(m\), and a plurality just above one third is not confidence. Conditioning on majority “No” also introduces selection noise.

   **Fix:** define tie handling and thresholds, e.g. \(p_D(\mathrm{No})\ge0.8\), \(p_P(T/F)\ge0.8\), and \(p_{\neg P}(F/T)\ge0.8\). Report denominator and missingness. A continuous alternative—mean product of these probabilities—is preferable.

8. **The overall “half the passing models” rule can pass with one eligible model.** The denominator is data-dependent; models are selected by gates and are not a random sample from a model population.

   **Fix:** require a minimum prespecified number of eligible models, report every model as a fixed replication, and avoid a population-level claim based on a majority vote. If an aggregate is essential, preregister a model-stratified meta-analytic estimand.

9. **Local and API measurements are not comparable.** Local models use first-token probabilities, while API models use repeated sampled answers ([lines 102–106](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/22_direct_indeterminacy/PREREGISTRATION_DRAFT.md:102)).

   **Fix:** either use identical constrained decoding/sampling for all models or analyze the two acquisition modes as separate studies. Verify option tokenization and leading-space variants.

### Statistical and control issues

- Forty items per cell is not justified for tied, noisy Spearman correlations based on only 5–10 samples per probe. Run a simulation-based power analysis for the smallest meaningful AUROC and dependent-correlation difference, then set item counts accordingly.
- Bootstrap the underlying stimulus/template or matched quartet, not individual paraphrases or paired inversions as though independent.
- Use paired bootstrap resamples for differences between dependent correlations.
- Predefine nulls separately: AUROC chance is 0.5, while 0.80 is a performance threshold; correlation-difference null is zero; DO’s random floor depends on \(m\), ties, and selection.
- Correct or hierarchically gate the many model × construction × form × hypothesis tests.
- Report all unparsed responses in denominators. Do not replace fully unparsed probes with uniform distributions as the experiment-12 runner does ([run.py:99–105](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/12_calibration_extension/run.py:99)).
- Add a matched “obscure but known” control, a future statement whose answer is already fixed by definition, prompt paraphrases, and simple cue-only baselines.
- The external datasets should receive bilateral probes too if they are intended to validate correspondence between direct and bilateral measures.

---

## Experiment 23: RNEL on real data

### Fatal issues

| Problem | Why it is fatal | Exact fix |
|---|---|---|
| **R3 is structurally favored by its labels and loss.** Chaos items with top vote below 60% are assigned directly to U ([draft line 30](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/PREREGISTRATION_DRAFT.md:30); [features.py:58–63](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/features.py:58)). RNEL is then trained to put evidence into `I_amb`. Chaos has no glut-positive examples, while the loss penalizes evidence on every non-target type ([neutro_edl.py:87–96](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/nel/neutro_edl.py:87)). Therefore `I_amb` is supervised by a threshold closely related to entropy, while `I_glut` is systematically suppressed. | `corr(I_amb, entropy) > corr(I_glut, entropy)` is almost built into the target construction. It does not demonstrate that the head discovered ambiguity rather than conflict. | Use data with independently annotated ambiguity and contradictory evidence, including positive examples of both in the same domain. Alternatively make the confirmatory target continuous vote-distribution prediction and compare RNEL against soft-label cross-entropy or KL baselines. Do not use a threshold of the evaluation target to create training labels. |
| **The softmax comparator in R3 is deliberately mismatched.** RNEL has a directly supervised U channel, but it is compared with the *entropy* of a flat classifier. A softmax confidently predicting the U class has low entropy even though it correctly detects ambiguity. | RNEL can win trivially because the comparator is the wrong output. | Compare `I_amb` with \(p_{\text{softmax}}(U)\), flat-EDL U mass, and a direct entropy-regression baseline. If predicting human entropy is the goal, train every method against the same continuous or vote-distribution target. |
| **R5 cannot meaningfully fail.** The reduction is an algebraic property already implemented and unit-tested ([test_neutro_edl.py:13–21](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/tests/test_neutro_edl.py:13)). Repeating it on real feature vectors adds no empirical evidence. Moreover, the test covers components, not equivalence to every loss commonly called “Sensoy EDL.” | It is a unit test presented as a hypothesis. | Move R5 to implementation validation. Specify whether equality concerns components, logits, loss, gradients, or trained predictions. If comparing losses, implement the exact same flat loss and copy the crisp rows of the same weights; otherwise do not claim reduction to the external baseline. |
| **The “contradiction” labels do not support R2’s construct on both datasets.** AVeriTeC merges conflicting evidence with cherry-picking, which may be supported-but-misleading rather than a glut. Climate-FEVER’s DISPUTED label is mechanically defined by having both supporting and refuting evidence ([README lines 20–24](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/data/README.md:20)). | Success cannot establish general separation of contradictory evidence. On one dataset the class is impure; on the other the label is the rule RNEL is supposed to learn. | Make Climate-FEVER a rule-recovery benchmark, not independent construct validation. Split AVeriTeC conflict and cherry-picking through existing metadata or a blinded annotation audit; otherwise narrow the claim to the dataset’s combined label. R2 should require a clean held-out contradiction dataset not defined by the same evidence-label rule. |
| **R4 measures dataset identity, not epistemic vacuity.** AVeriTeC consists of Q/A evidence with about 2.6 pieces per claim; Climate-FEVER has five Wikipedia sentences. Topics, formatting, evidence count, and label prevalence all differ. `I_vac = W/(N+W)` is merely inverse total evidence magnitude ([neutro_edl.py:72–84](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/nel/neutro_edl.py:72)), and the current loss does not regularize evidence on the correct class, so scale depends on optimization and training duration. | AUROC above 0.60 can be obtained by recognizing formatting or evidence count; failure or success says little about genuine OOD uncertainty. | Use a matched OOD design with identical serialization, evidence counts, length distributions, and label composition. Freeze training duration and scale controls. Compare against evidence count, input length, encoder norm, maximum softmax probability, energy, Mahalanobis distance, and flat-EDL vacuity. Prefer near-OOD examples within one dataset/source. |

### Major issues

1. **There is actual split leakage.** A read-only audit of the downloaded files found:

   - Eight normalized AVeriTeC claim strings occur in both train and dev; some have inconsistent train labels.
   - Climate-FEVER has 45 duplicated normalized claim groups, including 15 groups with different labels.
   - A row-level random 70/15/15 split can therefore put near-identical claims into train, validation, and test.

   **Fix:** publish a deterministic normalization and duplicate-group identifier before training. Split by group, not row. For AVeriTeC, exclude all cross-split overlap groups from the confirmatory dev analysis or reconstruct a leakage-free grouped split without examining model results. Cluster-bootstrap duplicate groups.

2. **The preregistration’s provenance statement is inaccurate.** It says no features have been computed ([lines 3–6](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/PREREGISTRATION_DRAFT.md:3)), but `features.log` records that ChaosNLI encoding reached at least 800/1,599 items ([features.log:12–13](C:/Users/HP/Documents/NEL_Neutrosophic_Evidential_Logic/experiments/23_rnel_real/features.log:12)). No completed `.npz` was present, but an extraction attempt occurred.

   **Fix:** state exactly what was run, when it stopped, what artifacts were produced, and whether anyone inspected representations or performance. Preserve the log. Preregister before rerunning.

3. **The encoder-contamination statement is too narrow.** Saying the encoder did not train on dataset labels does not eliminate contamination. ChaosNLI uses MNLI development texts, while the encoder was specifically trained on MNLI. Web/Wikipedia-based corpora may also overlap pretraining.

   **Fix:** call this task/source contamination, not merely a possibility. Add an encoder not trained on MNLI for ChaosNLI, report results by original MNLI genre, and avoid claims of uncontaminated generalization. Where feasible, use a frozen generic encoder and a claim-only control.

4. **Several label mappings are conceptually invalid or at least non-unique.**

   - NLI “neutral” means the premise neither entails nor contradicts the hypothesis; it does not necessarily mean “no evidence.”
   - “Not enough evidence” can reflect incomplete retrieval, dataset policy, or annotation failure rather than a Belnap gap.
   - Mapping NEE to U only as sensitivity does not resolve the lack of a validated semantic mapping.

   **Fix:** state that these are operational dataset labels, not ground-truth neutrosophic states. Make both NEE→gap and NEE→ambiguity co-primary or select one using an independent annotation study—not performance.

5. **Prediction and metric mappings are underdefined.** The draft does not say whether RNEL predicts by argmax of raw evidence, \(\alpha/S\), or derived components. `T` and `F` components double-count glut evidence and do not form a conventional class-probability vector.

   **Fix:** define primary prediction as argmax over the typed posterior \(\alpha/S\), with an explicit dataset-specific allowed-label mask. Define how predictions into a class absent from a dataset affect macro-F1. Do not use the derived T/F components as mutually exclusive label probabilities.

6. **“Softmax, 4 or 5 classes” is not a fixed method.** The class set, handling of absent C/U types, MLP dimensions, pooling operations, early stopping, epochs, batch size, optimization grid, and threshold grids remain unspecified. “Predefined grid” without listing the grid is not a preregistration.

   **Fix:** enumerate every architecture and hyperparameter value, selection rule, tie-breaker, epoch budget, early-stopping rule, and threshold grid before training. State that thresholds are refit separately on each training split’s validation portion and never on target-domain/test data.

7. **The “same encoder and MLP” claim is false for the rule and subjective-logic baselines.** They bypass the MLP. Subjective cumulative fusion also assumes evidence independence, despite sentences for one claim being strongly dependent.

   **Fix:** separate nonlearned baselines from capacity-matched learned baselines. Add mean/max pooled NLI logistic regression and a DeepSets softmax baseline. Include averaging or dependence-robust fusion alongside cumulative fusion.

8. **Class weighting compromises calibration.** Inverse-frequency weights change the learned posterior away from the empirical class prevalence, yet ECE is later reported. It also cannot apply to nontrained rules “for every method.”

   **Fix:** apply weights only to trainable classifiers and either omit them for calibration analysis or correct predictions back to the deployment prior. Report unweighted and balanced training as a sensitivity analysis.

9. **ECE of T is ill-defined.** It is unclear whether the event is “label is T,” whether the input is `T`, \(\alpha_T/S\), or a normalized component, and what bins are used.

   **Fix:** define one-vs-rest calibration using the typed posterior \(p(y=T)=\alpha_T/S\), fixed bin edges, minimum bin handling, and Brier/log loss. Add multiclass calibration; do not treat the Belnap T component as a categorical probability.

### Statistical problems

- AVeriTeC dev contains only 38 C examples. A 15% Climate test split contains about 23 C examples. Per-class F1 differences will be unstable and underpowered.
- Five training seeds do not solve the small independent test-class count. Conduct a simulation-based power analysis using plausible confusion matrices. If power is inadequate, use grouped repeated cross-validation or enlarge the clean contradiction evaluation set.
- “Bootstrap over claims, median over seeds” conditions on the chosen trained models and ignores seed and split variability. It is especially inadequate for Climate’s three random splits.
- Use paired resampling because all methods see the same claims. For Climate, use a hierarchical procedure over duplicate-groups, split replicates, and training seeds, or create fixed grouped out-of-fold predictions and bootstrap those groups.
- Define non-inferiority correctly: for \(\Delta=F1_{\rm RNEL}-F1_{\rm EDL}\), require the one-sided lower confidence bound to exceed \(-0.02\). Justify the margin independently.
- R2 contains four superiority requirements—two comparators on two datasets. State whether it is one conjunction claim and define paired difference intervals. Apply hierarchical testing or multiplicity control to the remaining R1–R4 families.
- R4 has two domain directions; both-vs-either must be specified.
- Report precision-recall AUC for rare C classes in addition to F1, with fixed class support and threshold rules.

### Missing baselines and controls

At minimum add:

- Capacity-matched five-type softmax and flat EDL heads.
- A simple pooled-NLI logistic/MLP classifier.
- Claim-only and evidence-only controls.
- Gold evidence-label rule versus predicted-NLI rule on Climate-FEVER.
- \(p(U)\), flat-EDL U mass, vote-distribution prediction, and direct entropy regression for ChaosNLI.
- Maximum softmax probability, energy, Mahalanobis distance, encoder norm, evidence count, and length for OOD.
- RNEL ablations for glut, ambiguity, gap, \(W\), regularization, and class weighting.
- A second encoder not trained on MNLI for the ChaosNLI claim.

## Minimum changes before registration

For experiment 22, the indispensable repairs are: add accuracy to the bilateral gate; redefine the direct response categories; remove D1’s circular gate; use genuinely matched calibration/target stimuli; operationalize rather than assert IO/IU status; and replace D3 with a valid superiority or equivalence test.

For experiment 23, the indispensable repairs are: redesign R3; move R5 to unit testing; remove duplicate leakage; narrow or independently validate the contradiction labels; redesign R4 as matched OOD evaluation; fully specify training/threshold selection; and disclose the pre-registration feature-extraction attempt.

Review framework informed by the academic methods checklist in 【paper-reviewer†C:/Users/HP/.codex/plugins/cache/claude-cowork/anthropic-skills/1.0.0/skills/paper-reviewer/SKILL.md】.