# RNEL neural head on real fact-checking data

This repository contains the code, the preregistration and the results of the empirical check of the
**Refined Neutrosophic Evidential Logic (RNEL)** evidence tuple (Definition 8.1 in Smarandache and
Leyva-Vázquez, *The Neutrosophic (T, I, N, F), Multi-Uncertainty, n-Valued Refined, and Refined Evidential
Logics as Extensions of Subjective Logic*, manuscript, 2026, Section 8.6).

A neural network outputs non-negative evidence for five kinds of report: true (T), false (F), contradictory
(C), undetermined (U) and neither (N). The RNEL tuple is T = e_T/S, F = e_F/S, C = e_C/S, U = e_U/S,
N = e_N/S and G = W/S, with S = e_T + e_F + e_C + e_U + e_N + W. With only T and F, the head reduces
exactly to the binomial Subjective Logic opinion and to evidential deep learning (Sensoy et al., 2018).

## Main result (grouped 5-fold cross-validation, nested tuning, 5 seeds)

| Method | AVeriTeC macro-F1 | AVeriTeC F1 on C | Climate-FEVER macro-F1 | Climate-FEVER F1 on C |
|---|---|---|---|---|
| RNEL head | 0.483 | 0.221 | 0.465 | 0.234 |
| Flat EDL | 0.488 | 0.223 | 0.472 | 0.242 |
| Flat softmax | 0.469 | 0.206 | 0.467 | 0.242 |
| Subjective Logic, cumulative fusion | 0.352 | 0.126 | 0.282 | 0.153 |
| Rule (support and refute gives C) | 0.334 | 0.098 | 0.288 | 0.123 |

- **Learned evidence beats non-learned combination.** On the contradiction class, the learned models
  do clearly better than Subjective Logic fusion and than the rule. RNEL − SL cumulative, F1 on C:
  +0.094 [0.042, 0.146] on AVeriTeC and +0.080 [0.011, 0.148] on Climate-FEVER.
- **The refined head matches flat EDL in accuracy (within 0.01).** It did not show better detection
  of contradiction than a flat classifier; the preregistered superiority hypothesis was not supported.
- **Its value is an interpretable decomposition.** Under typed supervision its components are
  deterministic summaries of the typed evidence.

Full results, deviations and the adversarial reviews: `experiment/RESULTS.md` and `experiment/review/`.

## Contents
- `nel/neutro_edl.py`: the RNEL head, its components and its loss. `tests/` covers the reduction to
  evidential deep learning.
- `experiment/PREREGISTRATION.md`: committed before training. `PREREGISTRATION_DRAFT.md` is v1, kept
  for the record.
- `experiment/features.py`: encodes every (evidence, claim) pair with the frozen
  `cross-encoder/nli-deberta-v3-base`.
- `experiment/run.py`: trains every method. `--nested` gives the corrected analysis, with
  hyperparameters tuned inside each fold.
- `experiment/analyze.py`: computes the hypotheses and the descriptive measures (`--nested`).
- `experiment/results/`: out-of-fold predictions, summaries and a SHA-256 manifest.
- `experiment/paper_section/`: the figures and numbers of Section 8.6.

## Reproduce
```bash
pip install -r requirements.txt
# download the data into experiment/data/raw/ (see experiment/data/README.md; licences CC BY-NC 4.0)
cd experiment
python features.py                         # CPU, about 1-2 h
python run.py --nested climate averitec chaos
python analyze.py --nested
python paper_section/make_figs.py
```
The raw datasets and encoder features are not redistributed, because of their non-commercial licences.

## Data
- AVeriTeC (Schlichtkrull, Guo and Vlachos, NeurIPS 2023 Datasets and Benchmarks).
- CLIMATE-FEVER (Diggelmann et al., arXiv:2012.00614).
- ChaosNLI (Nie, Zhou and Bansal, EMNLP 2020).

## Licence
The code is under the MIT licence. The datasets keep their own licences.
