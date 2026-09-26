# Data for experiment 23 (downloaded 2026-09-26, not redistributed)

The raw files are in `raw/`, which git ignores because of their non-commercial licences.

| File | Source | Licence | SHA-256 |
|---|---|---|---|
| averitec_train.json, averitec_dev.json | huggingface.co/chenxwh/AVeriTeC (`data/`) | CC BY-NC 4.0 | ae5eda7c…, 49979372… |
| climate-fever.jsonl | github.com/tdiggelm/climate-fever-dataset (`dataset/`) | not stated in the README; must be checked before any public release | 8a4b9032… |
| chaos_mnli.jsonl | huggingface.co/datasets/tasksource/chaos-mnli-ambiguity (mirror of the MNLI part of ChaosNLI; the original Dropbox link now returns HTML) | ChaosNLI: CC BY-NC 4.0 | 07d75670… |

## Counts (computed on download)

### AVeriTeC (real fact-checked claims; evidence given as question-answer pairs, 2.6 per claim on average)

| Split | Claims | Refuted | Supported | Not Enough Evidence | Conflicting Evidence/Cherrypicking |
|---|---|---|---|---|---|
| train | 3,068 | 1,742 | 849 | 282 | 195 |
| dev | 500 | 305 | 122 | 35 | 38 |

### Climate-FEVER (5 Wikipedia evidence sentences per claim)

- **Claims:** 1,535 in total: 654 SUPPORTS, 253 REFUTES, 474 NOT_ENOUGH_INFO and 154 DISPUTED.
- **DISPUTED is defined by a rule:** the 154 DISPUTED claims are exactly the claims that have both SUPPORTS and REFUTES evidence.
- **Evidence pairs:** 7,675 in total. Of these, 3,792 have annotators who disagree (at most 5 votes per pair).

### ChaosNLI-MNLI (MNLI items annotated by 100 people each)

- **Items:** 1,599, with a median label entropy of 1.09 (maximum 1.58).
- **Agreement:** on 604 items the top label has less than 60% of the votes; on 57 items it has 90% or more.
