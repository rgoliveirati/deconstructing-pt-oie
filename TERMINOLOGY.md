# Terminology

This file fixes the names used in the paper and in this repository. Code,
documentation, reports, and the manuscript use the same terms. When a term
changes, it changes here first.

## Evaluators

| Name | What it is | Do not call it |
|---|---|---|
| **PT-OIE-Benchmark evaluator** | `Evaluation.py` from the PT-OIE-Benchmark (Silva et al., LREC 2026), used unmodified at commit `f69b6bb`. Token-level F1 per slot (ARG0, V, ARG1), averaged over the three slots; greedy one-to-one assignment for precision; best match per gold tuple for recall; micro-averaged. Triples with an empty slot are kept, as the script reads them (the empty slot becomes NaN). Implemented through `src/benchmark/external/carb.py`, which reproduces it numerically. | "official", "the official evaluator", "CaRB" |
| **`bia_legacy`** | The project's legacy matcher, now fully specified: normalization, bidirectional word-subset and one-token prefix conditions, greedy assignment in gold order. Reported with both greedy and optimal assignment. | "official", "Official scorer" |
| **CaRB-style** (`carb_style`) | Internal protocol: weighted slot F1 (0.35 / 0.30 / 0.35), binary match at 0.60. It diverges from the PT-OIE-Benchmark evaluator on identical predictions. | "CaRB", an "external" protocol |
| **strict** | Exact match of the three slots after light normalization. | — |
| **tolerant** | Minimum per-slot token F1 of 0.70. | — |
| **trivial verb-split baseline** (trivial floor) | For each verb: everything to its left is ARG0, the verb is V, everything to its right is ARG1; two variants (VERB; VERB+AUX). Reported as the floor of every evaluator, including the PT-OIE-Benchmark evaluator (34.0–41.4 F1). | — |

Every F1 is reported with its protocol and its corpus. No unqualified "best
system".

## The extractor

| Name | Meaning | Do not say |
|---|---|---|
| **UD candidate generator** | UD patterns that produce candidate triples and their span variants. | — |
| **attention reranking** | Among competing candidates of the same (sentence, verb, pattern) group, keep the one with the highest attention confidence `conf_att`; ties keep the first candidate generated. In every competing group observed on BIA and the three PT-OIE-Benchmark corpora the scores tie, so the stage acts as a one-per-group pruning; random, inverted, or constant scores give identical output (`outputs/attention_controls/`). | "attention filter", "attention-based filtering", "selection by threshold", "attention selects the best candidate" |
| **attention-scoped candidates** | Candidates from the patterns in `attn_decision_patterns` (`verb+obj`, `verb+obl`, `cop`, `aux:pass`): 334 of 538 triples (62.1%) in `rq3_attn_on_thr0`. | — |
| **attention threshold τ** (`attn_threshold`) | Defined in the configuration but not applied on the active extraction path. | a working hyperparameter |
| **selected heads** | Top-10 heads per role, ranked on UD Bosque inside fixed layer windows (subject 2–5, object 7–12). The layer separation is imposed by the windows; without them the best subject head is L6H11 (`outputs/head_search/`). | "specialized heads"; "head segregation" as a finding |
| **open head search** | Ranking of all 144 heads without layer windows. | — |
| **validation layer** | Rule checks applied to candidates in mode `off`, `annotate`, or `filter`. | — |
| **E rules / S rules** | E1–E9 are **structural** rules; S1–S5 are **semantic and discourse** rules. | "S structural", "E syntactic-semantic" |
| **eleven executable checks** | E1, E2, E3, E4, E4_1, E5, E8, S1, S2, S3, S4. Under default settings, six fire (E1, E4, E5, S1, S3, S4), three are inert by configuration (E4_1, E8, S2), and two never fire on five corpora (E2, E3). | "sixteen rules" |
| **reserved slots** | E4_2, E6, E7, E9, S5: always pass by construction; E9 repeats the test of E4_2. | rules |

## Experiments

| Name | Meaning |
|---|---|
| **ablation labels** | The 26 labels in `src/run_all_ablations.py` (`rq1_baseline_ud_puro` … `rq8_best_config_no_E4`). They are shared with the notebook and must not be renamed. There is no RQ7: it was dropped during writing and the labels were kept. |
| **distinct configurations** | 25: `rq3_attn_on_thr0` and `rq3_attn_on_thr15` are the same configuration, because τ is not applied. Five more labels (`rq6b_no_E4_2`, `rq6b_no_E6`, `rq6b_no_E7`, `rq6b_no_E9`, `rq6_no_S5`) are identical to `rq4_theory_all_on` by construction. |

## Corpora

| Name | Role |
|---|---|
| **BIA** (Queiroz et al., 2023) | Evaluation corpus: 262 sentences, 427 gold triples. Shares annotation conventions with the extractor. |
| **PT-OIE-Benchmark corpora** | WIKI200, CETEN200, Gamalho, from the PT-OIE-Benchmark repository. External evaluation. |
| **OIEC-PT** | Development corpus. Used for the rule audit only; not an independent test set. |
| **UD Bosque** | Used only to rank attention heads. Never an evaluation set. |

## Compared systems

| Name | Meaning | Do not say |
|---|---|---|
| **authors' implementation / authors' checkpoint** | The code or model distributed by a system's authors, run without modification. | "official implementation", "official checkpoint" |
| **reproduced** | A score computed locally with the PT-OIE-Benchmark evaluator from per-sentence outputs. | "reported" |
| **Gemma 4** | 8B, Q4_K_M, local, zero-shot, fixed prompt, temperature 0. | — |
