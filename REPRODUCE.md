# Reproducing the article

Each table, figure, and number in the article is listed below with the script
that produces it and the output file that holds it. All outputs are committed,
so every number can be checked without running anything.

Two levels of reproduction:

- **Re-scoring** (section A) recomputes every score, interval, and table from the
  saved predictions. CPU only, minutes.
- **Re-extraction** (section B) regenerates the predictions themselves. The
  extractor runs on CPU or GPU; third-party systems need their own environments
  (section C).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import stanza; stanza.download('pt')"
python scripts/fetch_external_benchmark.py                        # PT-OIE-Benchmark, commit f69b6bb -> .external/
python scripts/fetch_oiec_pt.py                                   # OIEC-PT (Zenodo 10.5281/zenodo.17438351) -> data/external/
python scripts/fetch_biomed_pt.py --pmids data/biomed_pt_pmids.txt # 241 PubMed sentences -> data/biomed_pt_sentences.txt
python -m pytest -q
```

`fetch_external_benchmark.py` records URL, commit, license, and SHA-256 of the
benchmark files in `outputs/external_benchmark/provenance.json`.
`fetch_oiec_pt.py` checks the MD5 sums published by Zenodo, and
`fetch_biomed_pt.py --pmids` rebuilds the PubMed file byte for byte (SHA-256
`7d29338f…26269d6`).

The article's runs used Python 3.12.8, PyTorch 2.5.1 (CUDA 12.4), Transformers
4.48.2, and Stanza 1.10.1 on Ubuntu 24.04 (`outputs/benchmark/manifest.json`,
`outputs/external_benchmark/logs/run_log.json`).

## Scoring rules

- The PT-OIE-Benchmark evaluator is `Evaluation.py` from the benchmark,
  unmodified. `src/benchmark/external/carb.py` decomposes it per sentence for the
  bootstrap and checks the decomposition against the original on every
  system–corpus pair (`outputs/external_benchmark/evaluations/metrics.json`,
  35 pairs, maximum difference 5.6e−16).
- Triples with an empty slot are kept, as the evaluator reads them: an empty slot
  becomes NaN and matches only an empty reference slot.
- Bootstrap: sentence-level, 10,000 replicates, seed 42, one resampling matrix per
  corpus shared by all systems, so differences are paired.
- Extractor seed: 13.

## A. Re-scoring from saved predictions

### Tables

| Table | Command | Output |
|---|---|---|
| 1. UD patterns | `python scripts/pattern_shares.py` | `outputs/benchmark/metrics/pattern_shares.json` |
| 2. Validation checks | `python scripts/audit_rule_subsumption.py` (runs the generator; see B) | `outputs/rule_audit/subsumption.json` |
| 3. Evaluation corpora, redundancy column | `python scripts/gold_dedup_ranking.py` | `outputs/external_benchmark/gold_dedup_ranking.json` |
| 4. Ablation (`bia_legacy`) | `python scripts/ablation_cis.py` | `outputs/ablation_cis.json` |
| 4. Control row | `python scripts/attention_controls.py` (see B) | `outputs/attention_controls/attention_controls.json` |
| 5. Eight systems on BIA | `python scripts/rescore_app_bia_predictions.py` then `python scripts/bia_cis_and_trivial_floor.py outputs/benchmark_app_rescore/cis_and_trivial_floor.json` | `outputs/benchmark_app_rescore/{results,cis_and_trivial_floor}.json` |
| 6. External corpora | `python scripts/evaluate_external_benchmark.py`, `python scripts/bootstrap_external_benchmark.py`, `python scripts/report_external_benchmark.py` | `outputs/external_benchmark/tables/comparison_f1.csv`; BIA column and floors from Table 5 |
| 7. Evaluator × corpus grid | `python scripts/evaluator_corpus_grid.py` | `outputs/external_benchmark/evaluator_corpus_grid.json` |
| 8. Negation preservation | `python scripts/polarity_fidelity.py` (three groups, see B), then `python scripts/polarity_report.py --kind neg` | `outputs/polarity/aggregate_neg.csv` |

### Numbers in the text

| Section | Number | Source |
|---|---|---|
| Attention reranking | output identical for τ ∈ {0.00, …, 0.30} | `outputs/benchmark/metrics/attention_threshold_sweep.json` (`scripts/attention_threshold_sweep.py`) |
| Attention reranking | max conf_att = 0.174 | field `confidence` in `outputs/benchmark/normalized/pt_oie.jsonl` |
| Corpora | BIA 262 sentences, 427 triples, median 1, maximum 9 | `data/bia_gold_sentences.jsonl`; `outputs/benchmark/corpus_validation.json` (`scripts/validate_corpus.py`) |
| Evaluators | greedy vs. optimal assignment (+1.45, 0.97, 0.17, 0.00) | `python scripts/greedy_vs_optimal.py` → `outputs/benchmark/metrics/greedy_vs_optimal_bia_legacy.csv` |
| Compared systems | DptOIE 913 of 2,381 misaligned; 3 of 2,402 | `outputs/benchmark_app_rescore/results.json` (`misaligned_triples`, `n_triples`) |
| Compared systems | published F1 reproduced within 0.1 except ArgOE | `outputs/external_benchmark/tables/comparison_full.csv`, `reproduction_divergences.csv` |
| Statistics and environment | 0.12–0.17 s per sentence; no exception on 506 sentences | `seconds` and `error` in `outputs/external_benchmark/predictions/native/pt_oie/*.jsonl` |
| Ablation results | strict and tolerant precision and Δ of the validation filter | `python scripts/ablation_cis.py --protocol strict --output outputs/ablation_cis_strict.json` (same with `tolerant`) |
| BIA comparison | margins and paired intervals | `paired_vs_pt_oie` in `outputs/benchmark_app_rescore/cis_and_trivial_floor.json` |
| BIA comparison | Multi²OIE: 110 of 672 triples with an empty slot | `n_incomplete` in `outputs/benchmark_app_rescore/results.json` |
| External corpora | Gamalho Δ and ranges | `outputs/external_benchmark/bootstrap/paired_f1_differences.csv` |
| External corpora | RQ3 − RQ1 per corpus | `rq3_minus_rq1_benchmark_evaluator` in `outputs/external_benchmark/evaluator_corpus_grid.json` |
| Provenance | redundancy and ranking without reference variants | `outputs/external_benchmark/gold_dedup_ranking.json` |
| Provenance | relation length, slot F1 on WIKI200 | `python scripts/slot_profile.py` → `outputs/external_benchmark/slot_profile.json` |
| How the best neural systems match | encompassing and exact shares, length ratio, strict F1 | `python scripts/match_profile.py` → `outputs/external_benchmark/match_profile.json` |
| Attention stage | 293 groups, 60 competing, ties, controls | `outputs/attention_controls/attention_controls.json` |
| Validation layer | 8 E1, 24 E4, 6 S4, 1 S1, 17 S3; filter removes 31 of 538 | `python scripts/validation_bia_counts.py` → `outputs/rule_audit/bia_reference_violations.json` |
| Copula | `cop` 78 triples, 4 tolerant matches | `outputs/benchmark/metrics/metrics_by_pattern.csv` (`scripts/evaluate_benchmark.py`) |
| Generative ceiling | 41 null-subject triples; +0.94 on BIA | `python scripts/op1_bia.py` → `outputs/dev_oiec/op1_bia.json` |
| Generative ceiling | external Δ of the extension (−2.1, +1.9, −0.5) | rows `pt_oie` vs `pt_oie_op1` in `outputs/external_benchmark/bootstrap/paired_f1_differences.csv` |
| Generative ceiling | sentences without extraction; recall ceiling 38.5/34.3/50.7 | `python scripts/generative_ceiling.py` → `outputs/external_benchmark/generative_ceiling.json` |
| Generative ceiling | unreachable predicates and their families | `python scripts/unreachable_families.py` → `outputs/external_benchmark/unreachable_families.json` |
| Polarity | 1,309 sentences, 147 negated predicates; modality 78.8–100% | `outputs/polarity/{polarity_fidelity,bia_oiec,biomed}.json`, `aggregate_mod.csv` |
| Discussion, abstract | trivial floor 34–41 | `floor_external` and `bia` in `outputs/benchmark_app_rescore/cis_and_trivial_floor.json` |

### Figure 2 (open head search)

```bash
python scripts/head_search.py      # outputs/head_search/head_gains_open_search.json (needs BERTimbau)
python scripts/plot_head_search.py --output head_search.pdf
```

## B. Re-extraction

These commands regenerate predictions; compare the new files with the committed
ones.

```bash
# Ablation: 26 labels; outputs/ablation keeps sentence and triples per label
python -m src.run_all_ablations --output-dir runs/ablation --force
python scripts/ablation_cis.py --ablation-dir runs/ablation --output runs/ablation_cis.json

# PT-OIE-EXTRACTOR, pure UD baseline, and the extension on the external corpora
python scripts/run_external_benchmark.py --systems pt_oie ud_baseline pt_oie_op1 --force

# Our systems on BIA (also writes outputs/benchmark/normalized/)
python -m scripts.run_benchmark --config configs/benchmark.yaml --systems pt_oie ud_baseline
python -m scripts.evaluate_benchmark

# Analyses that run the extractor
python scripts/attention_controls.py
python scripts/attention_threshold_sweep.py
python scripts/audit_rule_subsumption.py
python scripts/validation_bia_counts.py
python scripts/op1_bia.py
python scripts/dev_oiec_baseline.py            # OIEC-PT development measurements (outputs/dev_oiec/)
python scripts/generative_ceiling.py
python scripts/unreachable_families.py

# Polarity: inputs on BIA, OIEC-PT, and BIOMED, then the three measurement groups
python scripts/run_on_sentences.py --system pt_oie --dataset BIOMED   # likewise ud_baseline, pt_oie_op1; BIA, OIEC-PT
python scripts/polarity_fidelity.py
python scripts/polarity_fidelity.py --datasets BIA OIEC-PT --output outputs/polarity/bia_oiec.json
python scripts/polarity_fidelity.py --datasets BIOMED --output outputs/polarity/biomed.json
python scripts/polarity_report.py --kind neg && python scripts/polarity_report.py --kind mod
```

The ablation outputs were first produced with Transformers 4.46.3; all runs added
for the article used 4.48.2 and reproduce them triple for triple.

## C. Third-party systems

Outputs of the eight PT-OIE-Benchmark systems on WIKI200, CETEN200, and Gamalho
are the files distributed by the benchmark (`.external/PT-OIE-Benchmark/System_Results`).

On BIA, the predictions of DetIE, ArgOE (LinguaKit), DptOIE, Multi²OIE, and
PragmaticOIE are in `outputs/bia_external_systems/`. They were produced with each
system's own implementation in a separate harness, which is not part of this
repository, and are re-scored here without new inference. Run notes for each
system are in [THIRD_PARTY.md](THIRD_PARTY.md). `scripts/fetch_external_systems.py`
clones the authors' repositories, and `scripts/run_pragmaticoie_bia.sh` runs
PragmaticOIE in R 4.1.3.

Gemma 4 runs through a local Ollama server (`configs/systems/ollama_gemma4.yaml`,
prompt in `configs/gemma4_prompt_pt.txt`). The model digest is checked before
every run (`scripts/inspect_ollama.py`, `outputs/benchmark/models/`).
