# Deconstructing PT-OIE-EXTRACTOR

Code, data, and outputs for the article *Deconstructing UD and Attention for Open
Information Extraction in Brazilian Portuguese: Component Contributions and
Measurement Validity* (Oliveira, Claro, and Araújo; submitted to the Journal of
Intelligent Information Systems).

PT-OIE-EXTRACTOR is a training-free Open Information Extraction system for
Brazilian Portuguese. It generates candidate triples from Universal Dependencies
patterns, keeps one candidate per (sentence, verb, pattern) group among competing
candidates using BERTimbau attention, and can apply eleven validation checks. The
article takes the system apart and uses it to test how far Portuguese OpenIE
scores can be trusted. Every number it reports comes from a script in
`scripts/` and an output in `outputs/`; [REPRODUCE.md](REPRODUCE.md) maps each
table, figure, and number in the text to its command and file.

Names of evaluators, components, corpora, and systems follow
[TERMINOLOGY.md](TERMINOLOGY.md), which the article uses as well.

## Repository structure

```text
├── src/
│   ├── extractor.py          # the extractor: legacy core + validation layer
│   ├── run_experiment.py     # one configuration
│   ├── run_all_ablations.py  # the 26 ablation labels
│   └── benchmark/            # corpora, protocols, system adapters, PT-OIE-Benchmark evaluator wrapper
├── scripts/                  # every analysis behind the article (see REPRODUCE.md)
├── configs/                  # benchmark, external evaluation, and system configurations
├── data/                     # BIA, UD Bosque, PubMed IDs (see data/README.md)
├── outputs/                  # predictions, metrics, intervals, and tables used in the article
└── tests/
```

Docstrings and comments in the code are in Portuguese; documentation is in
English.

## Quick start

Python 3.12 on Linux. A CUDA GPU speeds up attention but is not required.

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # with conda: export PYTHONNOUSERSITE=1
pip install -r requirements.txt
python -c "import stanza; stanza.download('pt')"
python scripts/fetch_external_benchmark.py   # PT-OIE-Benchmark at commit f69b6bb, into .external/
python -m pytest -q
```

Run the reference configuration on BIA:

```bash
python -m src.run_experiment \
  --gold data/bia_gold_sentences.jsonl --bosque data/pt_bosque-ud-train.conllu \
  --output-dir runs/rq3_attn_on_thr0 --dataset-name bia \
  --heads-mode rank --top-k-heads 10 --attn-threshold 0.0 --theory-mode off
```

The first attention-enabled run downloads `neuralmind/bert-base-portuguese-cased`
from Hugging Face. `--attn-threshold` is accepted but not applied on the active
extraction path (see TERMINOLOGY.md).

Re-score the saved predictions behind the main tables, without new inference:

```bash
python scripts/ablation_cis.py                   # ablation table
python scripts/bia_cis_and_trivial_floor.py outputs/benchmark_app_rescore/cis_and_trivial_floor.json
python scripts/evaluate_external_benchmark.py && python scripts/bootstrap_external_benchmark.py \
  && python scripts/report_external_benchmark.py # external table
```

## Ablation labels

`python -m src.run_all_ablations --list` lists the 26 labels (`rq1_baseline_ud_puro`
… `rq8_best_config_no_E4`). They contain 25 distinct configurations and produce
ten distinct outputs. The labels are shared with the article and must not be
renamed. `run_all_ablations` reuses cached metrics in its output directory; use
`--force` after any change to code, model, corpus, parser, or configuration.

## Licenses

- Code: MIT ([LICENSE](LICENSE)).
- Data and outputs: see [data/README.md](data/README.md). BIA is distributed under
  CC BY-SA 4.0 as a derivative of SQuAD 2.0; UD Bosque keeps its CC BY-SA 4.0
  license. The PT-OIE-Benchmark corpora, OIEC-PT, and the PubMed sentences are
  not redistributed; scripts download them.
- Third-party systems are run from their authors' repositories and are not
  included; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

If you use this code or the analyses, please cite the article (citation will be
updated on publication).
