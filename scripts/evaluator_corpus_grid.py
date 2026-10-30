"""Grade avaliador × corpus para as mesmas predições do PT-OIE-EXTRACTOR.

Para cada corpus (BIA, WIKI200, CETEN200, Gamalho) e cada avaliador (avaliador do
PT-OIE-Benchmark e os quatro protocolos internos), pontua as predições já salvas
de ``pt_oie`` (``rq3_attn_on_thr0``) e de ``ud_baseline`` (``rq1_baseline_ud_puro``).
Nenhuma inferência nova. Mede quanto o F1 varia quando só o avaliador muda (linha)
e quando só o corpus muda (coluna), para comparar com a amplitude da ablação.

Regras de pontuação, as mesmas do resto do repositório: todos os avaliadores mantêm
todas as triplas; para o avaliador do PT-OIE-Benchmark, campo vazio vira NaN, como
na leitura de um TSV pelo ``Evaluation.py``.

Uso:
    python scripts/evaluator_corpus_grid.py --output outputs/external_benchmark/evaluator_corpus_grid.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.benchmark.corpus import GoldTriple, Sentence, load_bia  # noqa: E402
from src.benchmark.evaluation import PROTOCOLS, evaluate_protocol  # noqa: E402
from src.benchmark.external.bootstrap import bootstrap_carb, make_indices, paired_delta  # noqa: E402
from src.benchmark.external.carb import load_benchmark_evaluator, score_dataset  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.runtime import write_json  # noqa: E402
from src.benchmark.schemas import read_jsonl  # noqa: E402

BENCH = REPO / ".external/PT-OIE-Benchmark"
PRED_DIR = REPO / "outputs/external_benchmark/predictions/benchmark_format"
SYSTEMS = ("pt_oie", "ud_baseline")
EXTERNAL = ("WIKI200", "CETEN200", "Gamalho")


def bia_inputs(system: str):
    sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    by = defaultdict(list)
    for r in read_jsonl(REPO / "outputs/benchmark/normalized" / f"{system}.jsonl"):
        by[r["sentence_id"]].append({k: r.get(k) or "" for k in ("arg1", "rel", "arg2")})
    gold = pd.DataFrame([{"Sentence": s.sentence, "ARG0": g.arg1, "V": g.rel, "ARG1": g.arg2}
                         for s in sents for g in s.gold])
    return sents, [by.get(s.sentence_id, []) for s in sents], gold


def external_inputs(system: str, name: str):
    # O texto da sentença no gold nem sempre é idêntico ao do arquivo de sentenças;
    # o avaliador do benchmark casa as duas pelo conjunto de palavras normalizadas,
    # e aqui se usa a mesma chave.
    key = lambda s: frozenset(load_benchmark_evaluator(str(BENCH)).normalize_words(s))  # noqa: E731
    data = load_dataset(name, BENCH)
    gold_by = defaultdict(list)
    for _, r in data.gold.iterrows():
        gold_by[key(r["Sentence"])].append(GoldTriple(str(r["ARG0"]), str(r["V"]), str(r["ARG1"])))
    sents = [Sentence(f"{name}_{i:04d}", s, None, i, gold_by.get(key(s), [])) for i, s in enumerate(data.sentences)]
    pred = pd.read_csv(PRED_DIR / system / f"{name}_results.csv", sep="\t", keep_default_na=False)
    pred_by = defaultdict(list)
    for _, r in pred.iterrows():
        pred_by[key(r["Sentence"])].append({"arg1": r["ARG0"], "rel": r["V"], "arg2": r["ARG1"]})
    unknown = set(pred_by) - {key(s) for s in data.sentences}
    if unknown:
        raise SystemExit(f"{system}/{name}: {len(unknown)} sentenças previstas fora do corpus")
    missing_gold = sum(1 for s in sents if not s.gold)
    return sents, [pred_by.get(key(s.sentence), []) for s in sents], data.gold, missing_gold


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/external_benchmark/evaluator_corpus_grid.json")
    args = p.parse_args(argv)

    grid: dict = {}
    boots: dict = {}
    for system in SYSTEMS:
        grid[system] = {}
        for corpus in ("BIA",) + EXTERNAL:
            if corpus == "BIA":
                sents, preds, gold_df = bia_inputs(system)
                note = None
            else:
                sents, preds, gold_df, missing = external_inputs(system, corpus)
                note = f"{missing} sentenças sem gold" if missing else None
            rows = [{"Sentence": s.sentence, "ARG0": t["arg1"], "V": t["rel"], "ARG1": t["arg2"]}
                    for s, ps in zip(sents, preds) for t in ps]
            pred_df = pd.DataFrame(rows, columns=["Sentence", "ARG0", "V", "ARG1"]).replace("", float("nan"))
            sc = score_dataset(gold_df, pred_df, str(BENCH))
            cell = {"benchmark_evaluator": sc["f1_score"] * 100}
            idx = make_indices(len(sc["_per_sentence"]))
            boots[(system, corpus)] = bootstrap_carb(sc["_per_sentence"], idx, sc["n_pred_orphan"])
            for proto in PROTOCOLS:
                cell[proto] = evaluate_protocol(proto, sents, preds)["f1"] * 100
            if note:
                cell["note"] = note
            grid[system][corpus] = cell
            print(system, corpus, {k: round(v, 2) for k, v in cell.items() if isinstance(v, float)}, flush=True)

    evaluators = ["benchmark_evaluator", *PROTOCOLS]
    spans = {}
    for system, by_corpus in grid.items():
        across_evaluators = {c: max(v[e] for e in evaluators) - min(v[e] for e in evaluators)
                             for c, v in by_corpus.items()}
        across_corpora = {e: max(v[e] for v in by_corpus.values()) - min(v[e] for v in by_corpus.values())
                          for e in evaluators}
        spans[system] = {"across_evaluators_by_corpus": across_evaluators,
                         "across_corpora_by_evaluator": across_corpora}
        print(system, "evaluator span by corpus", {k: round(v, 1) for k, v in across_evaluators.items()})
        print(system, "corpus span by evaluator", {k: round(v, 1) for k, v in across_corpora.items()})
    # RQ3 − RQ1 sob o avaliador do PT-OIE-Benchmark, pareado por sentença.
    attention_delta = {}
    for corpus in ("BIA",) + EXTERNAL:
        d = paired_delta(boots[("pt_oie", corpus)], boots[("ud_baseline", corpus)], "pt_oie", "ud_baseline")
        attention_delta[corpus] = {k: (v * 100 if k in ("delta_point_f1", "ci_low", "ci_high") else v)
                                   for k, v in d.items()}
        print("rq3 - rq1", corpus, {k: round(v, 2) for k, v in attention_delta[corpus].items() if isinstance(v, float)})
    write_json(args.output, {"grid": grid, "spans": spans, "rq3_minus_rq1_benchmark_evaluator": attention_delta})
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
