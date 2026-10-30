"""Teto de revocação do gerador de candidatos nos corpora do PT-OIE-Benchmark.

Para cada corpus (WIKI200, CETEN200, Gamalho), roda a configuração de referência
(``rq3_attn_on_thr0``) e registra, sob a revocação do avaliador do PT-OIE-Benchmark
(para cada tripla de referência, o melhor escore entre as predições):

* a revocação das triplas finais;
* o teto: a mesma revocação se todos os candidatos que o gerador produz fossem
  mantidos (``_generate_all_candidates``, antes de validação e de atenção);
* as sentenças sem nenhuma tripla final.

As referências são agrupadas pela chave do avaliador (conjunto de palavras
normalizadas da sentença). Campos vazios seguem a leitura do avaliador: NaN.

Uso:
    python scripts/generative_ceiling.py --output outputs/external_benchmark/generative_ceiling.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.external.carb import load_benchmark_evaluator  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402

BENCH = REPO / ".external/PT-OIE-Benchmark"
CORPORA = ("WIKI200", "CETEN200", "Gamalho")
SLOTS = ("ARG0", "V", "ARG1")


def clean(s) -> str | float:
    s = " ".join(str(s or "").split())
    return s if s else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/external_benchmark/generative_ceiling.json")
    args = p.parse_args(argv)

    from src.extractor import Config, OpenIEExtractorBIA

    ev = load_benchmark_evaluator(str(BENCH))
    key = lambda s: frozenset(ev.normalize_words(s))  # noqa: E731
    ex = OpenIEExtractorBIA(config=Config(
        bert_model="neuralmind/bert-base-portuguese-cased",
        bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"),
        seed=13, cop_mode="full", theory_mode="off", apply_theoretical_rules=False,
        no_attn=False, attn_decision_enabled=True, attn_rerank_enabled=True,
        heads_mode="rank", top_k_heads=10, attn_threshold=0.0,
    ), verbose=False)

    result = {"created_at": now_iso(), "config": "rq3_attn_on_thr0", "corpora": {}}
    for corpus in CORPORA:
        data = load_dataset(corpus, BENCH)
        gold_by = defaultdict(list)
        for _, r in data.gold.iterrows():
            gold_by[key(r["Sentence"])].append(tuple(r[s] for s in SLOTS))
        final_by, cand_by, empty = defaultdict(list), defaultdict(list), 0
        for s in data.sentences:
            rows, _ = ex.extract(s)
            cands, _, _ = ex._generate_all_candidates(s)
            final_by[key(s)] += [tuple(clean(r[k]) for k in ("arg1", "rel", "arg2")) for r in rows]
            cand_by[key(s)] += [tuple(clean(getattr(c, k)) for k in ("arg1", "rel", "arg2")) for c in cands]
            empty += not rows
        n_gold = sum(map(len, gold_by.values()))
        best = lambda preds, g: max((ev.get_tuple_match_score(t, g) for t in preds), default=0.0)  # noqa: E731
        recall = sum(best(final_by.get(k, []), g) for k, gs in gold_by.items() if k for g in gs) / n_gold
        ceiling = sum(best(cand_by.get(k, []), g) for k, gs in gold_by.items() if k for g in gs) / n_gold
        result["corpora"][corpus] = {
            "n_sentences": len(data.sentences), "n_gold": n_gold,
            "sentences_without_extraction": empty,
            "pct_sentences_without_extraction": 100 * empty / len(data.sentences),
            "recall": 100 * recall, "recall_ceiling_all_candidates": 100 * ceiling,
            "headroom_points": 100 * (ceiling - recall),
        }
        print(corpus, {k: round(v, 2) for k, v in result["corpora"][corpus].items()}, flush=True)
    write_json(args.output, result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
