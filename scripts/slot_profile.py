"""F1 por campo e comprimento da relação do PT-OIE-EXTRACTOR nos corpora externos.

Para cada predição de ``pt_oie`` (``rq3_attn_on_thr0``), toma a referência da
mesma sentença com maior escore no avaliador do PT-OIE-Benchmark e registra o F1
de palavras de cada campo (ARG0, V, ARG1) contra ela; reporta a média por corpus.
Registra também o comprimento médio da relação, em tokens separados por espaço, nas
referências e nas predições.

Uso:
    python scripts/slot_profile.py --output outputs/external_benchmark/slot_profile.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.benchmark.external.carb import load_benchmark_evaluator  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.runtime import write_json  # noqa: E402

BENCH = REPO / ".external/PT-OIE-Benchmark"
PRED_DIR = REPO / "outputs/external_benchmark/predictions/benchmark_format/pt_oie"
CORPORA = ("WIKI200", "CETEN200", "Gamalho")
SLOTS = ("ARG0", "V", "ARG1")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/external_benchmark/slot_profile.json")
    args = p.parse_args(argv)

    ev = load_benchmark_evaluator(str(BENCH))
    key = lambda s: frozenset(ev.normalize_words(s))  # noqa: E731
    length = lambda s: len(str(s).split())  # noqa: E731
    result = {}
    for corpus in CORPORA:
        gold = load_dataset(corpus, BENCH).gold
        pred = pd.read_csv(PRED_DIR / f"{corpus}_results.csv", sep="\t")
        gold_by = defaultdict(list)
        for _, r in gold.iterrows():
            gold_by[key(r["Sentence"])].append(tuple(r[s] for s in SLOTS))
        sums, n = [0.0, 0.0, 0.0], 0
        for _, r in pred.iterrows():
            pt = tuple(r[s] for s in SLOTS)
            gs = gold_by.get(key(r["Sentence"]))
            if not gs:
                continue
            gt = max(gs, key=lambda g: ev.get_tuple_match_score(pt, g))
            for i in range(3):
                sums[i] += ev.get_token_f1(ev.normalize_words(pt[i]), ev.normalize_words(gt[i]))
            n += 1
        result[corpus] = {
            "n_predictions": n,
            "slot_f1": {s: sums[i] / n for i, s in enumerate(SLOTS)},
            "gold_rel_tokens": gold["V"].map(length).mean(),
            "pred_rel_tokens": pred["V"].map(length).mean(),
        }
        r = result[corpus]
        print(corpus, {s: round(v, 2) for s, v in r["slot_f1"].items()},
              "rel gold", round(r["gold_rel_tokens"], 2), "pred", round(r["pred_rel_tokens"], 2))
    write_json(args.output, result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
