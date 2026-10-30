"""Atribuição gulosa contra atribuição ótima sob ``bia_legacy`` no BIA.

O ``bia_legacy`` percorre as referências na ordem do corpus e dá a cada uma a
primeira predição livre que casa com ela. Este script refaz o casamento com o
emparelhamento bipartido máximo (caminhos aumentantes de Kuhn) sobre o mesmo
critério de casamento (``triple_matches`` do extrator, com ``LegacyConfig``) e
compara TP e F1.

Uso:
    python scripts/greedy_vs_optimal.py --output outputs/benchmark/metrics/greedy_vs_optimal_bia_legacy.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import evaluate_protocol  # noqa: E402
from src.benchmark.schemas import read_jsonl  # noqa: E402

SYSTEMS = ("pt_oie", "ud_baseline", "ollama_gemma4", "multi2oie", "dptoie")


def max_matching(edges: list[list[int]], n_pred: int) -> int:
    """Tamanho do emparelhamento máximo; ``edges[g]`` lista as predições que casam com a referência ``g``."""
    owner = [-1] * n_pred

    def augment(g: int, seen: set[int]) -> bool:
        for p in edges[g]:
            if p not in seen:
                seen.add(p)
                if owner[p] == -1 or augment(owner[p], seen):
                    owner[p] = g
                    return True
        return False

    return sum(augment(g, set()) for g in range(len(edges)))


def f1(tp: int, fp: int, fn: int) -> float:
    return 200 * tp / (2 * tp + fp + fn) if tp else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/benchmark/metrics/greedy_vs_optimal_bia_legacy.csv")
    args = p.parse_args(argv)

    import src.extractor as ext

    config = ext.LegacyConfig()
    sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    rows = []
    for system in SYSTEMS:
        by = {}
        for r in read_jsonl(REPO / "outputs/benchmark/normalized" / f"{system}.jsonl"):
            by.setdefault(r["sentence_id"], []).append({k: r.get(k) or "" for k in ("arg1", "rel", "arg2")})
        preds = [by.get(s.sentence_id, []) for s in sents]
        greedy = evaluate_protocol("bia_legacy", sents, preds)
        tp = 0
        for s, ps in zip(sents, preds):
            golds = [{"arg1": g.arg1, "rel": g.rel, "arg2": g.arg2} for g in s.gold if g.valid]
            tp += max_matching([[i for i, q in enumerate(ps) if ext.triple_matches(q, g, config)] for g in golds], len(ps))
        n_pred, n_gold = sum(map(len, preds)), greedy["tp"] + greedy["fn"]
        g_f1 = f1(greedy["tp"], greedy["fp"], greedy["fn"])
        o_f1 = f1(tp, n_pred - tp, n_gold - tp)
        rows.append({"system": system, "greedy_tp": greedy["tp"], "greedy_fp": greedy["fp"], "greedy_fn": greedy["fn"],
                     "greedy_f1": round(g_f1, 2), "optimal_tp": tp, "optimal_fp": n_pred - tp, "optimal_fn": n_gold - tp,
                     "optimal_f1": round(o_f1, 2), "delta_tp": tp - greedy["tp"], "delta_f1_points": round(o_f1 - g_f1, 2)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(args.output, sep=";", index=False)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
