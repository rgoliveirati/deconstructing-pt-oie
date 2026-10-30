"""Baseline do extrator no OIEC-PT (conjunto de DESENVOLVIMENTO).

Reporta, além de P/R/F1: o teto de revocação do gerador de candidatos e a taxa
de acerto sobre os NEGATIVOS ROTULADOS (candidatos que os juízes humanos
rejeitaram). O segundo número é o guarda-corpo desta etapa: um padrão novo que
aumenta o F1 mas passa a gerar triplas humanamente rejeitadas está errado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BENCH = REPO / ".external/PT-OIE-Benchmark"
sys.path.insert(0, str(BENCH))

from src.benchmark.oiec import load_oiec  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402

DEFAULT_CORPUS = REPO / "data/external/OIEC-PT/OIEC-PT.txt"
MATCH_THRESHOLD = 0.30


def clean(s):
    return " ".join(str(s).replace("\t", " ").replace("\n", " ").split())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--output", default="outputs/dev_oiec/baseline.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config-overrides", default=None,
                   help="JSON com overrides de Config (ex.: '{\"obl_core_policy\":\"governed\"}')")
    p.add_argument("--label", default="baseline")
    args = p.parse_args(argv)

    from Evaluation import get_tuple_match_score  # avaliador do PT-OIE-Benchmark
    from src.extractor import Config, OpenIEExtractorBIA

    kwargs = dict(
        bert_model="neuralmind/bert-base-portuguese-cased",
        bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"),
        seed=13, cop_mode="full", theory_mode="off", apply_theoretical_rules=False,
        no_attn=False, attn_decision_enabled=True, attn_rerank_enabled=True,
        heads_mode="rank", top_k_heads=10, attn_threshold=0.0,
    )
    if args.config_overrides:
        import json as _json
        kwargs.update(_json.loads(args.config_overrides))
    ex = OpenIEExtractorBIA(config=Config(**kwargs), verbose=False)

    sents = load_oiec(args.corpus)
    if args.limit:
        sents = sents[: args.limit]

    n_gold = n_pred = n_rejected_total = 0
    rec_final = rec_ceiling = 0.0
    hits_on_rejected = 0
    n_empty = 0

    for s in sents:
        gold = [(t.arg1, t.rel, t.arg2) for t in s.gold]
        rejected = [(t.arg1, t.rel, t.arg2) for t in s.rejected]
        n_gold += len(gold)
        n_rejected_total += len(rejected)

        rows, _ = ex.extract(s.sentence)
        final = [(clean(r["arg1"]), clean(r["rel"]), clean(r["arg2"])) for r in rows
                 if clean(r["arg1"]) and clean(r["rel"]) and clean(r["arg2"])]
        cands, _, _ = ex._generate_all_candidates(s.sentence)
        allc = [(clean(c.arg1), clean(c.rel), clean(c.arg2)) for c in cands
                if clean(c.arg1) and clean(c.rel) and clean(c.arg2)]
        n_pred += len(final)
        if not final:
            n_empty += 1

        for g in gold:
            rec_final += max((get_tuple_match_score(p_, g) for p_ in final), default=0.0)
            rec_ceiling += max((get_tuple_match_score(p_, g) for p_ in allc), default=0.0)

        # quantas predições finais casam com algo que humanos REJEITARAM
        for p_ in final:
            best_g = max((get_tuple_match_score(p_, g) for g in gold), default=0.0)
            best_r = max((get_tuple_match_score(p_, r) for r in rejected), default=0.0)
            if best_r >= MATCH_THRESHOLD and best_r > best_g:
                hits_on_rejected += 1

    # precisão no mesmo espírito do CaRB (crédito parcial, casamento guloso omitido
    # aqui por simplicidade: usamos o melhor casamento por predição, que é um limite superior)
    recall = rec_final / n_gold if n_gold else 0.0
    ceiling = rec_ceiling / n_gold if n_gold else 0.0

    result = {
        "created_at": now_iso(),
        "label": args.label,
        "corpus": str(args.corpus),
        "config_overrides": args.config_overrides,
        "n_sentences": len(sents),
        "n_gold_valid": n_gold,
        "n_rejected_labeled": n_rejected_total,
        "n_predictions": n_pred,
        "predictions_per_sentence": n_pred / len(sents) if sents else 0.0,
        "sentences_without_extraction": n_empty,
        "pct_sentences_without_extraction": n_empty / len(sents) * 100 if sents else 0.0,
        "recall": recall * 100,
        "recall_ceiling_all_candidates": ceiling * 100,
        "headroom_points": (ceiling - recall) * 100,
        "predictions_matching_human_rejected": hits_on_rejected,
        "pct_predictions_on_rejected": hits_on_rejected / n_pred * 100 if n_pred else 0.0,
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)

    print(f"OIEC-PT ({args.label}): {len(sents)} sentenças, {n_gold} triplas válidas, "
          f"{n_rejected_total} rejeitadas por humanos")
    print(f"  predições                        {n_pred} ({result['predictions_per_sentence']:.2f}/sent)")
    print(f"  sentenças sem extração           {n_empty} ({result['pct_sentences_without_extraction']:.1f}%)")
    print(f"  revocação                        {result['recall']:.1f}%")
    print(f"  teto de revocação (candidatos)   {result['recall_ceiling_all_candidates']:.1f}%")
    print(f"  predições sobre REJEITADAS       {hits_on_rejected} ({result['pct_predictions_on_rejected']:.1f}%)")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
