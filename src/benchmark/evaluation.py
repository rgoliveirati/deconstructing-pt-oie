"""Protocolos de avaliação do benchmark.

Quatro protocolos, aplicados identicamente a todos os sistemas:

- ``strict``: igualdade exata dos três slots após minúsculas e colapso de
  espaços; atribuição um-para-um.
- ``tolerant``: F1 de tokens por slot; par elegível quando o menor F1 de
  slot é >= 0.70; atribuição um-para-um pelo F1 médio dos slots.
- ``carb_style``: score ponderado 0.35*F1(arg1) + 0.30*F1(rel) + 0.35*F1(arg2),
  par elegível quando >= 0.60; atribuição um-para-um pelo score.
- ``bia_legacy``: o avaliador legado do projeto, reutilizado sem alteração via ``src.extractor.evaluate_dataset_legacy``
  e sem deduplicação (fiel aos experimentos históricos do artigo).
  Matching guloso na ordem do gold com heurísticas de canonicalização
  (expansão de contrações, remoção de determinantes iniciais, prefixo,
  sufixo, subconjunto de palavras e relação parcial). É um-para-um por
  construção (cada predição casa com no máximo um gold e vice-versa),
  mas a ordem de varredura difere dos protocolos padronizados; por isso é
  mantido sob identificador próprio e não foi reimplementado.

Somente triplas gold com ``valid=true`` contam (comportamento do avaliador
legado, estendido igualmente aos demais protocolos). A deduplicação exata
(minúsculas/espaços) é aplicada às predições de todos os sistemas antes do
matching, apenas na avaliação.
"""

from __future__ import annotations

from typing import Any

from .assignment import one_to_one_assignment
from .corpus import Sentence
from .normalization import dedup_predictions, norm_space_lower, token_f1

PROTOCOLS = ("strict", "tolerant", "bia_legacy", "carb_style")

TOLERANT_MIN_SLOT_F1 = 0.70
CARB_THRESHOLD = 0.60
CARB_WEIGHTS = (0.35, 0.30, 0.35)  # arg1, rel, arg2


def _micro(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _slot_scores(pred: dict, gold: dict) -> tuple[float, float, float]:
    return (
        token_f1(pred.get("arg1", ""), gold["arg1"]),
        token_f1(pred.get("rel", ""), gold["rel"]),
        token_f1(pred.get("arg2", ""), gold["arg2"]),
    )


def _eval_standard(
    protocol: str,
    sentences: list[Sentence],
    preds_by_sent: list[list[dict]],
) -> dict[str, Any]:
    per_sentence: list[dict[str, Any]] = []
    tp = fp = fn = 0
    n_dedup_removed = 0

    for sent, raw_preds in zip(sentences, preds_by_sent):
        preds, removed = dedup_predictions(raw_preds)
        n_dedup_removed += removed
        golds = [
            {"arg1": g.arg1, "rel": g.rel, "arg2": g.arg2}
            for g in sent.gold
            if g.valid
        ]

        if protocol == "strict":
            def score_fn(gi: int, pi: int) -> float:
                g, p = golds[gi], preds[pi]
                ok = (
                    norm_space_lower(p.get("arg1", "")) == norm_space_lower(g["arg1"])
                    and norm_space_lower(p.get("rel", "")) == norm_space_lower(g["rel"])
                    and norm_space_lower(p.get("arg2", "")) == norm_space_lower(g["arg2"])
                )
                return 1.0 if ok else 0.0

            threshold = 1.0
        elif protocol == "tolerant":
            def score_fn(gi: int, pi: int) -> float:
                s = _slot_scores(preds[pi], golds[gi])
                if min(s) < TOLERANT_MIN_SLOT_F1:
                    return -1.0
                return sum(s) / 3.0

            threshold = 0.0
        elif protocol == "carb_style":
            def score_fn(gi: int, pi: int) -> float:
                a1, r, a2 = _slot_scores(preds[pi], golds[gi])
                w1, wr, w2 = CARB_WEIGHTS
                return w1 * a1 + wr * r + w2 * a2

            threshold = CARB_THRESHOLD
        else:  # pragma: no cover
            raise ValueError(f"protocolo desconhecido: {protocol}")

        matches = one_to_one_assignment(len(golds), len(preds), score_fn, threshold)
        s_tp = len(matches)
        s_fp = len(preds) - s_tp
        s_fn = len(golds) - s_tp
        tp, fp, fn = tp + s_tp, fp + s_fp, fn + s_fn
        per_sentence.append(
            {
                "sentence_id": sent.sentence_id,
                "tp": s_tp,
                "fp": s_fp,
                "fn": s_fn,
                "n_pred": len(preds),
                "n_gold": len(golds),
                "n_dedup_removed": removed,
                "matches": [
                    {
                        "gold_index": gi,
                        "pred_index": pi,
                        "score": round(score, 6),
                        "gold": golds[gi],
                        "pred": {
                            "arg1": preds[pi].get("arg1", ""),
                            "rel": preds[pi].get("rel", ""),
                            "arg2": preds[pi].get("arg2", ""),
                            "triple_id": preds[pi].get("triple_id"),
                        },
                    }
                    for gi, pi, score in matches
                ],
                "unmatched_gold": [
                    golds[gi] for gi in range(len(golds))
                    if gi not in {m[0] for m in matches}
                ],
                "unmatched_pred": [
                    {
                        "arg1": preds[pi].get("arg1", ""),
                        "rel": preds[pi].get("rel", ""),
                        "arg2": preds[pi].get("arg2", ""),
                        "triple_id": preds[pi].get("triple_id"),
                    }
                    for pi in range(len(preds))
                    if pi not in {m[1] for m in matches}
                ],
            }
        )

    result = {
        "protocol": protocol,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_dedup_removed": n_dedup_removed,
        **_micro(tp, fp, fn),
        "per_sentence": per_sentence,
        "matching": "one_to_one_greedy_by_score",
        "criteria": {
            "strict": "igualdade exata por slot após lower+espaços",
            "tolerant": f"min F1 de slot >= {TOLERANT_MIN_SLOT_F1}",
            "carb_style": f"0.35*F1(arg1)+0.30*F1(rel)+0.35*F1(arg2) >= {CARB_THRESHOLD}",
        }.get(protocol, ""),
    }
    return result


def _eval_bia_legacy(
    sentences: list[Sentence],
    preds_by_sent: list[list[dict]],
) -> dict[str, Any]:
    # Import tardio: puxa torch/stanza/transformers.
    from src.extractor import LegacyConfig, evaluate_dataset_legacy

    config = LegacyConfig()  # flags de matching idênticas às dos experimentos
    # O protocolo legado é preservado exatamente como nos experimentos do
    # artigo, que NÃO aplicavam deduplicação na avaliação — com dedup os
    # números históricos (ex.: F1 54.92 do rq3_attn_on_thr0) não se
    # reproduzem. A deduplicação uniforme vale só para os protocolos
    # padronizados (strict/tolerant/carb_style).
    deduped: list[list[dict]] = [list(p) for p in preds_by_sent]
    n_dedup_removed = 0

    gold_rows = [
        {
            "sentence": s.sentence,
            "gold": [
                {"arg1": g.arg1, "rel": g.rel, "arg2": g.arg2, "valid": g.valid}
                for g in s.gold
            ],
        }
        for s in sentences
    ]
    legacy = evaluate_dataset_legacy(deduped, gold_rows, config)

    per_sentence: list[dict[str, Any]] = []
    for sent, detail in zip(sentences, legacy["details"]):
        s_tp = sum(detail["pred_matched"])
        s_fp = len(detail["pred"]) - s_tp
        s_fn = len(detail["gold"]) - sum(detail["gold_matched"])
        per_sentence.append(
            {
                "sentence_id": sent.sentence_id,
                "tp": s_tp,
                "fp": s_fp,
                "fn": s_fn,
                "n_pred": len(detail["pred"]),
                "n_gold": len(detail["gold"]),
                "matches": [],
                "unmatched_gold": [
                    g for g, m in zip(detail["gold"], detail["gold_matched"]) if not m
                ],
                "unmatched_pred": [
                    {
                        "arg1": p.get("arg1", ""),
                        "rel": p.get("rel", ""),
                        "arg2": p.get("arg2", ""),
                        "triple_id": p.get("triple_id"),
                    }
                    for p, m in zip(detail["pred"], detail["pred_matched"])
                    if not m
                ],
            }
        )

    return {
        "protocol": "bia_legacy",
        "tp": legacy["TP"],
        "fp": legacy["FP"],
        "fn": legacy["FN"],
        "n_dedup_removed": n_dedup_removed,
        "precision": legacy["precision"],
        "recall": legacy["recall"],
        "f1": legacy["f1"],
        "per_sentence": per_sentence,
        "matching": "legacy_greedy_gold_order_one_to_one",
        "criteria": (
            "canonicalização legada (contrações expandidas, determinantes iniciais "
            "removidos, pontuação removida) + igualdade/sufixo/prefixo/subconjunto "
            "para argumentos e igualdade/prefixo parcial para relação"
        ),
    }


def evaluate_protocol(
    protocol: str,
    sentences: list[Sentence],
    preds_by_sent: list[list[dict]],
) -> dict[str, Any]:
    if protocol == "bia_legacy":
        return _eval_bia_legacy(sentences, preds_by_sent)
    if protocol in ("strict", "tolerant", "carb_style"):
        return _eval_standard(protocol, sentences, preds_by_sent)
    raise ValueError(f"protocolo desconhecido: {protocol}")
