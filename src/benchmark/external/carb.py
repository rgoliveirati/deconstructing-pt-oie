"""Pontuação CaRB-like usando o avaliador do PT-OIE-Benchmark.

Nada é reimplementado: ``normalize_words``, ``get_token_f1`` e
``get_tuple_match_score`` são importados de ``Evaluation.py`` do clone do PT-OIE-Benchmark.
A única adição é a decomposição por sentença dos numeradores de precisão e
revocação, necessária para o bootstrap com reamostragem por sentença.

``score_dataset`` reproduz numericamente ``Evaluation.evaluate``; a igualdade é
verificada em ``assert_equivalent_to_benchmark_evaluator`` e exercitada em
``tests/test_external_benchmark.py``.

Convenções do avaliador do PT-OIE-Benchmark que são preservadas deliberadamente:

* o casamento sentença↔sentença é feito por ``frozenset`` de palavras
  normalizadas (sem acento, sem pontuação, minúsculas);
* o denominador da precisão é o total de linhas previstas, inclusive as de
  sentenças que não casam com nenhuma sentença gold;
* o denominador da revocação é o total de linhas gold, inclusive as de grupos
  ignorados no laço (chave vazia, ou sentença sem predições);
* a precisão usa casamento guloso um-para-um; a revocação usa multi-match.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRIPLE_COLUMNS = ["ARG0", "V", "ARG1"]


@lru_cache(maxsize=4)
def load_benchmark_evaluator(bench_dir: str):
    """Importa ``Evaluation.py`` do clone do PT-OIE-Benchmark como módulo."""
    path = Path(bench_dir) / "Evaluation.py"
    if not path.exists():
        raise FileNotFoundError(
            f"avaliador do PT-OIE-Benchmark não encontrado em {path}. "
            "Rode scripts/fetch_external_benchmark.py primeiro."
        )
    spec = importlib.util.spec_from_file_location("ptoie_benchmark_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for fn in ("normalize_words", "get_token_f1", "get_tuple_match_score", "evaluate"):
        if not hasattr(module, fn):
            raise AttributeError(f"{path} não expõe {fn}()")
    return module


def _key(series: pd.Series, normalize_words) -> list[frozenset]:
    return [frozenset(normalize_words(s)) for s in series]


def per_sentence_stats(gold_df: pd.DataFrame, pred_df: pd.DataFrame, bench_dir: str) -> dict[str, Any]:
    """Decompõe os numeradores do CaRB por sentença gold.

    Retorna ``{"sentences": [...], "n_pred_total": int, "n_gold_total": int,
    "n_pred_orphan": int}``. Cada item de ``sentences`` traz ``prec_num``,
    ``rec_num``, ``n_pred`` e ``n_gold``, de modo que somá-los reproduz os
    totais do avaliador do PT-OIE-Benchmark.
    """
    ev = load_benchmark_evaluator(bench_dir)
    gold_df = gold_df.copy()
    pred_df = pred_df.copy()
    gold_df["_key"] = _key(gold_df["Sentence"], ev.normalize_words)
    pred_df["_key"] = _key(pred_df["Sentence"], ev.normalize_words)

    pred_groups = dict(list(pred_df.groupby("_key")))
    gold_keys = set(gold_df["_key"])

    sentences: list[dict[str, Any]] = []
    # Ordem determinística e estável, independente da ordem de iteração do groupby.
    for key, gold_group in sorted(gold_df.groupby("_key"), key=lambda kv: sorted(kv[0])):
        n_gold = len(gold_group)
        pred_group = pred_groups.get(key)
        n_pred = 0 if pred_group is None else len(pred_group)

        prec_num = 0.0
        rec_num = 0.0
        # Réplica exata das guardas do avaliador do PT-OIE-Benchmark.
        if key and n_gold and n_pred:
            gold_tuples = [tuple(r) for r in gold_group[TRIPLE_COLUMNS].values]
            pred_tuples = [tuple(r) for r in pred_group[TRIPLE_COLUMNS].values]
            m = np.zeros((n_pred, n_gold))
            for i in range(n_pred):
                for j in range(n_gold):
                    m[i, j] = ev.get_tuple_match_score(pred_tuples[i], gold_tuples[j])
            rec_num = float(np.sum(np.max(m, axis=0)))
            tmp = m.copy()
            for _ in range(min(n_pred, n_gold)):
                best = tmp.max()
                if best == 0:
                    break
                pi, gi = np.unravel_index(np.argmax(tmp), tmp.shape)
                prec_num += float(best)
                tmp[pi, :] = 0
                tmp[:, gi] = 0

        sentences.append(
            {
                "key": key,
                "sentence": str(gold_group["Sentence"].iloc[0]),
                "n_gold": n_gold,
                "n_pred": n_pred,
                "prec_num": prec_num,
                "rec_num": rec_num,
            }
        )

    n_pred_orphan = int(sum(len(g) for k, g in pred_groups.items() if k not in gold_keys))
    return {
        "sentences": sentences,
        "n_pred_total": int(len(pred_df)),
        "n_gold_total": int(len(gold_df)),
        "n_pred_orphan": n_pred_orphan,
    }


def aggregate(n_pred_total: int, n_gold_total: int, prec_num: float, rec_num: float) -> dict[str, float]:
    """Micro P/R/F1 com os mesmos denominadores do avaliador do PT-OIE-Benchmark."""
    precision = prec_num / n_pred_total if n_pred_total > 0 else 0.0
    recall = rec_num / n_gold_total if n_gold_total > 0 else 0.0
    f1 = 0.0
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1_score": f1}


def score_dataset(gold_df: pd.DataFrame, pred_df: pd.DataFrame, bench_dir: str) -> dict[str, Any]:
    """Pontua um corpus e devolve métricas + decomposição por sentença."""
    stats = per_sentence_stats(gold_df, pred_df, bench_dir)
    prec_num = sum(s["prec_num"] for s in stats["sentences"])
    rec_num = sum(s["rec_num"] for s in stats["sentences"])
    metrics = aggregate(stats["n_pred_total"], stats["n_gold_total"], prec_num, rec_num)

    evaluated = sum(1 for s in stats["sentences"] if s["n_pred"] > 0)
    n_sent = len(stats["sentences"])
    return {
        **metrics,
        "total_extractions": stats["n_pred_total"],
        "gold_extractions": stats["n_gold_total"],
        "total_sentences": n_sent,
        "evaluated_sentences": evaluated,
        "sentences_without_extraction": n_sent - evaluated,
        "pct_sentences_without_extraction": (n_sent - evaluated) / n_sent * 100 if n_sent else 0.0,
        "extractions_per_sentence": stats["n_pred_total"] / n_sent if n_sent else 0.0,
        "n_pred_orphan": stats["n_pred_orphan"],
        "_per_sentence": stats["sentences"],
    }


def benchmark_evaluator_scores(gold_df: pd.DataFrame, pred_df: pd.DataFrame, bench_dir: str) -> dict[str, Any]:
    """Chama ``Evaluation.evaluate`` dos autores, silenciando seus prints."""
    import contextlib
    import io

    ev = load_benchmark_evaluator(bench_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        return ev.evaluate(gold_df, pred_df)


def assert_equivalent_to_benchmark_evaluator(
    gold_df: pd.DataFrame, pred_df: pd.DataFrame, bench_dir: str, tol: float = 1e-9
) -> dict[str, Any]:
    """Prova de equivalência numérica exigida antes de usar a decomposição."""
    ours = score_dataset(gold_df, pred_df, bench_dir)
    theirs = benchmark_evaluator_scores(gold_df, pred_df, bench_dir)
    diffs = {}
    for k in ("precision", "recall", "f1_score", "total_extractions", "gold_extractions"):
        delta = abs(float(ours[k]) - float(theirs[k]))
        diffs[k] = delta
        if delta > tol:
            raise AssertionError(
                f"divergência com o avaliador do PT-OIE-Benchmark em {k}: "
                f"nosso={ours[k]!r} benchmark={theirs[k]!r} (delta={delta})"
            )
    return {"ok": True, "max_abs_diff": max(diffs.values()), "per_metric_abs_diff": diffs}
