"""Bootstrap com reamostragem por SENTENÇA sobre os numeradores do CaRB.

Diferença em relação a ``src/benchmark/bootstrap.py``: lá as métricas vêm de
contagens inteiras (tp/fp/fn); aqui o CaRB é de pontuação contínua, e o que se
reamostra são os numeradores parciais e os denominadores de cada sentença.

A matriz de índices é compartilhada entre sistemas (via ``make_indices`` do
módulo de bootstrap já existente), o que torna as diferenças pareadas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..bootstrap import CI_HIGH, CI_LOW, DEFAULT_N_SAMPLES, DEFAULT_SEED, make_indices

__all__ = ["bootstrap_carb", "paired_delta", "make_indices", "DEFAULT_N_SAMPLES", "DEFAULT_SEED"]


def _arrays(per_sentence: list[dict[str, Any]]):
    return (
        np.array([s["prec_num"] for s in per_sentence], dtype=float),
        np.array([s["rec_num"] for s in per_sentence], dtype=float),
        np.array([s["n_pred"] for s in per_sentence], dtype=float),
        np.array([s["n_gold"] for s in per_sentence], dtype=float),
    )


def bootstrap_carb(
    per_sentence: list[dict[str, Any]],
    indices: np.ndarray,
    n_pred_orphan: int = 0,
) -> dict[str, Any]:
    """IC95 percentil para P/R/F1 do CaRB, reamostrando sentenças.

    ``n_pred_orphan`` são linhas previstas cuja sentença não casa com nenhuma
    sentença gold. Elas entram no denominador da precisão do avaliador do PT-OIE-Benchmark
    mas não pertencem a nenhuma unidade de reamostragem; como a reamostragem
    sorteia exatamente ``n_sentences`` unidades, seu peso esperado é 1 e elas
    são somadas como constante em cada réplica.
    """
    prec_num, rec_num, n_pred, n_gold = _arrays(per_sentence)

    pn = prec_num[indices].sum(axis=1)
    rn = rec_num[indices].sum(axis=1)
    dp = n_pred[indices].sum(axis=1) + float(n_pred_orphan)
    dg = n_gold[indices].sum(axis=1)

    p = np.divide(pn, dp, out=np.zeros_like(pn), where=dp > 0)
    r = np.divide(rn, dg, out=np.zeros_like(rn), where=dg > 0)
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(p), where=denom > 0)

    def ci(arr: np.ndarray) -> dict[str, float]:
        lo, hi = np.percentile(arr, [CI_LOW, CI_HIGH])
        return {"mean": float(arr.mean()), "low": float(lo), "high": float(hi)}

    tot_dp = float(n_pred.sum()) + float(n_pred_orphan)
    tot_dg = float(n_gold.sum())
    point_p = float(prec_num.sum() / tot_dp) if tot_dp > 0 else 0.0
    point_r = float(rec_num.sum() / tot_dg) if tot_dg > 0 else 0.0
    point_f1 = 2 * point_p * point_r / (point_p + point_r) if (point_p + point_r) > 0 else 0.0

    return {
        "point": {"precision": point_p, "recall": point_r, "f1": point_f1},
        "precision": ci(p),
        "recall": ci(r),
        "f1": ci(f1),
        "n_samples": int(indices.shape[0]),
        "n_sentences": int(indices.shape[1]),
        "ci_method": "percentil 2.5/97.5 sobre reamostragem por sentença com reposição",
        "_f1_samples": f1,
    }


def paired_delta(res_a: dict[str, Any], res_b: dict[str, Any], label_a: str, label_b: str) -> dict[str, Any]:
    """Diferença pareada F1(A) − F1(B) nas MESMAS réplicas."""
    da = np.asarray(res_a["_f1_samples"])
    db = np.asarray(res_b["_f1_samples"])
    if da.shape != db.shape:
        raise ValueError("réplicas incompatíveis: os dois sistemas precisam usar a mesma matriz de índices")
    delta = da - db
    lo, hi = np.percentile(delta, [CI_LOW, CI_HIGH])
    return {
        "system_a": label_a,
        "system_b": label_b,
        "delta_point_f1": res_a["point"]["f1"] - res_b["point"]["f1"],
        "delta_mean": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_delta_gt_0": float((delta > 0).mean()),
        "significant_95": bool(lo > 0 or hi < 0),
    }
