"""Bootstrap pareado por sentença (unidade de reamostragem = sentença)."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_N_SAMPLES = 10_000
DEFAULT_SEED = 42
CI_LOW, CI_HIGH = 2.5, 97.5


def _micro_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return p, r, f1


def make_indices(n_sentences: int, n_samples: int = DEFAULT_N_SAMPLES,
                 seed: int = DEFAULT_SEED) -> np.ndarray:
    """Matriz de índices (n_samples, n_sentences) compartilhada entre sistemas
    para que as diferenças sejam pareadas."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_sentences, size=(n_samples, n_sentences))


def bootstrap_metrics(per_sentence_counts: list[dict], indices: np.ndarray) -> dict[str, Any]:
    """`per_sentence_counts`: lista com dicts {tp, fp, fn} na ordem do corpus."""
    tp = np.array([c["tp"] for c in per_sentence_counts], dtype=float)
    fp = np.array([c["fp"] for c in per_sentence_counts], dtype=float)
    fn = np.array([c["fn"] for c in per_sentence_counts], dtype=float)

    tp_s = tp[indices].sum(axis=1)
    fp_s = fp[indices].sum(axis=1)
    fn_s = fn[indices].sum(axis=1)
    p, r, f1 = _micro_from_counts(tp_s, fp_s, fn_s)

    def ci(arr: np.ndarray) -> dict[str, float]:
        lo, hi = np.percentile(arr, [CI_LOW, CI_HIGH])
        return {"mean": float(arr.mean()), "low": float(lo), "high": float(hi)}

    point_p, point_r, point_f1 = _micro_from_counts(
        np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()])
    )
    return {
        "point": {
            "precision": float(point_p[0]),
            "recall": float(point_r[0]),
            "f1": float(point_f1[0]),
        },
        "precision": ci(p),
        "recall": ci(r),
        "f1": ci(f1),
        "n_samples": int(indices.shape[0]),
        "n_sentences": int(indices.shape[1]),
        "_f1_samples": f1,  # reutilizado nas diferenças pareadas
    }


def paired_f1_difference(res_a: dict[str, Any], res_b: dict[str, Any]) -> dict[str, Any]:
    """Diferença pareada F1(A) - F1(B) sobre as mesmas reamostragens."""
    da = res_a["_f1_samples"]
    db = res_b["_f1_samples"]
    delta = da - db
    lo, hi = np.percentile(delta, [CI_LOW, CI_HIGH])
    return {
        "delta_point": res_a["point"]["f1"] - res_b["point"]["f1"],
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "prop_delta_gt_0": float((delta > 0).mean()),
        "prop_delta_lt_0": float((delta < 0).mean()),
        "n_samples": int(delta.shape[0]),
    }
