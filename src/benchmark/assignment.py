"""Atribuição um-para-um entre predições e triplas gold de uma sentença.

Algoritmo determinístico: todos os pares elegíveis são ordenados por
(-score, índice do gold, índice da predição) e atribuídos gulosamente.
Cada predição corresponde a no máximo um gold e vice-versa.
"""

from __future__ import annotations

from typing import Callable


def one_to_one_assignment(
    n_gold: int,
    n_pred: int,
    score_fn: Callable[[int, int], float],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Retorna a lista de correspondências [(gold_idx, pred_idx, score)].

    Só participam pares com score >= threshold. Empates são resolvidos
    deterministicamente pela ordem (gold_idx, pred_idx).
    """
    pairs: list[tuple[float, int, int]] = []
    for gi in range(n_gold):
        for pi in range(n_pred):
            s = score_fn(gi, pi)
            if s >= threshold:
                pairs.append((s, gi, pi))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for s, gi, pi in pairs:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, s))
    matches.sort(key=lambda t: (t[0], t[1]))
    return matches
