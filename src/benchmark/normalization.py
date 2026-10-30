"""Normalização textual e deduplicação usadas apenas na avaliação.

Os slots das predições nunca são alterados semanticamente: a normalização
produz chaves de comparação; o texto original é preservado nos arquivos
`raw/` e `normalized/`.
"""

from __future__ import annotations

import re
import unicodedata

_RX_WS = re.compile(r"\s+")
_RX_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def norm_space_lower(s: str) -> str:
    """Minúsculas + colapso de espaços. Base do protocolo strict."""
    return _RX_WS.sub(" ", (s or "").strip().lower())


def tokenize(s: str) -> list[str]:
    """Tokenização para F1 de tokens: minúsculas, pontuação removida,
    separação por espaço. Acentos são preservados."""
    s = norm_space_lower(s)
    s = _RX_PUNCT.sub(" ", s)
    return [t for t in _RX_WS.split(s) if t]


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def dedup_key(arg1: str, rel: str, arg2: str) -> tuple[str, str, str]:
    return (norm_space_lower(arg1), norm_space_lower(rel), norm_space_lower(arg2))


def dedup_predictions(preds: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicatas exatas (após minúsculas/espaços) mantendo a primeira
    ocorrência, na ordem original. Aplicada de forma idêntica a todos os
    sistemas, somente na avaliação. Retorna (únicos, n_removidos)."""
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    removed = 0
    for p in preds:
        key = dedup_key(p.get("arg1", ""), p.get("rel", ""), p.get("arg2", ""))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(p)
    return out, removed


def token_f1(pred: str, gold: str) -> float:
    """F1 de tokens (multiconjunto) entre dois spans."""
    pt, gt = tokenize(pred), tokenize(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    from collections import Counter

    common = Counter(pt) & Counter(gt)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pt)
    r = n_common / len(gt)
    return 2 * p * r / (p + r)
