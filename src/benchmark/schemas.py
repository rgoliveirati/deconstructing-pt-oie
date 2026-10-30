"""Esquema normalizado de predições e registros por sentença."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

PREDICTION_REQUIRED_FIELDS = (
    "sentence_id", "sentence", "system", "triple_id",
    "arg1", "rel", "arg2", "confidence", "raw_output",
    "status", "error", "runtime", "metadata",
)

VALID_STATUS = {"ok", "error", "empty", "timeout"}


@dataclass
class RuntimeInfo:
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0


@dataclass
class Prediction:
    """Uma tripla extraída, no esquema normalizado do benchmark."""

    sentence_id: str
    sentence: str
    system: str
    triple_id: str
    arg1: str
    rel: str
    arg2: str
    confidence: Optional[float] = None
    raw_output: Any = None
    status: str = "ok"
    error: Optional[str] = None
    runtime: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SentenceRecord:
    """Registro por sentença: saída bruta, status, erro e tempos.

    É a unidade dos arquivos `raw/<system>.jsonl` e `errors/<system>.jsonl`.
    Sentenças sem triplas têm `n_triples == 0` e `status == "ok"` apenas
    quando a lista vazia foi retornada explicitamente pelo sistema.
    """

    sentence_id: str
    sentence: str
    system: str
    status: str = "ok"
    n_triples: int = 0
    raw_output: Any = None
    error: Optional[str] = None
    runtime: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_prediction_dict(row: dict[str, Any]) -> list[str]:
    """Retorna a lista de problemas do registro (vazia quando válido)."""
    problems: list[str] = []
    for f in PREDICTION_REQUIRED_FIELDS:
        if f not in row:
            problems.append(f"campo ausente: {f}")
    for f in ("arg1", "rel", "arg2"):
        if f in row and not isinstance(row[f], str):
            problems.append(f"campo {f} deve ser string")
    if row.get("status") not in VALID_STATUS:
        problems.append(f"status inválido: {row.get('status')!r}")
    rt = row.get("runtime")
    if not isinstance(rt, dict):
        problems.append("runtime deve ser objeto")
    return problems


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            if hasattr(r, "to_dict"):
                r = r.to_dict()
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path, row) -> None:
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
