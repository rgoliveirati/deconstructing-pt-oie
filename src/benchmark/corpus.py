"""Carga e validação do corpus BIA para o benchmark.

O corpus não é modificado: as sentenças e triplas gold são lidas do arquivo
canônico e verificadas contra os totais publicados (262 sentenças, 427 triplas).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXPECTED_SENTENCES = 262
EXPECTED_GOLD_TRIPLES = 427


@dataclass
class GoldTriple:
    arg1: str
    rel: str
    arg2: str
    valid: bool = True


@dataclass
class Sentence:
    sentence_id: str
    sentence: str
    doc_id: Any
    phrase_index: Any
    gold: list[GoldTriple] = field(default_factory=list)


def corpus_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bia(path: str | Path) -> list[Sentence]:
    sentences: list[Sentence] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = f"bia_{int(row['doc_id']):04d}_{int(row['phrase_index']):02d}"
            gold = [
                GoldTriple(
                    arg1=g["arg1"], rel=g["rel"], arg2=g["arg2"],
                    valid=bool(g.get("valid", True)),
                )
                for g in row.get("gold", [])
            ]
            sentences.append(
                Sentence(
                    sentence_id=sid,
                    sentence=row["sentence"],
                    doc_id=row.get("doc_id"),
                    phrase_index=row.get("phrase_index"),
                    gold=gold,
                )
            )
    return sentences


def validate_corpus(path: str | Path) -> dict[str, Any]:
    sentences = load_bia(path)
    ids = [s.sentence_id for s in sentences]
    n_gold = sum(len(s.gold) for s in sentences)
    n_gold_valid = sum(1 for s in sentences for g in s.gold if g.valid)
    report = {
        "path": str(path),
        "sha256": corpus_sha256(path),
        "n_sentences": len(sentences),
        "n_gold_triples": n_gold,
        "n_gold_triples_valid": n_gold_valid,
        "unique_ids": len(set(ids)) == len(ids),
        "empty_sentences": sum(1 for s in sentences if not s.sentence.strip()),
        "expected_sentences": EXPECTED_SENTENCES,
        "expected_gold_triples": EXPECTED_GOLD_TRIPLES,
        "ok": (
            len(sentences) == EXPECTED_SENTENCES
            and n_gold == EXPECTED_GOLD_TRIPLES
            and len(set(ids)) == len(ids)
        ),
    }
    return report
