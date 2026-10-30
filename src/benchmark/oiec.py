"""Corpus OIEC-PT (Souza et al., 2025) — conjunto de DESENVOLVIMENTO.

Souza, Cabral, Claro e Salvador (2025), *Challenges in Expanding Portuguese
Resources: A View from Open Information Extraction* (arXiv:2501.11851).
Construído sobre as 300 primeiras sentenças da seção portuguesa do PUD
(Parallel Universal Dependencies).

Formato dos arquivos distribuídos (TSV, CRLF)::

    <id>\t<sentença>
    <arg1>\t<rel>\t<arg2>\t<f1>\t<f2>\t<f3>
    ...

**Semântica das flags — determinada e verificada, não suposta.** Os arquivos
contêm TODOS os candidatos julgados pelos anotadores (2.667), a maioria dos
quais foi *rejeitada*. A segunda flag é o marcador de validade. Verificação
independente contra a Tabela 2 do artigo:

===========  ===================  ==================
Subconjunto  ``f2 == 1`` (aqui)   Tabela 2 (artigo)
===========  ===================  ==================
GOLD (100)   136                  136
SILVER (200) 337                  337
Total (300)  473                  473
===========  ===================  ==================

``scripts/validate_oiec.py`` reexecuta essa verificação.

Consequências para o uso:

* as triplas VÁLIDAS são 1,58 por sentença — o corpus é *esparso*, não denso;
* as 2.194 triplas inválidas são **negativos rotulados por humanos**, úteis para
  calibrar padrões novos (um padrão que passa a gerar candidatos que os juízes
  rejeitaram está errado, mesmo que o F1 suba).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Índice (0-based) da flag de validade dentro das três colunas finais.
VALIDITY_FLAG_INDEX = 1

#: Estatísticas da Tabela 2 do artigo, usadas como verificação de integridade.
PAPER_TABLE2 = {
    "OIEC-PT-GOLD.txt": {"sentences": 100, "valid": 136},
    "OIEC-PT-SILVER.txt": {"sentences": 200, "valid": 337},
    "OIEC-PT.txt": {"sentences": 300, "valid": 473},
}

_ID_LINE = re.compile(r"^\d+$")


@dataclass
class OIECTriple:
    arg1: str
    rel: str
    arg2: str
    flags: tuple[str, str, str]

    @property
    def valid(self) -> bool:
        return self.flags[VALIDITY_FLAG_INDEX] == "1"


@dataclass
class OIECSentence:
    sentence_id: str
    sentence: str
    triples: list[OIECTriple] = field(default_factory=list)

    @property
    def gold(self) -> list[OIECTriple]:
        """Apenas as triplas julgadas válidas pelos anotadores."""
        return [t for t in self.triples if t.valid]

    @property
    def rejected(self) -> list[OIECTriple]:
        """Candidatos que os juízes humanos rejeitaram (negativos rotulados)."""
        return [t for t in self.triples if not t.valid]


def load_oiec(path: str | Path) -> list[OIECSentence]:
    """Lê um arquivo do OIEC-PT preservando a ordem original das sentenças."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sentences: list[OIECSentence] = []
    current: OIECSentence | None = None

    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 2 and _ID_LINE.fullmatch(parts[0].strip()):
            current = OIECSentence(sentence_id=parts[0].strip(), sentence=parts[1].strip())
            sentences.append(current)
            continue
        if len(parts) >= 6 and current is not None:
            current.triples.append(
                OIECTriple(
                    arg1=parts[0].strip(),
                    rel=parts[1].strip(),
                    arg2=parts[2].strip(),
                    flags=(parts[3].strip(), parts[4].strip(), parts[5].strip()),
                )
            )
    return sentences


def validate_oiec(path: str | Path) -> dict[str, Any]:
    """Confere as contagens contra a Tabela 2 do artigo (prova da semântica da flag)."""
    p = Path(path)
    sents = load_oiec(p)
    n_valid = sum(len(s.gold) for s in sents)
    n_total = sum(len(s.triples) for s in sents)
    expected = PAPER_TABLE2.get(p.name)
    report: dict[str, Any] = {
        "file": p.name,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "n_sentences": len(sents),
        "n_candidate_triples": n_total,
        "n_valid_triples": n_valid,
        "n_rejected_triples": n_total - n_valid,
        "valid_per_sentence": n_valid / len(sents) if sents else 0.0,
        "candidates_per_sentence": n_total / len(sents) if sents else 0.0,
        "validity_flag_index": VALIDITY_FLAG_INDEX,
    }
    if expected:
        report["expected_sentences"] = expected["sentences"]
        report["expected_valid"] = expected["valid"]
        report["ok"] = (len(sents) == expected["sentences"] and n_valid == expected["valid"])
    else:
        report["ok"] = None
    return report
