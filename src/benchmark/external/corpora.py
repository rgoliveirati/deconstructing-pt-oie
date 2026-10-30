"""Carga e validação dos três corpora do benchmark externo.

Os arquivos NÃO são recriados aqui: são lidos exatamente como distribuídos no
repositório do PT-OIE-Benchmark (``Gold/Sentences/*.txt`` e ``Gold/Results/*_gold.csv``).
Nenhuma anotação gold é corrigida, mesmo quando visivelmente incorreta —
o benchmark original é preservado.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def _relative(path: Path) -> str:
    """Caminho relativo à raiz do repositório, quando estiver dentro dela."""
    p = Path(path).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(path)

#: Contagens declaradas na Tabela 2 do artigo (sentenças, extrações gold).
DATASETS: dict[str, dict[str, Any]] = {
    "WIKI200": {"domain": "Wikipédia", "expected_sentences": 201, "expected_gold": 427},
    "CETEN200": {"domain": "Notícias (CETENFolha)", "expected_sentences": 202, "expected_gold": 366},
    "Gamalho": {"domain": "Científico (ecologia)", "expected_sentences": 103, "expected_gold": 358},
}

GOLD_COLUMNS = ["Sentence", "ARG0", "V", "ARG1"]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class ExternalDataset:
    name: str
    domain: str
    sentences: list[str]
    gold: pd.DataFrame
    sentences_path: Path
    gold_path: Path
    sentences_sha256: str
    gold_sha256: str
    warnings: list[str] = field(default_factory=list)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_gold(self) -> int:
        return len(self.gold)

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "sentences_path": _relative(self.sentences_path),
            "gold_path": _relative(self.gold_path),
            "sentences_sha256": self.sentences_sha256,
            "gold_sha256": self.gold_sha256,
            "n_sentences": self.n_sentences,
            "n_gold_triples": self.n_gold,
            "expected_sentences": DATASETS[self.name]["expected_sentences"],
            "expected_gold": DATASETS[self.name]["expected_gold"],
            "warnings": self.warnings,
        }


def load_dataset(name: str, bench_dir: str | Path) -> ExternalDataset:
    """Lê um corpus do clone do PT-OIE-Benchmark. Não converte nem altera o gold."""
    if name not in DATASETS:
        raise KeyError(f"corpus desconhecido: {name!r}")
    bench = Path(bench_dir)
    sent_path = bench / "Gold" / "Sentences" / f"{name}_Sentences.txt"
    gold_path = bench / "Gold" / "Results" / f"{name}_gold.csv"
    for p in (sent_path, gold_path):
        if not p.exists():
            raise FileNotFoundError(
                f"arquivo do benchmark ausente: {p}. "
                "Rode scripts/fetch_external_benchmark.py primeiro."
            )

    # Preserva a ordem original; ignora apenas linhas em branco terminais.
    raw = sent_path.read_text(encoding="utf-8").splitlines()
    sentences = [ln for ln in raw if ln.strip()]

    gold = pd.read_csv(gold_path, sep="\t")
    missing = [c for c in GOLD_COLUMNS if c not in gold.columns]
    if missing:
        raise ValueError(f"{gold_path}: colunas ausentes {missing}")

    warnings: list[str] = []
    meta = DATASETS[name]
    if len(sentences) != meta["expected_sentences"]:
        warnings.append(
            f"n_sentences={len(sentences)} difere do declarado no artigo "
            f"({meta['expected_sentences']})"
        )
    if len(gold) != meta["expected_gold"]:
        warnings.append(
            f"n_gold={len(gold)} difere do declarado no artigo ({meta['expected_gold']})"
        )

    return ExternalDataset(
        name=name,
        domain=meta["domain"],
        sentences=sentences,
        gold=gold,
        sentences_path=sent_path,
        gold_path=gold_path,
        sentences_sha256=sha256_file(sent_path),
        gold_sha256=sha256_file(gold_path),
        warnings=warnings,
    )


def validate_datasets(bench_dir: str | Path, names: list[str] | None = None) -> dict[str, Any]:
    """Relatório de validação dos corpora, no espírito de ``corpus.validate_corpus``."""
    names = names or list(DATASETS)
    report: dict[str, Any] = {"ok": True, "datasets": {}}
    for name in names:
        ds = load_dataset(name, bench_dir)
        prov = ds.provenance()
        prov["ok"] = not ds.warnings
        if ds.warnings:
            report["ok"] = False
        report["datasets"][name] = prov
    totals_sent = sum(d["n_sentences"] for d in report["datasets"].values())
    totals_gold = sum(d["n_gold_triples"] for d in report["datasets"].values())
    report["total_sentences"] = totals_sent
    report["total_gold_triples"] = totals_gold
    return report
