"""Execução do PT-OIE-EXTRACTOR sobre um corpus externo.

Reusa os adaptadores já existentes (``src/benchmark/systems``) — as regras do
extrator não são tocadas e a configuração é a mesma do artigo do BIA.
Produz duas saídas por corpus:

* nativa (JSONL): todos os campos do extrator, sentenças sem extração, falhas e tempos;
* benchmark (TSV ``Sentence/ARG0/V/ARG1``): formato exigido pelo avaliador, com todas
  as triplas; um campo vazio fica vazio, e o avaliador o lê como NaN.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from ..registry import build_system, load_yaml, system_config_path


def _clean(value: Any) -> str:
    """TSV de uma linha por tripla: sem tabs, sem quebras, espaços colapsados."""
    return " ".join(str(value).replace("\t", " ").replace("\n", " ").split())


def sentence_id(dataset: str, index: int) -> str:
    """Identificador único e estável, derivado do corpus e da posição original."""
    return f"{dataset}_{index:04d}"


def system_config(name: str, configs_dir: Any, extra: dict[str, Any]) -> dict[str, Any]:
    """Config do sistema: o YAML de ``configs/systems/<nome>.yaml`` (quando existe)
    mais os campos comuns do experimento.

    Sistemas como ``ollama_gemma4`` dependem inteiramente do YAML (modelo, prompt,
    parâmetros de geração); os adaptadores do PT-OIE usam apenas ``bosque_path``
    e ``seed``. Mesclar os dois cobre ambos sem ramificação por sistema.
    """
    path = system_config_path(configs_dir, name)
    cfg = load_yaml(path) if Path(path).exists() else {}
    cfg.update(extra)
    return cfg


def run_extractor_on_dataset(
    system_name: str,
    dataset_name: str,
    sentences: list[str],
    config: dict[str, Any],
    *,
    progress_every: int = 50,
    log: Any = None,
) -> dict[str, Any]:
    """Roda um sistema sobre as sentenças, preservando a ordem original."""
    system = build_system(system_name, config)
    t_setup = time.perf_counter()
    system.setup()
    setup_seconds = time.perf_counter() - t_setup

    native: list[dict[str, Any]] = []
    tsv_rows: list[tuple[str, str, str, str]] = []
    failures: list[dict[str, Any]] = []
    per_sentence_seconds: list[float] = []
    n_empty_slot = 0

    t0 = time.perf_counter()
    for i, sentence in enumerate(sentences):
        sid = sentence_id(dataset_name, i)
        t_s = time.perf_counter()
        try:
            preds = system.extract(sid, sentence)
            error = None
        except Exception as exc:  # falha isolada não interrompe o experimento
            preds = []
            error = {"type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc(limit=5)}
            failures.append({"sentence_id": sid, "sentence": sentence, **error})
        elapsed = time.perf_counter() - t_s
        per_sentence_seconds.append(elapsed)

        kept = []
        for p in preds:
            a1, rel, a2 = _clean(p.arg1), _clean(p.rel), _clean(p.arg2)
            kept.append({
                "arg1": p.arg1, "rel": p.rel, "arg2": p.arg2,
                "triple_id": p.triple_id, "confidence": p.confidence,
                "pattern_ud": (p.metadata or {}).get("pattern_ud"),
            })
            tsv_rows.append((sentence, a1, rel, a2))
            n_empty_slot += not (a1 and rel and a2)

        native.append({
            "sentence_id": sid, "index": i, "dataset": dataset_name,
            "sentence": sentence, "n_triples": len(kept),
            "seconds": elapsed, "error": error, "triples": kept,
        })
        if progress_every and (i + 1) % progress_every == 0 and log:
            log(f"  {dataset_name}: {i + 1}/{len(sentences)} ({time.perf_counter() - t0:.0f}s)")

    total_seconds = time.perf_counter() - t0
    system.teardown()

    n_without = sum(1 for r in native if r["n_triples"] == 0)
    return {
        "system": system_name,
        "dataset": dataset_name,
        "native": native,
        "tsv_rows": tsv_rows,
        "failures": failures,
        "metadata": system.metadata(),
        "stats": {
            "n_sentences": len(sentences),
            "n_triples_native": sum(r["n_triples"] for r in native),
            "n_triples_tsv": len(tsv_rows),
            "n_empty_slot": n_empty_slot,
            "n_sentences_without_extraction": n_without,
            "pct_sentences_without_extraction": n_without / len(sentences) * 100 if sentences else 0.0,
            "n_failures": len(failures),
            "pct_failures": len(failures) / len(sentences) * 100 if sentences else 0.0,
            "setup_seconds": setup_seconds,
            "total_seconds": total_seconds,
            "mean_seconds_per_sentence": statistics.mean(per_sentence_seconds) if per_sentence_seconds else 0.0,
            "median_seconds_per_sentence": statistics.median(per_sentence_seconds) if per_sentence_seconds else 0.0,
        },
    }


def write_predictions(result: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Grava saída nativa (JSONL) e do benchmark (TSV)."""
    system, dataset = result["system"], result["dataset"]
    native_dir = out_dir / "native" / system
    tsv_dir = out_dir / "benchmark_format" / system
    native_dir.mkdir(parents=True, exist_ok=True)
    tsv_dir.mkdir(parents=True, exist_ok=True)

    native_path = native_dir / f"{dataset}.jsonl"
    with native_path.open("w", encoding="utf-8") as fh:
        for row in result["native"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    tsv_path = tsv_dir / f"{dataset}_results.csv"
    with tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Sentence", "ARG0", "V", "ARG1"])
        writer.writerows(result["tsv_rows"])

    return {"native": native_path, "benchmark_format": tsv_path}
