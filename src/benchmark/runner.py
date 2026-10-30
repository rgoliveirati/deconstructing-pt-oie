"""Orquestração da execução do benchmark.

Para cada sistema: setup -> extração sentença a sentença (na mesma ordem do
corpus para todos os sistemas) -> teardown. Saída bruta por sentença em
``raw/<system>.jsonl``, triplas normalizadas em ``normalized/<system>.jsonl``,
erros em ``errors/<system>.jsonl``. A retomada (``--resume``) pula sentenças
já registradas no arquivo bruto. Falha de um sistema não interrompe os demais.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from .corpus import Sentence
from .runtime import now_iso
from .schemas import Prediction, SentenceRecord, append_jsonl, read_jsonl, validate_prediction_dict
from .systems.base import OpenIESystem, SystemUnavailable, Timer


class RunResult:
    def __init__(self, system: str):
        self.system = system
        self.status = "pending"
        self.reason: str | None = None
        self.evidence: str | None = None
        self.n_ok = 0
        self.n_error = 0
        self.n_triples = 0
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _done_sentence_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    return {r["sentence_id"] for r in read_jsonl(raw_path)}


def run_system(
    system: OpenIESystem,
    sentences: list[Sentence],
    output_dir: str | Path,
    *,
    resume: bool = False,
    fail_fast: bool = False,
    limit: int | None = None,
) -> RunResult:
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw" / f"{system.name}.jsonl"
    norm_path = output_dir / "normalized" / f"{system.name}.jsonl"
    err_path = output_dir / "errors" / f"{system.name}.jsonl"
    for p in (raw_path, norm_path, err_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    result = RunResult(system.name)
    result.started_at = now_iso()

    if not resume:
        for p in (raw_path, norm_path, err_path):
            p.unlink(missing_ok=True)
    done = _done_sentence_ids(raw_path) if resume else set()

    try:
        system.setup()
    except SystemUnavailable as exc:
        result.status = "unavailable"
        result.reason = exc.reason
        result.evidence = exc.evidence
        result.finished_at = now_iso()
        return result
    except Exception as exc:
        result.status = "setup_error"
        result.reason = f"{type(exc).__name__}: {exc}"
        result.evidence = traceback.format_exc()[-2000:]
        result.finished_at = now_iso()
        return result

    todo = sentences[:limit] if limit else sentences

    if system.supports_batch:
        pending = [s for s in todo if s.sentence_id not in done]
        batch_result: dict[str, list[Prediction]] = {}
        batch_error: str | None = None
        with Timer() as bt:
            if pending:
                try:
                    batch_result = system.extract_batch(
                        [(s.sentence_id, s.sentence) for s in pending]
                    )
                except Exception as exc:
                    batch_error = f"{type(exc).__name__}: {exc}"
        # Tempo amortizado: o custo de um processo em lote é rateado por
        # sentença e sinalizado como tal na saída.
        share = bt.duration_seconds / len(pending) if pending else 0.0
        runtime = {
            "started_at": bt.started_at,
            "finished_at": bt.finished_at,
            "duration_seconds": round(share, 6),
            "amortized_from_batch": True,
            "batch_duration_seconds": round(bt.duration_seconds, 6),
            "batch_size": len(pending),
        }
        for sent in todo:
            if sent.sentence_id in done:
                rec_prev = next(
                    r for r in read_jsonl(raw_path)
                    if r["sentence_id"] == sent.sentence_id
                )
                result.n_ok += 1 if rec_prev["status"] == "ok" else 0
                result.n_error += 1 if rec_prev["status"] != "ok" else 0
                result.n_triples += rec_prev.get("n_triples", 0)
                continue
            preds = batch_result.get(sent.sentence_id)
            if batch_error is not None or preds is None:
                error = (
                    batch_error
                    or getattr(system, "batch_errors", {}).get(sent.sentence_id)
                    or "sentença ausente do resultado em lote"
                )
                rec = SentenceRecord(
                    sentence_id=sent.sentence_id, sentence=sent.sentence,
                    system=system.name, status="error", n_triples=0,
                    error=error, runtime=runtime,
                )
                append_jsonl(raw_path, rec)
                append_jsonl(err_path, rec)
                result.n_error += 1
                continue
            for p in preds:
                p.runtime = runtime
                row = p.to_dict()
                problems = validate_prediction_dict(row)
                if problems:
                    raise RuntimeError(
                        f"predição fora do esquema ({system.name}/{sent.sentence_id}): {problems}"
                    )
                append_jsonl(norm_path, row)
            rec = SentenceRecord(
                sentence_id=sent.sentence_id, sentence=sent.sentence,
                system=system.name, status="ok", n_triples=len(preds),
                raw_output=[p.raw_output for p in preds] or None,
                runtime=runtime,
            )
            append_jsonl(raw_path, rec)
            result.n_ok += 1
            result.n_triples += len(preds)
        try:
            result.metadata = system.metadata()
        except Exception as exc:
            result.metadata = {"metadata_error": str(exc)}
        finally:
            system.teardown()
        result.status = "ok" if result.n_error == 0 else "ok_with_errors"
        result.finished_at = now_iso()
        return result

    for sent in todo:
        if sent.sentence_id in done:
            rec_prev = next(
                r for r in read_jsonl(raw_path) if r["sentence_id"] == sent.sentence_id
            )
            result.n_ok += 1 if rec_prev["status"] == "ok" else 0
            result.n_error += 1 if rec_prev["status"] != "ok" else 0
            result.n_triples += rec_prev.get("n_triples", 0)
            continue
        with Timer() as t:
            try:
                preds = system.extract(sent.sentence_id, sent.sentence)
                error = None
                status = "ok"
            except Exception as exc:
                preds = None
                status = "timeout" if "timeout" in type(exc).__name__.lower() else "error"
                error = f"{type(exc).__name__}: {exc}"
        runtime = t.runtime()

        if preds is not None:
            for p in preds:
                p.runtime = runtime
                row = p.to_dict()
                problems = validate_prediction_dict(row)
                if problems:
                    raise RuntimeError(
                        f"predição fora do esquema ({system.name}/{sent.sentence_id}): {problems}"
                    )
                append_jsonl(norm_path, row)
            raw_out = [p.raw_output for p in preds] or getattr(
                system, "last_empty_meta", None
            )
            rec = SentenceRecord(
                sentence_id=sent.sentence_id,
                sentence=sent.sentence,
                system=system.name,
                status="ok",
                n_triples=len(preds),
                raw_output=raw_out,
                runtime=runtime,
            )
            append_jsonl(raw_path, rec)
            result.n_ok += 1
            result.n_triples += len(preds)
        else:
            rec = SentenceRecord(
                sentence_id=sent.sentence_id,
                sentence=sent.sentence,
                system=system.name,
                status=status,
                n_triples=0,
                error=error,
                runtime=runtime,
            )
            append_jsonl(raw_path, rec)
            append_jsonl(err_path, rec)
            result.n_error += 1
            if fail_fast:
                break

    try:
        result.metadata = system.metadata()
    except Exception as exc:
        result.metadata = {"metadata_error": str(exc)}
    finally:
        system.teardown()

    result.status = "ok" if result.n_error == 0 else "ok_with_errors"
    result.finished_at = now_iso()
    return result
