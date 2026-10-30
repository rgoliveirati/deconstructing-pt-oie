"""Tabelas de métricas, métricas de execução e tabela LaTeX."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from .schemas import read_jsonl


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter=";")
        w.writeheader()
        w.writerows(rows)


def runtime_metrics(raw_path: str | Path, system: str) -> dict[str, Any]:
    rows = read_jsonl(raw_path)
    durations = [
        r["runtime"].get("duration_seconds", 0.0)
        for r in rows
        if isinstance(r.get("runtime"), dict)
    ]
    n_err = sum(1 for r in rows if r["status"] != "ok")
    n_triples = sum(r.get("n_triples", 0) for r in rows)
    per_sent = [r.get("n_triples", 0) for r in rows if r["status"] == "ok"]
    out: dict[str, Any] = {
        "system": system,
        "n_sentences": len(rows),
        "n_sentences_error": n_err,
        "failure_rate": n_err / len(rows) if rows else 0.0,
        "n_triples": n_triples,
        "triples_per_sentence_mean": statistics.mean(per_sent) if per_sent else 0.0,
        "triples_per_sentence_median": statistics.median(per_sent) if per_sent else 0.0,
        "total_seconds": sum(durations),
        "mean_seconds": statistics.mean(durations) if durations else 0.0,
        "median_seconds": statistics.median(durations) if durations else 0.0,
        "p95_seconds": (
            sorted(durations)[max(0, int(0.95 * len(durations)) - 1)] if durations else 0.0
        ),
        "sentences_per_second": (
            len(durations) / sum(durations) if durations and sum(durations) > 0 else 0.0
        ),
    }
    return out


def _fmt_pct(x: float | None) -> str:
    return f"{100 * x:.2f}" if x is not None else "--"


def latex_table(
    rows: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    *,
    gemma_digest: str | None,
) -> str:
    """`rows`: um dict por sistema disponível, com métricas por protocolo e IC."""
    best_f1 = max((r["bia_legacy_f1"] for r in rows), default=None)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"System & P\textsubscript{legacy} & R\textsubscript{legacy} & "
        r"F\textsubscript{1,legacy} & 95\% CI (F\textsubscript{1}) & "
        r"F\textsubscript{1,strict} & F\textsubscript{1,tol} & "
        r"F\textsubscript{1,CaRB} & Median s/sent & Fail\% & Supervised \\",
        r"\midrule",
    ]
    for r in rows:
        f1 = _fmt_pct(r["bia_legacy_f1"])
        if best_f1 is not None and r["bia_legacy_f1"] == best_f1:
            f1 = r"\textbf{" + f1 + "}"
        ci = f"[{_fmt_pct(r['f1_ci_low'])}, {_fmt_pct(r['f1_ci_high'])}]"
        lines.append(
            f"{r['display_name']} & {_fmt_pct(r['bia_legacy_p'])} & "
            f"{_fmt_pct(r['bia_legacy_r'])} & {f1} & {ci} & "
            f"{_fmt_pct(r['strict_f1'])} & {_fmt_pct(r['tolerant_f1'])} & "
            f"{_fmt_pct(r['carb_f1'])} & {r['median_seconds']:.2f} & "
            f"{100 * r['failure_rate']:.1f} & {r['supervised']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    notes = [
        "P/R/F$_1$ under the \\texttt{bia\\_legacy} protocol (the project's legacy "
        "matcher); protocols are not mutually comparable.",
    ]
    if gemma_digest:
        notes.append(
            f"Gemma~4 executed locally via Ollama, zero-shot, model tag "
            f"\\texttt{{gemma4:latest}}, digest \\texttt{{{gemma_digest[:12]}}}."
        )
    for u in unavailable:
        notes.append(
            f"\\texttt{{{u['system']}}} unavailable: {u['reason']} "
            "(not scored; unavailability is not zero)."
        )
    lines.append(r"\caption{Comparative benchmark on the BIA corpus "
                 r"(262 sentences, 427 gold triples). " + " ".join(notes) + "}")
    lines += [r"\label{tab:benchmark}", r"\end{table*}"]
    return "\n".join(lines)
