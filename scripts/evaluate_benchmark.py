"""Avalia as predições salvas sob os 4 protocolos, com bootstrap pareado.

Consome apenas os arquivos em ``normalized/`` e ``raw/`` — pode ser
reexecutado sem repetir a extração (determinismo verificável).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.bootstrap import (  # noqa: E402
    bootstrap_metrics, make_indices, paired_f1_difference,
)
from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import PROTOCOLS, evaluate_protocol  # noqa: E402
from src.benchmark.reporting import runtime_metrics, write_csv  # noqa: E402
from src.benchmark.runtime import write_json  # noqa: E402
from src.benchmark.schemas import read_jsonl  # noqa: E402

PAIRED_COMPARISONS = [
    ("pt_oie", "ud_baseline"),
    ("pt_oie", "dptoie"),
    ("pt_oie", "multi2oie"),
    ("pt_oie", "portnoie"),
    ("pt_oie", "ollama_gemma4"),
    ("dptoie", "ollama_gemma4"),
    ("multi2oie", "ollama_gemma4"),
]


def load_predictions(norm_path: Path, sentences) -> list[list[dict]]:
    by_sent: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(norm_path):
        by_sent[row["sentence_id"]].append(row)
    return [by_sent.get(s.sentence_id, []) for s in sentences]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", default="outputs/benchmark/normalized")
    p.add_argument("--gold", default="data/bia_gold_sentences.jsonl")
    p.add_argument("--output-dir", default="outputs/benchmark")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    sentences = load_bia(args.gold)
    norm_dir = Path(args.predictions_dir)
    systems = sorted(p.stem for p in norm_dir.glob("*.jsonl"))
    if not systems:
        print("nenhuma predição encontrada em", norm_dir, file=sys.stderr)
        return 1
    print("sistemas avaliados:", systems)

    indices = make_indices(len(sentences), args.bootstrap_samples, args.seed)

    metrics_rows = []
    sentence_rows = []
    pattern_rows = []
    boot_results: dict[tuple[str, str], dict] = {}

    for system in systems:
        preds_by_sent = load_predictions(norm_dir / f"{system}.jsonl", sentences)
        norm_rows = read_jsonl(norm_dir / f"{system}.jsonl")
        pattern_by_triple = {
            r["triple_id"]: (r.get("metadata") or {}).get("pattern_ud")
            for r in norm_rows
        }
        for protocol in args.protocols:
            res = evaluate_protocol(protocol, sentences, preds_by_sent)
            match_dir = out / "matches" / protocol
            match_dir.mkdir(parents=True, exist_ok=True)
            write_json(match_dir / f"{system}.json", {
                k: res[k] for k in ("protocol", "tp", "fp", "fn", "precision",
                                     "recall", "f1", "matching", "criteria",
                                     "per_sentence")
            })
            boot = bootstrap_metrics(res["per_sentence"], indices)
            boot_results[(system, protocol)] = boot
            metrics_rows.append({
                "system": system,
                "protocol": protocol,
                "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
                "precision": round(res["precision"], 6),
                "recall": round(res["recall"], 6),
                "f1": round(res["f1"], 6),
                "n_dedup_removed": res.get("n_dedup_removed", 0),
                "f1_ci_low": round(boot["f1"]["low"], 6),
                "f1_ci_high": round(boot["f1"]["high"], 6),
            })
            for s in res["per_sentence"]:
                sentence_rows.append({
                    "system": system, "protocol": protocol,
                    "sentence_id": s["sentence_id"],
                    "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                    "n_pred": s["n_pred"], "n_gold": s["n_gold"],
                })
                # Análise por padrão UD: apenas quando o próprio sistema anota
                # pattern_ud (pt_oie, ud_baseline); não inferimos padrão para
                # saídas externas.
                for m in s.get("matches", []):
                    tid = (m.get("pred") or {}).get("triple_id")
                    pat = pattern_by_triple.get(tid)
                    if pat:
                        pattern_rows.append({
                            "system": system, "protocol": protocol,
                            "pattern_ud": pat, "outcome": "tp",
                        })
                for up in s.get("unmatched_pred", []):
                    pat = pattern_by_triple.get(up.get("triple_id"))
                    if pat:
                        pattern_rows.append({
                            "system": system, "protocol": protocol,
                            "pattern_ud": pat, "outcome": "fp",
                        })

    # Consolidação por padrão
    pat_counts: dict[tuple, dict] = {}
    for r in pattern_rows:
        key = (r["system"], r["protocol"], r["pattern_ud"])
        c = pat_counts.setdefault(key, {"tp": 0, "fp": 0})
        c[r["outcome"]] += 1
    pattern_table = [
        {
            "system": s, "protocol": pr, "pattern_ud": pat,
            "tp": c["tp"], "fp": c["fp"],
            "precision": round(c["tp"] / (c["tp"] + c["fp"]), 6)
            if (c["tp"] + c["fp"]) else 0.0,
        }
        for (s, pr, pat), c in sorted(pat_counts.items())
    ]

    # Sentença: análises adicionais (comprimento, nº de golds)
    feats = {
        s.sentence_id: {
            "sentence_len_tokens": len(s.sentence.split()),
            "n_gold": sum(1 for g in s.gold if g.valid),
        }
        for s in sentences
    }
    for r in sentence_rows:
        r.update(feats[r["sentence_id"]])

    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / "metrics_by_system_protocol.csv", metrics_rows)
    write_json(metrics_dir / "metrics_by_system_protocol.json", metrics_rows)
    write_csv(metrics_dir / "metrics_by_sentence.csv", sentence_rows)
    write_csv(metrics_dir / "metrics_by_pattern.csv", pattern_table)

    # Bootstrap: ICs e diferenças pareadas
    boot_dir = out / "bootstrap"
    boot_dir.mkdir(parents=True, exist_ok=True)
    ci_rows = []
    for (system, protocol), b in boot_results.items():
        for metric in ("precision", "recall", "f1"):
            ci_rows.append({
                "system": system, "protocol": protocol, "metric": metric,
                "point": round(b["point"][metric], 6),
                "boot_mean": round(b[metric]["mean"], 6),
                "ci_low": round(b[metric]["low"], 6),
                "ci_high": round(b[metric]["high"], 6),
                "n_samples": b["n_samples"], "n_sentences": b["n_sentences"],
            })
    write_csv(boot_dir / "confidence_intervals.csv", ci_rows)

    diff_rows = []
    dist_rows = []
    for a, b in PAIRED_COMPARISONS:
        for protocol in args.protocols:
            ra, rb = boot_results.get((a, protocol)), boot_results.get((b, protocol))
            if ra is None or rb is None:
                continue  # sistema indisponível: sem comparação, sem zero
            d = paired_f1_difference(ra, rb)
            diff_rows.append({
                "system_a": a, "system_b": b, "protocol": protocol,
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()},
            })
            dist_rows.append({
                "system_a": a, "system_b": b, "protocol": protocol,
                "delta_f1_percentiles": {
                    str(q): round(float(__import__("numpy").percentile(
                        ra["_f1_samples"] - rb["_f1_samples"], q)), 6)
                    for q in (1, 2.5, 5, 25, 50, 75, 95, 97.5, 99)
                },
            })
    write_csv(boot_dir / "paired_f1_differences.csv", diff_rows)
    with open(boot_dir / "bootstrap_distributions.jsonl", "w", encoding="utf-8") as fh:
        for r in dist_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Métricas de execução
    rt_rows = []
    ollama_rows = []
    raw_dir = out / "raw"
    for system in systems:
        raw_path = raw_dir / f"{system}.jsonl"
        if raw_path.exists():
            rt_rows.append(runtime_metrics(raw_path, system))
        if system == "ollama_gemma4" and raw_path.exists():
            for r in read_jsonl(raw_path):
                meta = {}
                if isinstance(r.get("raw_output"), dict):
                    meta = r["raw_output"].get("metadata", {})
                elif isinstance(r.get("raw_output"), list) and r["raw_output"]:
                    pass
                ollama_rows.append({
                    "sentence_id": r["sentence_id"],
                    "status": r["status"],
                    "n_triples": r.get("n_triples", 0),
                    "duration_seconds": (r.get("runtime") or {}).get("duration_seconds"),
                })
    rt_dir = out / "runtime"
    rt_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rt_dir / "runtime_metrics.csv", rt_rows)
    if ollama_rows:
        write_csv(rt_dir / "ollama_metrics.csv", ollama_rows)

    print(f"\nmétricas: {metrics_dir / 'metrics_by_system_protocol.csv'}")
    for r in metrics_rows:
        if r["protocol"] == "bia_legacy":
            print(f"  {r['system']:>14} bia_legacy: P={r['precision']:.4f} "
                  f"R={r['recall']:.4f} F1={r['f1']:.4f} "
                  f"IC95=[{r['f1_ci_low']:.4f}, {r['f1_ci_high']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
