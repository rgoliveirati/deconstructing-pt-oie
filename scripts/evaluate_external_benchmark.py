"""Avalia PT-OIE-EXTRACTOR e os sistemas do artigo com o avaliador do PT-OIE-Benchmark.

Para cada par (sistema, corpus) grava métricas e a decomposição por sentença
(insumo do bootstrap). Antes de usar a decomposição, prova numericamente que
ela reproduz ``Evaluation.evaluate`` dos autores.

Os sistemas de referência são pontuados a partir das saídas por sentença
distribuídas no repositório do PT-OIE-Benchmark — isto é reprodução local, não cópia do
artigo, e é marcado como tal em ``source: reproduced``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.external.carb import assert_equivalent_to_benchmark_evaluator, score_dataset  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.registry import load_yaml  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402


def predictions_path(out_dir: Path, bench_dir: Path, system: str, dataset: str) -> Path | None:
    """Nossas predições ficam em outputs/; as dos sistemas do artigo, no clone."""
    ours = out_dir / "predictions" / "benchmark_format" / system / f"{dataset}_results.csv"
    if ours.exists():
        return ours
    theirs = bench_dir / "System_Results" / system / f"{dataset}_results.csv"
    if theirs.exists():
        return theirs
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/external_benchmark.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--skip-equivalence", action="store_true")
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    bench_dir = REPO / cfg["benchmark_repo"]["clone_dir"]
    out_dir = Path(args.output_dir or (REPO / cfg["output_dir"]))
    eval_dir = out_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d["name"] for d in cfg["datasets"]]
    ours = list(cfg["systems"])
    theirs = list(cfg["reference_systems"])

    results: dict = {"created_at": now_iso(), "evaluator": "PT-OIE-Benchmark evaluator (Evaluation.py, sem modificação)",
                     "equivalence_checks": {}, "systems": {}}
    rows = []

    for system in ours + theirs:
        results["systems"][system] = {"source": "reproduced", "is_ours": system in ours, "datasets": {}}
        for name in datasets:
            ds = load_dataset(name, bench_dir)
            path = predictions_path(out_dir, bench_dir, system, name)
            if path is None:
                results["systems"][system]["datasets"][name] = {"status": "no_output"}
                rows.append({"system": system, "dataset": name, "status": "no_output",
                             "precision": None, "recall": None, "f1": None})
                print(f"  {system:<18} {name:<10} sem saída no repositório do PT-OIE-Benchmark")
                continue

            pred = pd.read_csv(path, sep="\t")
            score = score_dataset(ds.gold, pred, str(bench_dir))

            # Prova de equivalência com o avaliador do PT-OIE-Benchmark (uma vez por par).
            if not args.skip_equivalence:
                check = assert_equivalent_to_benchmark_evaluator(ds.gold, pred, str(bench_dir))
                results["equivalence_checks"][f"{system}/{name}"] = check

            per_sentence = score.pop("_per_sentence")
            ps_dir = eval_dir / "per_sentence" / system
            ps_dir.mkdir(parents=True, exist_ok=True)
            with (ps_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
                for s in per_sentence:
                    fh.write(json.dumps({k: v for k, v in s.items() if k != "key"},
                                        ensure_ascii=False) + "\n")

            score["predictions_path"] = str(path.relative_to(REPO))
            score["status"] = "ok"
            results["systems"][system]["datasets"][name] = score
            rows.append({
                "system": system, "dataset": name, "status": "ok",
                "precision": score["precision"] * 100,
                "recall": score["recall"] * 100,
                "f1": score["f1_score"] * 100,
                "n_pred": score["total_extractions"],
                "n_gold": score["gold_extractions"],
                "n_sentences": score["total_sentences"],
                "extractions_per_sentence": score["extractions_per_sentence"],
                "pct_sentences_without_extraction": score["pct_sentences_without_extraction"],
                "n_pred_orphan": score["n_pred_orphan"],
            })
            print(f"  {system:<18} {name:<10} P={score['precision']*100:5.1f} "
                  f"R={score['recall']*100:5.1f} F1={score['f1_score']*100:5.1f} "
                  f"n={score['total_extractions']}")

    write_json(eval_dir / "metrics.json", results)
    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "metrics_by_system_dataset.csv", index=False)

    n_checks = len(results["equivalence_checks"])
    if n_checks:
        worst = max(c["max_abs_diff"] for c in results["equivalence_checks"].values())
        print(f"\nequivalência com o avaliador do PT-OIE-Benchmark: {n_checks} pares verificados, "
              f"diferença absoluta máxima = {worst:.2e}")
    print(f"métricas -> {eval_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
