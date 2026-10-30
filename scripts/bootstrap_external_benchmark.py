"""IC95 por bootstrap (reamostragem por sentença) e comparações pareadas.

As saídas por sentença dos sistemas do artigo estão no repositório do PT-OIE-Benchmark, o
que torna a comparação PAREADA possível — não apenas o confronto de agregados.
Todos os sistemas de um mesmo corpus compartilham a matriz de índices, de modo
que as diferenças são calculadas nas mesmas réplicas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.external.bootstrap import bootstrap_carb, make_indices, paired_delta  # noqa: E402
from src.benchmark.registry import load_yaml  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402


def load_per_sentence(eval_dir: Path, system: str, dataset: str) -> list[dict] | None:
    path = eval_dir / "per_sentence" / system / f"{dataset}.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/external_benchmark.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    out_dir = Path(args.output_dir or (REPO / cfg["output_dir"]))
    eval_dir = out_dir / "evaluations"
    boot_dir = out_dir / "bootstrap"
    boot_dir.mkdir(parents=True, exist_ok=True)

    n_samples = args.n_samples or int(cfg["bootstrap"]["n_samples"])
    seed = args.seed if args.seed is not None else int(cfg["bootstrap"]["seed"])
    datasets = [d["name"] for d in cfg["datasets"]]
    ours = list(cfg["systems"])
    all_systems = ours + list(cfg["reference_systems"])

    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))

    ci_rows, paired_rows = [], []
    summary: dict = {
        "created_at": now_iso(),
        "n_samples": n_samples,
        "seed": seed,
        "resampling_unit": "sentença",
        "ci_method": "percentil 2.5/97.5",
        "paired": "matriz de índices compartilhada por corpus",
        "datasets": {},
    }

    for dataset in datasets:
        results: dict[str, dict] = {}
        indices = None
        for system in all_systems:
            ps = load_per_sentence(eval_dir, system, dataset)
            if ps is None:
                continue
            if indices is None:
                indices = make_indices(len(ps), n_samples=n_samples, seed=seed)
            elif indices.shape[1] != len(ps):
                raise ValueError(
                    f"{system}/{dataset}: {len(ps)} sentenças, esperado {indices.shape[1]}"
                )
            orphan = metrics["systems"][system]["datasets"][dataset].get("n_pred_orphan", 0)
            results[system] = bootstrap_carb(ps, indices, n_pred_orphan=orphan)

        summary["datasets"][dataset] = {
            "n_sentences": int(indices.shape[1]) if indices is not None else 0,
            "systems_evaluated": sorted(results),
        }

        for system, res in results.items():
            ci_rows.append({
                "dataset": dataset, "system": system,
                "precision": res["point"]["precision"] * 100,
                "recall": res["point"]["recall"] * 100,
                "f1": res["point"]["f1"] * 100,
                "f1_ci_low": res["f1"]["low"] * 100,
                "f1_ci_high": res["f1"]["high"] * 100,
                "precision_ci_low": res["precision"]["low"] * 100,
                "precision_ci_high": res["precision"]["high"] * 100,
                "recall_ci_low": res["recall"]["low"] * 100,
                "recall_ci_high": res["recall"]["high"] * 100,
            })

        # Cada sistema nosso contra cada sistema do artigo, e a configuração de
        # referência (o primeiro de ``systems``) também contra os nossos, nas mesmas réplicas.
        for mine in ours:
            if mine not in results:
                continue
            for other, res_b in results.items():
                if other == mine or (other in ours and mine != ours[0]):
                    continue
                d = paired_delta(results[mine], res_b, mine, other)
                d["dataset"] = dataset
                paired_rows.append(d)
                # escala percentual para leitura
                for k in ("delta_point_f1", "delta_mean", "ci_low", "ci_high"):
                    d[k] = d[k] * 100

    pd.DataFrame(ci_rows).to_csv(boot_dir / "confidence_intervals.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(boot_dir / "paired_f1_differences.csv", index=False)
    write_json(boot_dir / "bootstrap_summary.json", summary)

    print(f"{'corpus':<10} {'sistema':<18} {'F1':>6}  IC95")
    for r in ci_rows:
        print(f"{r['dataset']:<10} {r['system']:<18} {r['f1']:6.1f}  "
              f"[{r['f1_ci_low']:5.1f}, {r['f1_ci_high']:5.1f}]")
    print(f"\nIC -> {boot_dir / 'confidence_intervals.csv'}")
    print(f"pareado -> {boot_dir / 'paired_f1_differences.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
