"""Executa o PT-OIE-EXTRACTOR sobre WIKI200, CETEN200 e Gamalho.

A configuração do extrator é a MESMA usada no artigo do BIA (rq3_attn_on_thr0),
via os adaptadores de ``src/benchmark/systems``. Nenhuma regra é ajustada para
estes corpora — eles são conjuntos de teste externos.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.external.runner import (  # noqa: E402
    run_extractor_on_dataset,
    system_config,
    write_predictions,
)
from src.benchmark.registry import load_yaml  # noqa: E402
from src.benchmark.runtime import collect_environment, git_state, now_iso, write_json  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/external_benchmark.yaml")
    p.add_argument("--systems", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="apenas as N primeiras sentenças (depuração)")
    p.add_argument("--force", action="store_true", help="reexecuta ignorando predições existentes")
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    bench_dir = REPO / cfg["benchmark_repo"]["clone_dir"]
    out_dir = Path(args.output_dir or (REPO / cfg["output_dir"])).resolve()
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    systems = args.systems or cfg["systems"]
    datasets = args.datasets or [d["name"] for d in cfg["datasets"]]
    extractor_seed = int(cfg.get("extractor_seed", 13))

    common_config = {
        "bosque_path": str(REPO / "data/pt_bosque-ud-train.conllu"),
        "seed": extractor_seed,
    }

    started_at = now_iso()
    run_log: dict = {
        "started_at": started_at,
        "config": str(args.config),
        "benchmark_commit": git_state(str(bench_dir)).get("commit"),
        "extractor_seed": extractor_seed,
        "environment": collect_environment(),
        "runs": {},
    }

    for system in systems:
        for name in datasets:
            tsv_path = pred_dir / "benchmark_format" / system / f"{name}_results.csv"
            if tsv_path.exists() and not args.force:
                print(f"[skip] {system}/{name}: predições já existem (use --force)")
                continue

            ds = load_dataset(name, bench_dir)
            sentences = ds.sentences[: args.limit] if args.limit else ds.sentences
            print(f"[run ] {system} em {name}: {len(sentences)} sentenças")

            scfg = system_config(system, REPO / "configs", common_config)
            result = run_extractor_on_dataset(
                system, name, sentences, scfg, log=lambda m: print(m, flush=True)
            )
            paths = write_predictions(result, pred_dir)
            st = result["stats"]
            print(f"       {st['n_triples_tsv']} triplas | "
                  f"{st['n_sentences_without_extraction']} sentenças sem extração | "
                  f"{st['n_failures']} falhas | {st['total_seconds']:.0f}s")

            run_log["runs"][f"{system}/{name}"] = {
                "stats": st,
                "metadata": result["metadata"],
                "paths": {k: str(v.relative_to(REPO)) for k, v in paths.items()},
                "failures": result["failures"],
            }

    run_log["finished_at"] = now_iso()
    write_json(log_dir / "run_log.json", run_log)
    print(f"\nlog -> {log_dir / 'run_log.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
