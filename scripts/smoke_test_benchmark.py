"""Smoke test: poucas sentenças, verificação apenas de formato e erros técnicos.

Escreve em outputs/benchmark/tmp/smoke para não contaminar a execução real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.registry import build_system, load_yaml, system_config_path  # noqa: E402
from src.benchmark.runner import run_system  # noqa: E402
from src.benchmark.schemas import read_jsonl, validate_prediction_dict  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["pt_oie", "ud_baseline", "ollama_gemma4"])
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--gold", default="data/bia_gold_sentences.jsonl")
    p.add_argument("--output-dir", default="outputs/benchmark/tmp/smoke")
    args = p.parse_args(argv)

    sentences = load_bia(args.gold)
    out = Path(args.output_dir)
    failures = 0
    for name in args.systems:
        cfg_path = system_config_path(REPO / "configs", name)
        scfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        print(f"\n=== smoke {name} ({args.limit} sentenças) ===")
        system = build_system(name, scfg)
        result = run_system(system, sentences, out, limit=args.limit)
        print(f"status={result.status} ok={result.n_ok} err={result.n_error} "
              f"triplas={result.n_triples} reason={result.reason}")
        if result.status in ("unavailable", "setup_error"):
            failures += 1
            continue
        norm = out / "normalized" / f"{name}.jsonl"
        if norm.exists():
            for row in read_jsonl(norm):
                problems = validate_prediction_dict(row)
                if problems:
                    print(f"  ESQUEMA INVÁLIDO: {problems}")
                    failures += 1
        if result.n_error:
            for row in read_jsonl(out / "errors" / f"{name}.jsonl"):
                print(f"  erro em {row['sentence_id']}: {row['error']}")
    print(f"\nsmoke test: {'OK' if failures == 0 else f'{failures} problema(s)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
