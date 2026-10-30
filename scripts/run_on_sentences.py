"""Roda um sistema registrado sobre BIA, OIEC-PT ou BIOMED e grava no formato do benchmark.

Gera as entradas da análise de polaridade fora dos três corpora do
PT-OIE-Benchmark: ``outputs/external_benchmark/predictions/{native,benchmark_format}/
<sistema>/<corpus>``. As sentenças são as de ``polarity_fidelity.load_sentences``.

Uso:
    python scripts/run_on_sentences.py --system pt_oie --dataset BIOMED
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from polarity_fidelity import load_sentences  # noqa: E402
from src.benchmark.external.runner import run_extractor_on_dataset, system_config, write_predictions  # noqa: E402
from src.benchmark.registry import load_yaml  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True)
    p.add_argument("--dataset", required=True, choices=["BIA", "OIEC-PT", "BIOMED"])
    p.add_argument("--config", default="configs/external_benchmark.yaml")
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs/external_benchmark/predictions")
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    common = {"bosque_path": str(REPO / "data/pt_bosque-ud-train.conllu"), "seed": int(cfg.get("extractor_seed", 13))}
    sentences = load_sentences(args.dataset)
    result = run_extractor_on_dataset(args.system, args.dataset, sentences,
                                      system_config(args.system, REPO / "configs", common),
                                      log=lambda m: print(m, flush=True))
    paths = write_predictions(result, args.output_dir)
    st = result["stats"]
    print(f"{args.system}/{args.dataset}: {st['n_triples_tsv']} triplas, {st['n_failures']} falhas -> {paths['benchmark_format']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
