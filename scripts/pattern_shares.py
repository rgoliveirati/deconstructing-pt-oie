"""Participação de cada padrão UD nas triplas do ``rq3_attn_on_thr0`` no BIA.

Lê ``outputs/benchmark/normalized/pt_oie.jsonl`` (campo ``metadata.pattern_ud``).

Uso:
    python scripts/pattern_shares.py --output outputs/benchmark/metrics/pattern_shares.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.runtime import write_json  # noqa: E402
from src.benchmark.schemas import read_jsonl  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=REPO / "outputs/benchmark/normalized/pt_oie.jsonl")
    p.add_argument("--output", type=Path, default=REPO / "outputs/benchmark/metrics/pattern_shares.json")
    args = p.parse_args(argv)

    counts = Counter((r.get("metadata") or {}).get("pattern_ud", "unknown") for r in read_jsonl(args.predictions))
    n = sum(counts.values())
    result = {"n_triples": n, "patterns": {k: {"n": c, "pct": 100 * c / n} for k, c in counts.most_common()}}
    for k, v in result["patterns"].items():
        print(f"{k:<18} {v['n']:4d} {v['pct']:5.1f}")
    write_json(args.output, result)
    print(f"{n} triplas -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
