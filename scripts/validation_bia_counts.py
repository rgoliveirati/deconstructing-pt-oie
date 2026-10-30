"""Violações da camada de validação na configuração de referência, no BIA.

Roda ``rq3_attn_on_thr0`` com a camada de validação ligada em modo ``annotate``
(a saída não muda) e conta, sobre as triplas finais, as violações críticas e os
avisos por regra. Roda também o modo ``filter`` e registra quantas triplas ele
remove.

Uso:
    python scripts/validation_bia_counts.py --output outputs/rule_audit/bia_reference_violations.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402


def run(mode: str, sents):
    from src.extractor import Config, OpenIEExtractorBIA

    ex = OpenIEExtractorBIA(config=Config(
        bert_model="neuralmind/bert-base-portuguese-cased",
        bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"),
        seed=13, cop_mode="full", theory_mode=mode, apply_theoretical_rules=True,
        no_attn=False, attn_decision_enabled=True, attn_rerank_enabled=True,
        heads_mode="rank", top_k_heads=10, attn_threshold=0.0,
    ), verbose=False)
    return [ex.extract(s.sentence)[0] for s in sents]


def split(field: str) -> list[str]:
    return [x for x in (field or "").split(";") if x]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/rule_audit/bia_reference_violations.json")
    args = p.parse_args(argv)

    sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    annotated = [r for rows in run("annotate", sents) for r in rows]
    filtered = sum(map(len, run("filter", sents)))
    critical = Counter(v for r in annotated for v in split(r.get("critical_violations")))
    warnings = Counter(v for r in annotated for v in split(r.get("warnings")))
    blocked = sum(bool(r.get("theory_blocked")) for r in annotated)
    result = {
        "created_at": now_iso(), "config": "rq3_attn_on_thr0", "corpus": "BIA",
        "n_triples_annotate": len(annotated), "n_triples_filter": filtered,
        "removed_by_filter": len(annotated) - filtered, "blocked_in_annotate": blocked,
        "critical_violations": dict(critical.most_common()), "warnings": dict(warnings.most_common()),
    }
    for k, v in result.items():
        print(f"{k}: {v}")
    write_json(args.output, result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
