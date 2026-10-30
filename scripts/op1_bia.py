"""Extensões do gerador (Oportunidade 1) medidas no BIA sob ``bia_legacy``.

Roda a configuração de referência (``rq3_attn_on_thr0``) com cada extensão ligada:
oblíquo regido (1a, com e sem oblíquo temporal único), sujeito nulo (1b, antecedente)
e passiva com agente (1c), e a combinação congelada 1a+1c (``pt_oie_op1``).

Uso:
    python scripts/op1_bia.py --output outputs/dev_oiec/op1_bia.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import evaluate_protocol  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402

VARIANTS = {
    "default": {},
    "1a_obl": {"obl_core_policy": "governed", "obl_allow_sole_temporal": True},
    "1a_obl_no_temporal": {"obl_core_policy": "governed"},
    "1b_null_subject": {"allow_null_subject": True, "null_subject_arg1_source": "antecedent"},
    "1c_passive": {"allow_passive_agent": True},
    "1a_1c_frozen": {"obl_core_policy": "governed", "obl_allow_sole_temporal": True, "allow_passive_agent": True},
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/dev_oiec/op1_bia.json")
    args = p.parse_args(argv)

    from src.extractor import Config, OpenIEExtractorBIA

    sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    result = {"created_at": now_iso(), "protocol": "bia_legacy", "variants": {}}
    for name, overrides in VARIANTS.items():
        ex = OpenIEExtractorBIA(config=Config(
            bert_model="neuralmind/bert-base-portuguese-cased",
            bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"),
            seed=13, cop_mode="full", theory_mode="off", apply_theoretical_rules=False,
            no_attn=False, attn_decision_enabled=True, attn_rerank_enabled=True,
            heads_mode="rank", top_k_heads=10, attn_threshold=0.0, **overrides,
        ), verbose=False)
        preds = [[{k: r.get(k) or "" for k in ("arg1", "rel", "arg2")} for r in ex.extract(s.sentence)[0]] for s in sents]
        ev = evaluate_protocol("bia_legacy", sents, preds)
        result["variants"][name] = {"overrides": overrides, "n_triples": sum(map(len, preds)),
                                    "precision": ev["precision"] * 100, "recall": ev["recall"] * 100,
                                    "f1": ev["f1"] * 100}
        v = result["variants"][name]
        print(f"{name:<20} triplas={v['n_triples']} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}", flush=True)
    base = result["variants"]["default"]["f1"]
    for v in result["variants"].values():
        v["delta_f1"] = v["f1"] - base
    write_json(args.output, result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
