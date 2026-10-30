"""Sweep do limiar de atencao (tau).

O artigo reporta apenas tau=0.00 e tau=0.15, que produzem contagens
identicas. Este script varre uma faixa mais ampla e avalia cada ponto sob os
quatro protocolos do benchmark, mantendo tudo o mais constante (mesma
configuracao rq3_attn_on_thr0, mesmo corpus, mesma ordem, seed 13).

Uso:
    python -m scripts.attention_threshold_sweep \
        --taus 0.0 0.05 0.10 0.15 0.20 0.30 \
        --output outputs/benchmark/metrics/attention_threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import PROTOCOLS, evaluate_protocol  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--taus", nargs="+", type=float,
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument("--gold", default="data/bia_gold_sentences.jsonl")
    p.add_argument("--bosque", default="data/pt_bosque-ud-train.conllu")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output",
                   default="outputs/benchmark/metrics/attention_threshold_sweep.json")
    args = p.parse_args(argv)

    from src.extractor import Config, OpenIEExtractorBIA

    sentences = load_bia(args.gold)
    results: dict[str, dict] = {}

    for tau in args.taus:
        config = Config(
            bert_model="neuralmind/bert-base-portuguese-cased",
            bosque_path=args.bosque,
            seed=args.seed,
            cop_mode="full",
            theory_mode="off",
            apply_theoretical_rules=False,
            no_attn=False,
            attn_decision_enabled=True,
            attn_rerank_enabled=True,
            heads_mode="rank",
            top_k_heads=10,
            attn_threshold=tau,
        )
        extractor = OpenIEExtractorBIA(config=config, verbose=False)
        preds_by_sent = []
        for s in sentences:
            rows, _ = extractor.extract(s.sentence)
            preds_by_sent.append([
                {"arg1": str(r.get("arg1", "")), "rel": str(r.get("rel", "")),
                 "arg2": str(r.get("arg2", "")),
                 "triple_id": f"{s.sentence_id}_tau{tau}_{i:03d}",
                 "conf_att": r.get("conf_att")}
                for i, r in enumerate(rows)
            ])
        n_triples = sum(len(x) for x in preds_by_sent)
        n_scored = sum(1 for x in preds_by_sent for t in x
                       if t.get("conf_att") is not None)
        entry = {"n_triples": n_triples, "n_attention_scored": n_scored}
        print(f"\n=== tau = {tau:.2f}: {n_triples} triplas "
              f"({n_scored} com score de atencao) ===")
        for protocol in PROTOCOLS:
            r = evaluate_protocol(protocol, sentences, preds_by_sent)
            entry[protocol] = {
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                "precision": round(100 * r["precision"], 2),
                "recall": round(100 * r["recall"], 2),
                "f1": round(100 * r["f1"], 2),
            }
            print(f"  {protocol:>11}: P={100*r['precision']:6.2f} "
                  f"R={100*r['recall']:6.2f} F1={100*r['f1']:6.2f}")
        results[f"{tau:.2f}"] = entry
        del extractor

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nsalvo em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
