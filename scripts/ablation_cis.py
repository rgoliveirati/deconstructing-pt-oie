"""IC95 pareado de cada configuração da ablação contra o ``rq1_baseline_ud_puro``.

Lê as predições salvas por ``src.run_all_ablations`` (um diretório ``abl_<label>``
por configuração; ``outputs/ablation`` guarda só sentença e triplas de cada rótulo), agrupa os rótulos cuja saída é idêntica tripla a tripla e, para
cada saída distinta, calcula F1 sob um protocolo (``bia_legacy`` por padrão) com bootstrap pareado por
sentença (10.000 réplicas, seed 42, mesma matriz de índices). Nenhuma inferência nova.

Uso:
    python scripts/ablation_cis.py --output outputs/ablation_cis.json
    python scripts/ablation_cis.py --protocol strict --output outputs/ablation_cis_strict.json
    python scripts/ablation_cis.py --protocol tolerant --output outputs/ablation_cis_tolerant.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.bootstrap import bootstrap_metrics, make_indices, paired_f1_difference  # noqa: E402
from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import PROTOCOLS, evaluate_protocol  # noqa: E402
from src.benchmark.runtime import write_json  # noqa: E402
from src.run_all_ablations import BASELINE_LABEL, build_ablation_specs  # noqa: E402


def load_predictions(path: Path, sents) -> list[list[dict]]:
    by_sentence = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_sentence[row["sentence"]] = [{k: p.get(k) or "" for k in ("arg1", "rel", "arg2")} for p in row["pred"]]
    return [by_sentence[s.sentence] for s in sents]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation-dir", type=Path, default=REPO / "outputs/ablation")
    p.add_argument("--output", type=Path, default=REPO / "outputs/ablation_cis.json")
    p.add_argument("--protocol", choices=PROTOCOLS, default="bia_legacy")
    args = p.parse_args(argv)

    sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    labels = [s["label"] for s in build_ablation_specs(bert_model="", bosque_path="", seed=13)]
    def path(lab: str) -> Path:
        # ``outputs/ablation`` guarda a versão enxuta; uma rodada nova de run_all_ablations grava a completa.
        d = args.ablation_dir / f"abl_{lab}"
        slim = d / f"bia_abl_{lab}_predictions.jsonl"
        return slim if slim.exists() else d / f"bia_abl_{lab}_predictions_enriched.jsonl"

    preds = {lab: load_predictions(path(lab), sents) for lab in labels}

    # Agrupa rótulos com saída idêntica; o representante é o primeiro na ordem do script.
    groups: dict[str, list[str]] = {}
    for lab in labels:
        rep = next((r for r in groups if preds[r] == preds[lab]), None)
        groups.setdefault(rep or lab, [])
        if rep:
            groups[rep].append(lab)

    idx = make_indices(len(sents))
    boots, rows = {}, {}
    for rep in groups:
        ev = evaluate_protocol(args.protocol, sents, preds[rep])
        boots[rep] = bootstrap_metrics(ev["per_sentence"], idx)
        rows[rep] = {"identical_labels": groups[rep], "n_triples": sum(map(len, preds[rep])),
                     "tp": ev["tp"], "fp": ev["fp"], "fn": ev["fn"],
                     "precision": ev["precision"] * 100, "recall": ev["recall"] * 100, "f1": ev["f1"] * 100,
                     "f1_ci": [boots[rep]["f1"]["low"] * 100, boots[rep]["f1"]["high"] * 100]}
    for rep in groups:
        d = paired_f1_difference(boots[rep], boots[BASELINE_LABEL])
        rows[rep]["delta_vs_rq1"] = {"point": d["delta_point"] * 100, "ci_low": d["ci_low"] * 100,
                                     "ci_high": d["ci_high"] * 100, "p_gt0": d["prop_delta_gt_0"]}
        r = rows[rep]
        print(f"{rep:<28} F1={r['f1']:5.2f}  Δ={r['delta_vs_rq1']['point']:+5.2f} "
              f"[{r['delta_vs_rq1']['ci_low']:+5.2f}, {r['delta_vs_rq1']['ci_high']:+5.2f}]  "
              f"+{len(r['identical_labels'])} idênticos")
    write_json(args.output, {"ablation_dir": str(args.ablation_dir.resolve().relative_to(REPO)) if args.ablation_dir.resolve().is_relative_to(REPO) else str(args.ablation_dir), "protocol": args.protocol,
                             "n_labels": len(labels), "n_distinct_outputs": len(groups), "outputs": rows})
    print(f"{len(labels)} rótulos, {len(groups)} saídas distintas -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
