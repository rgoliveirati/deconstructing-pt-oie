"""IC bootstrap pareado para o delta da extracao copular (RQ2 vs RQ1)."""
import json, sys
sys.path.insert(0, ".")
from collections import defaultdict
from src.benchmark.corpus import load_bia
from src.benchmark.evaluation import evaluate_protocol
from src.benchmark.bootstrap import bootstrap_metrics, make_indices, paired_f1_difference
from src.benchmark.schemas import read_jsonl
from src.extractor import Config, OpenIEExtractorBIA

sentences = load_bia("data/bia_gold_sentences.jsonl")
BASE = dict(bert_model="neuralmind/bert-base-portuguese-cased",
            bosque_path="data/pt_bosque-ud-train.conllu", seed=13,
            theory_mode="off", apply_theoretical_rules=False, no_attn=True)

configs = {
    "rq2_cop_restricted": dict(BASE, cop_mode="restricted", extract_cop=True),
    "rq2_cop_off": dict(BASE, cop_mode="off", extract_cop=False),
}
preds_all = {}
# RQ1 ja executado: reaproveita as predicoes salvas do ud_baseline
by = defaultdict(list)
for r in read_jsonl("outputs/benchmark/normalized/ud_baseline.jsonl"):
    by[r["sentence_id"]].append(r)
preds_all["rq1_baseline_ud_puro"] = [by.get(s.sentence_id, []) for s in sentences]

for label, cfg in configs.items():
    ex = OpenIEExtractorBIA(config=Config(**cfg), verbose=False)
    out = []
    for s in sentences:
        rows, _ = ex.extract(s.sentence)
        out.append([{"arg1": str(r.get("arg1","")), "rel": str(r.get("rel","")),
                     "arg2": str(r.get("arg2","")),
                     "triple_id": f"{s.sentence_id}_{label}_{i:03d}"}
                    for i, r in enumerate(rows)])
    preds_all[label] = out
    del ex

idx = make_indices(len(sentences), 10000, 42)
boots, points = {}, {}
for label, preds in preds_all.items():
    r = evaluate_protocol("bia_legacy", sentences, preds)
    points[label] = round(100*r["f1"], 2)
    boots[label] = bootstrap_metrics(r["per_sentence"], idx)
    ci = boots[label]["f1"]
    print(f"{label:>24}: F1={100*r['f1']:.2f}  IC95=[{100*ci['low']:.2f}, {100*ci['high']:.2f}]")

print()
rows = []
for label in ("rq2_cop_restricted", "rq2_cop_off"):
    d = paired_f1_difference(boots[label], boots["rq1_baseline_ud_puro"])
    rows.append({"comparison": f"{label} - rq1_baseline_ud_puro", "protocol": "bia_legacy",
                 **{k: (round(100*v,2) if k.startswith("delta") or k.startswith("ci") else v)
                    for k,v in d.items()}})
    print(f"{label} vs RQ1: delta={100*d['delta_point']:+.2f} "
          f"IC95=[{100*d['ci_low']:+.2f}, {100*d['ci_high']:+.2f}] "
          f"P(delta<0)={d['prop_delta_lt_0']:.4f}")

json.dump({"point_f1": points, "paired": rows},
          open("outputs/benchmark/bootstrap/copular_delta_ci.json","w"),
          ensure_ascii=False, indent=2)
print("\nsalvo em outputs/benchmark/bootstrap/copular_delta_ci.json")
