"""ICs no BIA e piso trivial sob o avaliador do PT-OIE-Benchmark.

(1) Para os oito sistemas do BIA (os quatro de ``outputs/benchmark/normalized`` e os
quatro de ``outputs/bia_external_systems``, ver ``rescore_app_bia_predictions.py``): F1 com IC95
sob o avaliador do PT-OIE-Benchmark e sob os quatro protocolos, todos com todas as
triplas (ver ``to_frame``), e Δ pareado contra o PT-OIE-EXTRACTOR. Bootstrap por
sentença, 10.000 réplicas, seed 42, mesma matriz de índices para todos.
(2) Baseline trivial verb-split (proposto por um revisor de uma versão anterior deste trabalho), variantes VERB e
VERB+AUX, pontuado pelo avaliador do PT-OIE-Benchmark no BIA e nos três corpora do
benchmark.

Uso:
    python scripts/bia_cis_and_trivial_floor.py outputs/benchmark_app_rescore/cis_and_trivial_floor.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from types import SimpleNamespace  # noqa: E402

import stanza  # noqa: E402

from rescore_app_bia_predictions import DEFAULT_RUN, from_app, from_normalized, gold_frame, to_frame  # noqa: E402
from src.benchmark.bootstrap import bootstrap_metrics, make_indices, paired_f1_difference  # noqa: E402
from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import PROTOCOLS, evaluate_protocol  # noqa: E402
from src.benchmark.external.bootstrap import bootstrap_carb, paired_delta  # noqa: E402
from src.benchmark.external.carb import score_dataset  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402

OUT = Path(sys.argv[1])
Sentence = lambda text: SimpleNamespace(sentence=text)  # noqa: E731
BENCH = str(REPO / ".external/PT-OIE-Benchmark")
sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
gold_df = gold_frame(sents)
idx = make_indices(len(sents))

systems = {n: from_normalized(REPO / f"outputs/benchmark/normalized/{n}.jsonl", sents)
           for n in ("pt_oie", "ud_baseline", "ollama_gemma4", "multi2oie")}
for n in ("DetIE", "ArgOE_LinguaKit", "DptOIE", "PragmaticOIE"):
    systems[n] = from_app(DEFAULT_RUN / f"{n}.jsonl", sents)

nlp = stanza.Pipeline("pt", processors="tokenize,mwt,pos", tokenize_no_ssplit=True, verbose=False)


def trivial(text_list, allowed):
    out = []
    for text in text_list:
        words = [w for s in nlp(text).sentences for w in s.words]
        out.append([{"arg1": " ".join(x.text for x in words[:i]), "rel": w.text,
                     "arg2": " ".join(x.text for x in words[i + 1:])}
                    for i, w in enumerate(words) if w.upos in allowed])
    return out


VARIANTS = {"verb": {"VERB"}, "verb_aux": {"VERB", "AUX"}}
for variant, allowed in VARIANTS.items():
    systems[f"trivial_{variant}"] = trivial([s.sentence for s in sents], allowed)

res = {"bia": {}, "floor_external": {}}
boot_bench, boot_prot = {}, {p: {} for p in PROTOCOLS}
for name, preds in systems.items():
    sc = score_dataset(gold_df, to_frame(sents, preds), BENCH)
    boot_bench[name] = bootstrap_carb(sc["_per_sentence"], idx, sc.get("n_pred_orphan", 0))
    entry = {"benchmark_evaluator": {k: boot_bench[name][k] for k in ("point", "f1")}, "protocols": {}}
    for p in PROTOCOLS:
        r = evaluate_protocol(p, sents, preds)
        boot_prot[p][name] = bootstrap_metrics(r["per_sentence"], idx)
        entry["protocols"][p] = {"point": boot_prot[p][name]["point"], "f1": boot_prot[p][name]["f1"]}
    res["bia"][name] = entry
    print(name, round(boot_bench[name]["point"]["f1"] * 100, 1),
          {p: round(entry["protocols"][p]["point"]["f1"] * 100, 2) for p in PROTOCOLS}, flush=True)

res["paired_vs_pt_oie"] = {}
for name in systems:
    if name == "pt_oie":
        continue
    d = {"benchmark_evaluator": paired_delta(boot_bench["pt_oie"], boot_bench[name], "pt_oie", name)}
    for p in PROTOCOLS:
        x = paired_f1_difference(boot_prot[p]["pt_oie"], boot_prot[p][name])
        d[p] = {"delta_point_f1": x["delta_point"], "ci_low": x["ci_low"], "ci_high": x["ci_high"],
                "p_delta_gt_0": x["prop_delta_gt_0"]}
    res["paired_vs_pt_oie"][name] = d

for ds in ("WIKI200", "CETEN200", "Gamalho"):
    data = load_dataset(ds, BENCH)
    for variant, allowed in VARIANTS.items():
        preds = trivial(data.sentences, allowed)
        sc = score_dataset(data.gold, to_frame([Sentence(s) for s in data.sentences], preds), BENCH)
        res["floor_external"][f"{ds}/{variant}"] = {k: sc[k] for k in ("precision", "recall", "f1_score", "total_extractions")}
        print("trivial", ds, variant, round(sc["f1_score"] * 100, 1), flush=True)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    return o


OUT.write_text(json.dumps(strip(res), indent=2, default=float))
for n, d in res["paired_vs_pt_oie"].items():
    b = d["benchmark_evaluator"]
    print(f"pt_oie - {n:<18} bench Δ={b['delta_point_f1']*100:6.2f} [{b['ci_low']*100:6.2f},{b['ci_high']*100:6.2f}]  "
          f"legacy Δ={d['bia_legacy']['delta_point_f1']*100:6.2f} [{d['bia_legacy']['ci_low']*100:6.2f},{d['bia_legacy']['ci_high']*100:6.2f}]")
