"""Controles da atenção: o que o reranking por atenção decide de fato?

Todas as variantes usam a configuração do ``rq3_attn_on_thr0`` (UD + reranking,
camada de validação desligada) e mudam só as cabeças ou a regra de escolha dentro
de cada grupo de competição (sentença, verbo, padrão):

    ranked       cabeças ranqueadas no UD Bosque (a configuração do artigo)
    placeholder  as 10 primeiras cabeças de cada janela (o que um Bosque vazio produz)
    random_<k>   10 cabeças sorteadas dentro de cada janela
    all_window   todas as cabeças da janela
    keep_first   escore constante: fica o primeiro candidato gerado
    keep_min     fica o candidato de MENOR conf_att (atenção invertida)

No BIA roda todas as variantes e o ``rq1_baseline_ud_puro``, com bootstrap pareado
sobre o ``bia_legacy``. Nos corpora do PT-OIE-Benchmark roda ``ranked``,
``keep_first`` e ``keep_min``. Para cada corpus registra os grupos com mais de um
candidato e por que o escore empata.

Uso:
    python scripts/attention_controls.py --output-dir outputs/attention_controls
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.bootstrap import bootstrap_metrics, make_indices, paired_f1_difference  # noqa: E402
from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import evaluate_protocol  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402
from src.extractor import Config, OpenIEExtractorBIA  # noqa: E402

EXTERNAL = ("WIKI200", "CETEN200", "Gamalho")


def window_heads(win):
    return [(layer, head) for layer in range(win[0], win[1] + 1) for head in range(1, 13)]


class Runner:
    def __init__(self, ext: OpenIEExtractorBIA):
        self.ext = ext
        self.score = ext._rerank_score
        self.rerank = ext._apply_attention_rerank
        self.ranked = (list(ext.heads_s), list(ext.heads_o))
        self.groups: list[dict] = []

    def _logging_rerank(self, rows):
        ext = self.ext
        groups: dict = {}
        for r in rows:
            if ext._pattern_uses_attention_rerank(r.get("pattern_ud", "unknown")):
                groups.setdefault(ext._rerank_group_key(r), []).append(r)
        for g in groups.values():
            scores = [ext._rerank_score(r) for r in g]
            top = max(scores)
            self.groups.append({
                "size": len(g),
                "tie_at_top": sum(s == top for s in scores) > 1,
                "first_is_top": scores[0] == top,
                "same_head_tokens": len({(r.get("subj_id"), r.get("obj_id")) for r in g}) == 1,
                "missing_score": any(r.get("conf_att") is None for r in g),
                "pattern": g[0].get("pattern_ud"),
            })
        return self.rerank(rows)

    def set_variant(self, name: str, cfg: Config) -> None:
        ext = self.ext
        ext.heads_s, ext.heads_o = self.ranked
        ext._rerank_score = self.score
        if name == "placeholder":
            ext.heads_s, ext.heads_o = window_heads(cfg.window_s)[:10], window_heads(cfg.window_o)[:10]
        elif name.startswith("random_"):
            rng = random.Random(1000 + int(name.split("_")[1]))
            ext.heads_s = sorted(rng.sample(window_heads(cfg.window_s), 10))
            ext.heads_o = sorted(rng.sample(window_heads(cfg.window_o), 10))
        elif name == "all_window":
            ext.heads_s, ext.heads_o = window_heads(cfg.window_s), window_heads(cfg.window_o)
        elif name == "keep_first":
            ext._rerank_score = lambda row: 0.0
        elif name == "keep_min":
            ext._rerank_score = lambda row: -self.score(row)

    def run(self, sentences: list[str], log: bool = False) -> list[list[tuple]]:
        self.ext._apply_attention_rerank = self._logging_rerank if log else self.rerank
        out = []
        for s in sentences:
            rows, _ = self.ext.extract(s)
            out.append([(r.get("arg1") or "", r.get("rel") or "", r.get("arg2") or "") for r in rows])
        self.ext._apply_attention_rerank = self.rerank
        return out


def group_summary(groups: list[dict]) -> dict:
    multi = [g for g in groups if g["size"] > 1]
    return {
        "n_groups": len(groups),
        "n_competing_groups": len(multi),
        "candidates_in_competing_groups": sum(g["size"] for g in multi),
        "competing_with_tie_at_top": sum(g["tie_at_top"] for g in multi),
        "competing_first_is_top": sum(g["first_is_top"] for g in multi),
        "competing_same_head_tokens": sum(g["same_head_tokens"] for g in multi),
        "competing_missing_score": sum(g["missing_score"] for g in multi),
        "competing_by_pattern": dict(Counter(g["pattern"] for g in multi)),
    }


def as_dicts(preds):
    return [[{"arg1": a, "rel": r, "arg2": b} for a, r, b in ps] for ps in preds]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs/attention_controls")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)

    common = dict(bert_model="neuralmind/bert-base-portuguese-cased",
                  bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"), seed=args.seed,
                  cop_mode="full", theory_mode="off", apply_theoretical_rules=False)
    cfg = Config(**common, no_attn=False, attn_decision_enabled=True, attn_rerank_enabled=True,
                 heads_mode="rank", top_k_heads=10, attn_threshold=0.0)
    random.seed(args.seed)
    runner = Runner(OpenIEExtractorBIA(config=cfg, verbose=False))
    result: dict = {"created_at": now_iso(), "config": "rq3_attn_on_thr0", "seed": args.seed,
                    "heads_ranked": {"subj": runner.ranked[0], "obj": runner.ranked[1]}, "corpora": {}}

    bia = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    corpora = {"BIA": [s.sentence for s in bia]}
    corpora.update({name: load_dataset(name, REPO / ".external/PT-OIE-Benchmark").sentences for name in EXTERNAL})
    variants_bia = ["ranked", "placeholder"] + [f"random_{k}" for k in range(5)] + ["all_window", "keep_first", "keep_min"]

    bia_preds = {}
    for corpus, sentences in corpora.items():
        variants = variants_bia if corpus == "BIA" else ["ranked", "keep_first", "keep_min"]
        runner.groups = []
        preds = {}
        for v in variants:
            runner.set_variant(v, cfg)
            preds[v] = runner.run(sentences, log=(v == "ranked"))
            same = sum(a == b for a, b in zip(preds[v], preds["ranked"]))
            print(f"{corpus:<9} {v:<12} triples={sum(map(len, preds[v])):4d} "
                  f"sentences identical to ranked={same}/{len(sentences)}", flush=True)
        result["corpora"][corpus] = {
            "n_sentences": len(sentences),
            "groups": group_summary(runner.groups),
            "variants": {v: {"n_triples": sum(map(len, ps)),
                             "sentences_identical_to_ranked": sum(a == b for a, b in zip(ps, preds["ranked"]))}
                         for v, ps in preds.items()},
        }
        if corpus == "BIA":
            bia_preds = preds

    # BIA: métricas e delta pareado contra o RQ1 sob bia_legacy.
    runner.set_variant("ranked", cfg)
    ext1 = OpenIEExtractorBIA(config=Config(**common, no_attn=True), verbose=False)
    rq1 = [[(r.get("arg1") or "", r.get("rel") or "", r.get("arg2") or "") for r in ext1.extract(s)[0]]
           for s in corpora["BIA"]]
    idx = make_indices(len(bia))
    evals = {v: evaluate_protocol("bia_legacy", bia, as_dicts(ps)) for v, ps in {**bia_preds, "rq1": rq1}.items()}
    boots = {v: bootstrap_metrics(e["per_sentence"], idx) for v, e in evals.items()}
    for v, e in evals.items():
        d = paired_f1_difference(boots[v], boots["rq1"])
        result["corpora"]["BIA"].setdefault("bia_legacy", {})[v] = {
            "n_triples": sum(map(len, ({**bia_preds, "rq1": rq1})[v])),
            "precision": e["precision"] * 100, "recall": e["recall"] * 100, "f1": e["f1"] * 100,
            "delta_vs_rq1": {"point": d["delta_point"] * 100, "ci_low": d["ci_low"] * 100,
                             "ci_high": d["ci_high"] * 100, "p_gt0": d["prop_delta_gt_0"]},
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "attention_controls.json", result)
    for corpus, r in result["corpora"].items():
        print(corpus, json.dumps(r["groups"]))
    print(f"-> {args.output_dir / 'attention_controls.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
