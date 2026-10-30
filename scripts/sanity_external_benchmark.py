"""Verificação de sanidade e análise de erros da avaliação externa (Etapa 6).

Checagens automáticas:
  1. cada sentença processada exatamente uma vez;
  2. alinhamento IDs ↔ sentenças ↔ gold ↔ predições;
  3. nenhuma predição órfã (sentença que não casa com o gold);
  4. duplicidades, saídas vazias e extrações anormalmente longas.

Amostras reprodutíveis (seed fixa) para inspeção manual: alta correspondência,
falsos positivos, falsos negativos e sentenças sem extração.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.external.carb import load_benchmark_evaluator  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.registry import load_yaml  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402

LONG_ARG_TOKENS = 25  # limiar para "extração anormalmente longa"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/external_benchmark.yaml")
    p.add_argument("--system", default="pt_oie")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    bench_dir = REPO / cfg["benchmark_repo"]["clone_dir"]
    out_dir = Path(args.output_dir or (REPO / cfg["output_dir"]))
    err_dir = out_dir / "error_analysis"
    err_dir.mkdir(parents=True, exist_ok=True)

    ev = load_benchmark_evaluator(str(bench_dir))
    n_samples = int(cfg["error_analysis"]["n_samples_per_bucket"])
    rng = np.random.default_rng(int(cfg["error_analysis"]["seed"]))

    checks: dict = {"created_at": now_iso(), "system": args.system, "ok": True, "datasets": {}}
    samples: dict = {}

    for meta in cfg["datasets"]:
        name = meta["name"]
        ds = load_dataset(name, bench_dir)
        native_path = out_dir / "predictions" / "native" / args.system / f"{name}.jsonl"
        tsv_path = out_dir / "predictions" / "benchmark_format" / args.system / f"{name}_results.csv"
        native = [json.loads(l) for l in native_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        pred = pd.read_csv(tsv_path, sep="\t")

        c: dict = {}
        # 1. cada sentença uma única vez, na ordem original
        ids = [r["sentence_id"] for r in native]
        c["n_native_rows"] = len(native)
        c["ids_unique"] = len(set(ids)) == len(ids)
        c["count_matches_corpus"] = len(native) == ds.n_sentences
        c["order_preserved"] = [r["sentence"] for r in native] == ds.sentences

        # 2/3. alinhamento e predições órfãs
        key = lambda s: frozenset(ev.normalize_words(s))  # noqa: E731
        gold_keys = {key(s) for s in ds.gold["Sentence"]}
        pred_keys = [key(s) for s in pred["Sentence"]]
        c["n_pred_orphan"] = int(sum(1 for k in pred_keys if k not in gold_keys))
        c["gold_sentences_covered_by_corpus"] = int(sum(1 for k in gold_keys if k in {key(s) for s in ds.sentences}))
        c["n_gold_sentences"] = len(gold_keys)

        # 4. duplicidades, vazios, extrações longas
        dup = pred.duplicated(subset=["Sentence", "ARG0", "V", "ARG1"]).sum()
        c["n_duplicate_triples"] = int(dup)
        c["n_empty_slot_rows"] = int(
            ((pred.ARG0.astype(str).str.strip() == "")
             | (pred.V.astype(str).str.strip() == "")
             | (pred.ARG1.astype(str).str.strip() == "")).sum()
        )
        tok = lambda s: len(str(s).split())  # noqa: E731
        c["n_long_extractions"] = int(
            ((pred.ARG0.map(tok) > LONG_ARG_TOKENS) | (pred.ARG1.map(tok) > LONG_ARG_TOKENS)).sum()
        )
        c["max_arg_tokens"] = int(max(pred.ARG0.map(tok).max(), pred.ARG1.map(tok).max()))
        c["n_sentences_without_extraction"] = int(sum(1 for r in native if r["n_triples"] == 0))

        c["ok"] = bool(
            c["ids_unique"] and c["count_matches_corpus"] and c["order_preserved"]
            and c["n_pred_orphan"] == 0 and c["n_empty_slot_rows"] == 0
        )
        if not c["ok"]:
            checks["ok"] = False
        checks["datasets"][name] = c

        # ---- amostras reprodutíveis para inspeção manual ----
        gold_by_key: dict = {}
        for _, r in ds.gold.iterrows():
            gold_by_key.setdefault(key(r.Sentence), []).append((r.ARG0, r.V, r.ARG1))

        scored = []
        for _, r in pred.iterrows():
            k = key(r.Sentence)
            gts = gold_by_key.get(k, [])
            best, best_g = 0.0, None
            for g in gts:
                s = ev.get_tuple_match_score((r.ARG0, r.V, r.ARG1), g)
                if s > best:
                    best, best_g = s, g
            scored.append({"sentence": r.Sentence, "pred": [r.ARG0, r.V, r.ARG1],
                           "best_gold": list(best_g) if best_g else None, "score": best})

        # falsos negativos: gold cuja melhor predição pontua baixo
        fn = []
        pred_by_key: dict = {}
        for _, r in pred.iterrows():
            pred_by_key.setdefault(key(r.Sentence), []).append((r.ARG0, r.V, r.ARG1))
        for k, gts in gold_by_key.items():
            pts = pred_by_key.get(k, [])
            for g in gts:
                best = max((ev.get_tuple_match_score(pt, g) for pt in pts), default=0.0)
                if best < 0.3:
                    fn.append({"gold": list(g), "best_pred_score": best,
                               "n_pred_for_sentence": len(pts)})

        def pick(pool, n):
            if not pool:
                return []
            idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            return [pool[int(i)] for i in sorted(idx)]

        high = [s for s in scored if s["score"] >= 0.8]
        fp = [s for s in scored if s["score"] < 0.3]
        no_ext = [{"sentence_id": r["sentence_id"], "sentence": r["sentence"]}
                  for r in native if r["n_triples"] == 0]

        samples[name] = {
            "alta_correspondencia": pick(high, n_samples),
            "falsos_positivos": pick(fp, n_samples),
            "falsos_negativos": pick(fn, n_samples),
            "sentencas_sem_extracao": pick(no_ext, n_samples),
            "counts": {"alta_correspondencia": len(high), "falsos_positivos": len(fp),
                       "falsos_negativos": len(fn), "sentencas_sem_extracao": len(no_ext)},
        }

    write_json(err_dir / f"sanity_checks_{args.system}.json", checks)
    write_json(err_dir / f"samples_{args.system}.json", samples)

    print(f"sanidade ({args.system}): {'OK' if checks['ok'] else 'FALHOU'}")
    for name, c in checks["datasets"].items():
        print(f"  {name:<10} ordem={c['order_preserved']} ids_unicos={c['ids_unique']} "
              f"orfas={c['n_pred_orphan']} duplicadas={c['n_duplicate_triples']} "
              f"longas={c['n_long_extractions']} (max {c['max_arg_tokens']} tokens) "
              f"sem_extracao={c['n_sentences_without_extraction']}")
    print(f"\nchecagens -> {err_dir / f'sanity_checks_{args.system}.json'}")
    print(f"amostras  -> {err_dir / f'samples_{args.system}.json'}")
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
