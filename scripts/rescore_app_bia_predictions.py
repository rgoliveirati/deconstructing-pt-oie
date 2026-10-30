"""Repontua no BIA as predições de ``outputs/bia_external_systems``.

Uma rodada comparativa em 12/09/2026 executou DetIE e ArgOE/LinguaKit no BIA
(ausentes de ``scripts/run_benchmark.py``) e reexecutou DptOIE, Multi²OIE e
PragmaticOIE, cada um com a implementação dos autores. Este script pontua essas
predições com os mesmos procedimentos dos demais sistemas, sem nova inferência:

* avaliador do PT-OIE-Benchmark (``Evaluation.py``, LREC 2026), sem modificação;
* os quatro protocolos internos (strict, tolerant, bia_legacy, carb_style).

Os dois mantêm todas as triplas. O avaliador lê as predições com ``pd.read_csv``
padrão, de modo que um campo vazio vira NaN, conta no total de predições e vale 0
naquele campo (``str(NaN)`` é a palavra ``nan``); ``to_frame`` reproduz essa leitura.

Antes de pontuar os sistemas novos, reproduz os números de referência a partir de
``outputs/benchmark/normalized`` e aborta se divergirem.

Duas conversões são aplicadas às predições do app:

* ``@`` vira espaço: o LinguaKit junta palavras de expressões multipalavra com
  ``@``, e ``normalize_words`` do avaliador apaga pontuação, fundindo-as numa
  palavra inexistente. As saídas do ArgOE distribuídas pelo benchmark não
  contêm ``@``;
* sentenças são casadas pelo texto, não pela posição.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import PROTOCOLS, evaluate_protocol  # noqa: E402
from src.benchmark.external.carb import assert_equivalent_to_benchmark_evaluator, score_dataset  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402
from src.benchmark.schemas import read_jsonl  # noqa: E402

# Predições da rodada comparativa de 2026-09-12 (DetIE, ArgOE/LinguaKit, DptOIE, Multi²OIE,
# PragmaticOIE no BIA), copiadas sem alteração.
DEFAULT_RUN = REPO / "outputs/bia_external_systems"
APP_SYSTEMS = ["DetIE", "ArgOE_LinguaKit", "DptOIE", "Multi2OIE", "PragmaticOIE"]

# Valores com todas as triplas mantidas (avaliador sem modificação).
EXPECTED_BENCHMARK_EVALUATOR = {"pt_oie": 59.6, "ud_baseline": 57.3, "ollama_gemma4": 52.5, "multi2oie": 51.5, "dptoie": 14.7}
EXPECTED_PROTOCOLS = {"pt_oie": {"strict": 18.37, "tolerant": 29.14, "bia_legacy": 54.92, "carb_style": 61.67}}


def complete(t: dict) -> bool:
    return all((t.get(k) or "").strip() for k in ("arg1", "rel", "arg2"))


def to_frame(sents, preds_by_sent) -> pd.DataFrame:
    """Todas as triplas, com campo vazio como NaN, como o avaliador lê um TSV."""
    rows = [{"Sentence": s.sentence, "ARG0": t["arg1"], "V": t["rel"], "ARG1": t["arg2"]}
            for s, preds in zip(sents, preds_by_sent) for t in preds]
    df = pd.DataFrame(rows, columns=["Sentence", "ARG0", "V", "ARG1"])
    return df.replace("", float("nan"))


def gold_frame(sents) -> pd.DataFrame:
    return pd.DataFrame([{"Sentence": s.sentence, "ARG0": g.arg1, "V": g.rel, "ARG1": g.arg2}
                         for s in sents for g in s.gold],
                        columns=["Sentence", "ARG0", "V", "ARG1"])


def from_normalized(path: Path, sents) -> list[list[dict]]:
    by = defaultdict(list)
    for r in read_jsonl(path):
        by[r["sentence_id"]].append({k: r.get(k) or "" for k in ("arg1", "rel", "arg2")})
    return [by.get(s.sentence_id, []) for s in sents]


def from_app(path: Path, sents) -> list[list[dict]]:
    index = {s.sentence: i for i, s in enumerate(sents)}
    out: list[list[dict]] = [[] for _ in sents]
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        for p in d["pred"]:
            out[index[d["sentence"]]].append(
                {k: (p.get(k) or "").replace("@", " ") for k in ("arg1", "rel", "arg2")})
    return out


def score(sents, gold_df, preds, bench_dir: str) -> dict:
    pred_df = to_frame(sents, preds)
    bench = score_dataset(gold_df, pred_df, bench_dir)
    bench.pop("_per_sentence")
    bench["equivalence_max_abs_diff"] = assert_equivalent_to_benchmark_evaluator(gold_df, pred_df, bench_dir)["max_abs_diff"]
    protocols = {p: evaluate_protocol(p, sents, preds)["f1"] * 100 for p in PROTOCOLS}
    return {
        "n_triples": sum(map(len, preds)),
        "n_incomplete": sum(not complete(t) for ps in preds for t in ps),
        "benchmark_evaluator": {"precision": bench["precision"] * 100, "recall": bench["recall"] * 100,
                          "f1": bench["f1_score"] * 100, "n_scored": bench["total_extractions"],
                          "equivalence_max_abs_diff": bench["equivalence_max_abs_diff"]},
        "protocols_f1": protocols,
    }


def misaligned(sents, preds) -> int:
    """Triplas com mais da metade das palavras fora da sentença a que foram atribuídas."""
    import re
    import unicodedata

    def words(s: str) -> set[str]:
        s = unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()
        return set(re.sub(r"[^\w\s]", "", s).split())

    n = 0
    for s, ps in zip(sents, preds):
        sw = words(s.sentence)
        for t in ps:
            tw = words(" ".join(t.values()))
            n += bool(tw) and len(tw - sw) / len(tw) > 0.5
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app-run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--output-dir", type=Path, default=REPO / "outputs/benchmark_app_rescore")
    args = p.parse_args(argv)

    bench_dir = str(REPO / ".external/PT-OIE-Benchmark")
    sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    gold_df = gold_frame(sents)
    result: dict = {"created_at": now_iso(), "app_run": str(args.app_run.resolve().relative_to(REPO)) if args.app_run.resolve().is_relative_to(REPO) else str(args.app_run), "reproduction": {}, "systems": {}}

    for name, expected in EXPECTED_BENCHMARK_EVALUATOR.items():
        preds = from_normalized(REPO / "outputs/benchmark/normalized" / f"{name}.jsonl", sents)
        r = score(sents, gold_df, preds, bench_dir)
        r["misaligned_triples"] = misaligned(sents, preds)
        got = round(r["benchmark_evaluator"]["f1"], 1)
        if got != expected:
            raise SystemExit(f"reprodução falhou: {name} PT-OIE-Benchmark F1 {got} != {expected}")
        for proto, value in EXPECTED_PROTOCOLS.get(name, {}).items():
            if round(r["protocols_f1"][proto], 2) != value:
                raise SystemExit(f"reprodução falhou: {name} {proto} {r['protocols_f1'][proto]:.2f} != {value}")
        result["reproduction"][name] = r
        print(f"reproduzido  {name:<14} PT-OIE-Benchmark F1={got:5.1f}  desalinhadas={r['misaligned_triples']}")

    for name in APP_SYSTEMS:
        preds = from_app(args.app_run / f"{name}.jsonl", sents)
        r = score(sents, gold_df, preds, bench_dir)
        r["misaligned_triples"] = misaligned(sents, preds)
        result["systems"][name] = r
        pf = r["protocols_f1"]
        print(f"app          {name:<14} PT-OIE-Benchmark F1={r['benchmark_evaluator']['f1']:5.1f}  "
              f"strict={pf['strict']:5.2f} tolerant={pf['tolerant']:5.2f} "
              f"bia_legacy={pf['bia_legacy']:5.2f} carb_style={pf['carb_style']:5.2f}  "
              f"desalinhadas={r['misaligned_triples']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "results.json", result)
    print(f"-> {args.output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
