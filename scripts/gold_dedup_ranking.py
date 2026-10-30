"""Efeito da redundância da referência no ranking dos sistemas (BIA e Gamalho).

Duas triplas de referência da mesma sentença são variantes da mesma proposição
quando a sequência de palavras de uma (arg1 + rel + arg2) é subsequência da sequência
da outra e o arg1 de uma é subsequência do arg1 da outra; isso cobre fronteira
deslocada (``é | afiliada a X`` vs ``é afiliada a | X``) e expansão (``Berlim`` vs
``Berlim com um amigo``), respeita a ordem (as triplas copulares invertidas do
WIKI200, ``X é Y`` e ``Y é X``, não são variantes) e exclui orações encaixadas
(``o sufixo indica X`` dentro de ``outros apontaram que o sufixo indica X``).
A deduplicação é gulosa: percorre as triplas numa ordem e descarta as que são
variantes de uma já mantida, sem encadear proposições distintas por meio de uma
tripla maior. Três ordens, para que o resultado não dependa da escolha: a da
anotação, da mais curta para a mais longa e da mais longa para a mais curta.

Pontua todos os sistemas com o avaliador do PT-OIE-Benchmark na referência original
e nas deduplicadas, e registra o F1, a posição de cada sistema e o Δ pareado
(bootstrap por sentença, 10.000 réplicas, seed 42) dos pares que trocam de posição.
Predições lidas como o avaliador as lê (todas as triplas, campo vazio como NaN).
Normalização de palavras: a de ``match_profile.py``.

Uso:
    python scripts/gold_dedup_ranking.py --output outputs/external_benchmark/gold_dedup_ranking.json
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd  # noqa: E402

from match_profile import SYSTEMS as GAMALHO_SYSTEMS, words  # noqa: E402
from rescore_app_bia_predictions import DEFAULT_RUN, from_app, from_normalized, gold_frame, to_frame  # noqa: E402
from src.benchmark.bootstrap import make_indices  # noqa: E402
from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.external.bootstrap import bootstrap_carb, paired_delta  # noqa: E402
from src.benchmark.external.carb import score_dataset  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.runtime import write_json  # noqa: E402

BENCH = REPO / ".external/PT-OIE-Benchmark"
SLOTS = ("ARG0", "V", "ARG1")
KEEP = ("first", "shortest", "longest")


def subseq(a: list[str], b: list[str]) -> bool:
    it = iter(b)
    return all(x in it for x in a)


def variants(x: tuple, y: tuple) -> bool:
    """``x`` e ``y`` exprimem a mesma proposição (ver docstring do módulo)."""
    sx, sy = words(" ".join(x)), words(" ".join(y))
    ax, ay = words(x[0]), words(y[0])
    return (subseq(sx, sy) and subseq(ax, ay)) or (subseq(sy, sx) and subseq(ay, ax))


def dedup(gold: pd.DataFrame, keep: str) -> tuple[pd.DataFrame, float]:
    """Referência deduplicada e fração de sentenças com pelo menos um par de variantes."""
    out, redundant, n = [], 0, 0
    for _, g in gold.groupby("Sentence", sort=False):
        rows = [tuple(str(r[s]) for s in SLOTS) for _, r in g.iterrows()]
        n += 1
        redundant += any(variants(x, y) for x, y in combinations(rows, 2))
        order = list(range(len(rows)))
        if keep != "first":
            size = lambda k: len(words(" ".join(rows[k])))  # noqa: E731
            order.sort(key=size, reverse=keep == "longest")
        kept: list[int] = []
        for i in order:
            if not any(variants(rows[i], rows[k]) for k in kept):
                kept.append(i)
        out.extend(g.iloc[i] for i in sorted(kept))
    return pd.DataFrame(out, columns=gold.columns).reset_index(drop=True), redundant / n


def bia_systems(sents) -> dict[str, pd.DataFrame]:
    systems = {n: from_normalized(REPO / f"outputs/benchmark/normalized/{n}.jsonl", sents)
               for n in ("pt_oie", "ud_baseline", "ollama_gemma4", "multi2oie")}
    for n in ("DetIE", "ArgOE_LinguaKit", "DptOIE", "PragmaticOIE"):
        systems[n] = from_app(DEFAULT_RUN / f"{n}.jsonl", sents)
    return {n: to_frame(sents, p) for n, p in systems.items()}


def gamalho_systems() -> dict[str, pd.DataFrame]:
    out = {}
    for n, folder in GAMALHO_SYSTEMS.items():
        path = folder / "Gamalho_results.csv"
        if path.exists():
            out[n] = pd.read_csv(path, sep="\t")[["Sentence", *SLOTS]]
    return out


def evaluate(gold: pd.DataFrame, systems: dict[str, pd.DataFrame], idx) -> tuple[dict, dict]:
    boots, rows = {}, {}
    for n, pred in systems.items():
        sc = score_dataset(gold, pred, str(BENCH))
        boots[n] = bootstrap_carb(sc["_per_sentence"], idx, sc.get("n_pred_orphan", 0))
        rows[n] = {"f1": sc["f1_score"] * 100, "precision": sc["precision"] * 100, "recall": sc["recall"] * 100}
    for rank, n in enumerate(sorted(rows, key=lambda k: -rows[k]["f1"]), 1):
        rows[n]["rank"] = rank
    return rows, boots


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/external_benchmark/gold_dedup_ranking.json")
    args = p.parse_args(argv)

    sents = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    corpora = {"BIA": (gold_frame(sents), bia_systems(sents)),
               "Gamalho": (load_dataset("Gamalho", BENCH).gold[["Sentence", *SLOTS]], gamalho_systems())}
    result = {}
    for corpus, (gold, systems) in corpora.items():
        idx = None
        base, base_boot = None, None
        entry = {"n_gold": len(gold), "variants": {}}
        for keep in ("original", *KEEP):
            g, share = (gold, None) if keep == "original" else dedup(gold, keep)
            if idx is None:
                idx = make_indices(g["Sentence"].nunique())
            rows, boots = evaluate(g, systems, idx)
            v = {"n_gold": len(g), "systems": rows}
            if share is not None:
                v["redundant_sentence_pct"] = share * 100
            if base is None:
                base, base_boot = rows, boots
            else:
                swaps = []
                for a, b in combinations(rows, 2):
                    if (base[a]["rank"] < base[b]["rank"]) != (rows[a]["rank"] < rows[b]["rank"]):
                        hi, lo = (a, b) if rows[a]["rank"] < rows[b]["rank"] else (b, a)
                        d = paired_delta(boots[hi], boots[lo], hi, lo)
                        swaps.append({"now_ahead": hi, "now_behind": lo, "delta_f1": d["delta_point_f1"] * 100,
                                      "ci": [d["ci_low"] * 100, d["ci_high"] * 100]})
                v["swaps"] = swaps
            entry["variants"][keep] = v
            print(f"{corpus} {keep:<9} gold={len(g):4d}" + (f" redundantes={share*100:4.1f}%" if share is not None else ""))
            for n in sorted(rows, key=lambda k: rows[k]["rank"]):
                print(f"   {rows[n]['rank']:2d} {n:<18} F1={rows[n]['f1']:5.1f}")
            for s in v.get("swaps", []):
                print(f"   troca: {s['now_ahead']} > {s['now_behind']}  Δ={s['delta_f1']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]")
        result[corpus] = entry
    for corpus in ("WIKI200", "CETEN200"):
        gold = load_dataset(corpus, BENCH).gold[["Sentence", *SLOTS]]
        result[f"{corpus}_redundant_sentence_pct"] = dedup(gold, "first")[1] * 100
        print(f"{corpus} redundantes={result[f'{corpus}_redundant_sentence_pct']:.1f}%")
    write_json(args.output, result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
