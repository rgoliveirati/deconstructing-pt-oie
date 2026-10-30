"""Perfil de casamento dos sistemas nos corpora do PT-OIE-Benchmark.

Para cada sistema e corpus, refaz a atribuição gulosa um-para-um que o avaliador
do PT-OIE-Benchmark usa na precisão e classifica cada par casado (escore > 0):

* exato: o ARG1 previsto tem o mesmo conjunto de palavras normalizadas que o ARG1
  de referência;
* englobante: as palavras do ARG1 de referência são um subconjunto próprio das do
  ARG1 previsto (a predição cobre a referência e acrescenta material);
* outro: os demais.

A classificação usa o ARG1 (o segundo argumento) porque é nele que os spans longos
aparecem; ``triple_exact_pct`` registra também a igualdade nos três campos.

Também registra a razão entre o total de palavras do ARG1 previsto e do ARG1 de
referência nos pares casados, e o F1 do protocolo ``strict`` (média dos três
corpora). As predições são lidas como o avaliador as lê: todas as triplas, com
campo vazio como NaN (``nan`` no casamento, vazio no ``strict``). Normalização de palavras: ``normalize_words`` do avaliador, mantendo
repetições para medir comprimento.

Uso:
    python scripts/match_profile.py --output outputs/external_benchmark/match_profile.json
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.benchmark.corpus import GoldTriple, Sentence  # noqa: E402
from src.benchmark.evaluation import evaluate_protocol  # noqa: E402
from src.benchmark.external.carb import load_benchmark_evaluator  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.runtime import write_json  # noqa: E402

BENCH = REPO / ".external/PT-OIE-Benchmark"
CORPORA = ("WIKI200", "CETEN200", "Gamalho")
SLOTS = ("ARG0", "V", "ARG1")
SYSTEMS = {
    "PragmaticOIE": BENCH / "System_Results/PragmaticOIE",
    "MultiOIE": BENCH / "System_Results/MultiOIE",
    "DetIE": BENCH / "System_Results/DetIE",
    "TransAlign": BENCH / "System_Results/TransAlign",
    "GraphIE": BENCH / "System_Results/GraphIE",
    "PortOIE-LLama": BENCH / "System_Results/PortOIE-LLama",
    "DptOIE": BENCH / "System_Results/DptOIE",
    "pt_oie": REPO / "outputs/external_benchmark/predictions/benchmark_format/pt_oie",
    "ollama_gemma4": REPO / "outputs/external_benchmark/predictions/benchmark_format/ollama_gemma4",
}


def words(s) -> list[str]:
    s = unicodedata.normalize("NFD", str(s).lower()).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"[^\w\s]", "", s).split()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/external_benchmark/match_profile.json")
    args = p.parse_args(argv)

    ev = load_benchmark_evaluator(str(BENCH))
    key = lambda s: frozenset(ev.normalize_words(s))  # noqa: E731
    data = {c: load_dataset(c, BENCH) for c in CORPORA}
    result = {}
    for system, folder in SYSTEMS.items():
        counts = {"exact": 0, "encompassing": 0, "other": 0}
        triple_exact = 0
        pred_len = gold_len = 0
        strict = []
        for corpus in CORPORA:
            path = folder / f"{corpus}_results.csv"
            if not path.exists():
                continue
            gold = data[corpus].gold
            pred = pd.read_csv(path, sep="\t")
            gold_by, pred_by = defaultdict(list), defaultdict(list)
            for _, r in gold.iterrows():
                gold_by[key(r["Sentence"])].append(tuple(str(r[s]) for s in SLOTS))
            for _, r in pred.iterrows():
                pred_by[key(r["Sentence"])].append(tuple(str(r[s]) for s in SLOTS))
            for k, g in gold_by.items():
                ps = pred_by.get(k, [])
                if not k or not ps:
                    continue
                m = np.array([[ev.get_tuple_match_score(pt, gt) for gt in g] for pt in ps])
                for _ in range(min(len(ps), len(g))):
                    if m.max() == 0:
                        break
                    pi, gi = np.unravel_index(np.argmax(m), m.shape)
                    pw = [set(words(x)) for x in ps[pi]]
                    gw = [set(words(x)) for x in g[gi]]
                    triple_exact += pw == gw
                    if pw[2] == gw[2]:
                        counts["exact"] += 1
                    elif gw[2] < pw[2]:
                        counts["encompassing"] += 1
                    else:
                        counts["other"] += 1
                    pred_len += len(words(ps[pi][2]))
                    gold_len += len(words(g[gi][2]))
                    m[pi, :] = 0
                    m[:, gi] = 0
            # strict, com as sentenças alinhadas pela mesma chave do avaliador
            sents = [Sentence(f"{corpus}_{i:04d}", s, None, i,
                              [GoldTriple(*t) for t in gold_by.get(key(s), [])])
                     for i, s in enumerate(data[corpus].sentences)]
            blank = lambda x: "" if x == "nan" else x  # noqa: E731
            preds = [[{"arg1": blank(a), "rel": blank(r), "arg2": blank(b)} for a, r, b in pred_by.get(key(s.sentence), [])]
                     for s in sents]
            strict.append(evaluate_protocol("strict", sents, preds)["f1"] * 100)
        n = sum(counts.values())
        result[system] = {
            "matched_pairs": n,
            "exact_pct": 100 * counts["exact"] / n,
            "encompassing_pct": 100 * counts["encompassing"] / n,
            "triple_exact_pct": 100 * triple_exact / n,
            "arg2_length_ratio": pred_len / gold_len,
            "strict_f1_mean": sum(strict) / len(strict),
            "n_corpora": len(strict),
        }
        r = result[system]
        print(f"{system:<15} pares={n:4d} exato={r['exact_pct']:5.1f}% englobante={r['encompassing_pct']:5.1f}% "
              f"razão_arg2={r['arg2_length_ratio']:4.2f} strict={r['strict_f1_mean']:5.1f} ({len(strict)} corpora)")
    write_json(args.output, result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
