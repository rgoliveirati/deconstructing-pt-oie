"""Famílias sintáticas das triplas de referência que o gerador não alcança.

Uma tripla de referência é inalcançável quando nenhum candidato do gerador
(``_generate_all_candidates``, configuração ``rq3_attn_on_thr0``) tem relação com
alguma palavra em comum com a relação da referência (F1 do campo V igual a zero no
avaliador do PT-OIE-Benchmark): o predicado nunca é proposto.

A família é a relação de dependência UD (parse do Stanza) do predicado da
referência: a primeira palavra da sentença que pertence à relação e é VERB ou
AUX; se for ``cop`` ou ``aux``, sobe para o núcleo. ``root`` é oração principal;
``acl`` oração reduzida ou participial; ``acl:relcl`` relativa; ``advcl``
adverbial; ``conj`` coordenada; ``ccomp``/``xcomp``/``csubj`` completiva;
sem verbo na relação, ``no_verb``.

Uso:
    python scripts/unreachable_families.py --output outputs/external_benchmark/unreachable_families.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.external.carb import load_benchmark_evaluator  # noqa: E402
from src.benchmark.external.corpora import load_dataset  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402

BENCH = REPO / ".external/PT-OIE-Benchmark"
CORPORA = ("WIKI200", "CETEN200", "Gamalho")
FAMILY = {"root": "main_clause", "parataxis": "main_clause", "acl": "reduced_or_participial",
          "acl:relcl": "relative", "advcl": "adverbial", "conj": "coordinated",
          "ccomp": "complement", "xcomp": "complement", "csubj": "complement"}


def family(rel_words: set[str], words, norm) -> str:
    for w in words:
        if w.upos in ("VERB", "AUX") and norm(w.text) & rel_words:
            head = words[w.head - 1] if w.deprel in ("cop", "aux", "aux:pass") and w.head else w
            dep = head.deprel if head.deprel == "acl:relcl" else head.deprel.split(":")[0]
            return FAMILY.get(dep, "other")
    return "no_verb"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "outputs/external_benchmark/unreachable_families.json")
    args = p.parse_args(argv)

    import stanza
    from src.extractor import Config, OpenIEExtractorBIA

    ev = load_benchmark_evaluator(str(BENCH))
    norm = ev.normalize_words
    key = lambda s: frozenset(norm(s))  # noqa: E731
    nlp = stanza.Pipeline("pt", processors="tokenize,mwt,pos,lemma,depparse", tokenize_no_ssplit=True, verbose=False)
    ex = OpenIEExtractorBIA(config=Config(
        bert_model="neuralmind/bert-base-portuguese-cased",
        bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"),
        seed=13, cop_mode="full", theory_mode="off", apply_theoretical_rules=False,
        no_attn=False, attn_decision_enabled=True, attn_rerank_enabled=True,
        heads_mode="rank", top_k_heads=10, attn_threshold=0.0,
    ), verbose=False)

    result = {"created_at": now_iso(), "config": "rq3_attn_on_thr0", "corpora": {}}
    for corpus in CORPORA:
        data = load_dataset(corpus, BENCH)
        gold_by = defaultdict(list)
        for _, r in data.gold.iterrows():
            gold_by[key(r["Sentence"])].append(str(r["V"]))
        # Como no avaliador, sentenças com a mesma chave somam seus candidatos e
        # compartilham as referências; o parse é o da primeira.
        cand_by, words_by = defaultdict(list), {}
        for s in data.sentences:
            cand_by[key(s)] += [norm(c.rel) for c in ex._generate_all_candidates(s)[0]]
            words_by.setdefault(key(s), [w for sent in nlp(s).sentences for w in sent.words])
        counts, n_gold = Counter(), 0
        for k, words in words_by.items():
            golds = gold_by.get(k, [])
            n_gold += len(golds)
            for g in golds:
                gw = norm(g)
                if not any(gw & c for c in cand_by[k]):
                    counts[family(gw, words, norm)] += 1
        n = sum(counts.values())
        result["corpora"][corpus] = {
            "n_gold": n_gold, "n_unreachable": n, "unreachable_pct": 100 * n / n_gold,
            "families_pct": {f: 100 * c / n for f, c in counts.most_common()},
            "families_n": dict(counts.most_common()),
        }
        print(corpus, n, "/", n_gold, {f: round(100 * c / n, 1) for f, c in counts.most_common()}, flush=True)
    write_json(args.output, result)
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
