"""Fidelidade de polaridade e modalidade das extrações.

Pergunta: quando um sistema extrai um predicado que está **negado** (ou
**modalizado**) na sentença, ele preserva o marcador no descritor de relação?

Por que isso importa e por que o F1 não mede: em ``(a droga, não reduz, a
mortalidade)`` o token ``não`` é 1 de ~8 da tripla. Um sistema que o descarta
produz ``(a droga, reduz, a mortalidade)`` — o fato **oposto** — e ainda assim
casa quase perfeitamente sob CaRB. A métrica é cega exatamente ao erro que mais
importa em texto científico e clínico.

Desenho — ancorado na SINTAXE, não no gold:

1. cada sentença é analisada com Stanza; localizam-se os verbos que têm um
   dependente de negação (``advmod``/``det`` com lema em ``NEG_LEMMAS``) ou um
   auxiliar modal (``aux`` com lema em ``MODAL_LEMMAS``);
2. para cada sistema, uma tripla **cobre** esse predicado se a forma de
   superfície do verbo aparece no descritor de relação;
3. entre as triplas que cobrem, mede-se quantas **preservam** o marcador.

Isso não depende do gold, portanto é imune a diferenças de convenção de anotação
entre corpora — o que importa aqui, dado que os corpora discordam entre si.

Duas taxas são reportadas:

* ``por tripla``  — entre as triplas que cobrem o predicado, quantas preservam;
* ``por predicado`` — entre os predicados cobertos, para quantos ao menos uma
  tripla preserva (mais generoso: basta acertar uma vez).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.runtime import now_iso, write_json  # noqa: E402

BENCH = REPO / ".external/PT-OIE-Benchmark"
LREC_DATASETS = ("WIKI200", "CETEN200", "Gamalho")

NEG_LEMMAS = {"não", "nao", "nunca", "jamais", "nem", "nada", "ninguém", "ninguem",
              "nenhum", "nenhuma", "nenhuns", "nenhumas"}
MODAL_LEMMAS = {"poder", "dever", "precisar", "pretender", "costumar", "tender"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^\w\s]", " ", s)


def toks(s: str) -> list[str]:
    return norm(s).split()


def find_marked_predicates(sent) -> dict[str, list[dict]]:
    """Verbos negados ou modalizados, no parse da sentença.

    Modalidade tem DUAS realizações em UD para o português, e só a segunda é
    frequente no modelo do Stanza:

    * ``aux`` — o modal é auxiliar do verbo pleno;
    * **núcleo + ``xcomp``** — o modal é o próprio núcleo e o verbo pleno é seu
      complemento oracional ("pode indicar" -> pode=VERB raiz, indicar=xcomp).
      Aqui o predicado marcado é o verbo do ``xcomp`` e o marcador é o modal.
    """
    out = {"neg": [], "mod": []}
    by_id = {w.id: w for w in sent.words}
    for w in sent.words:
        head = by_id.get(w.head)
        lemma = norm(w.lemma or w.text).strip()
        deprel = w.deprel or ""

        if head is not None and (head.upos or "") in ("VERB", "AUX"):
            if lemma in NEG_LEMMAS and deprel.startswith(("advmod", "det")):
                out["neg"].append({"verb": head.text, "marker": w.text})
                continue
            if lemma in MODAL_LEMMAS and deprel.startswith("aux"):
                out["mod"].append({"verb": head.text, "marker": w.text})
                continue

        # modal como núcleo, verbo pleno em xcomp
        if lemma in MODAL_LEMMAS and (w.upos or "") in ("VERB", "AUX"):
            for c in sent.words:
                if c.head == w.id and (c.deprel or "").startswith("xcomp") \
                        and (c.upos or "") in ("VERB", "AUX"):
                    out["mod"].append({"verb": c.text, "marker": w.text})
    return out


#: Nomes equivalentes entre as duas origens de predição (nosso benchmark do BIA
#: usa minúsculas; o repositório do LREC usa CamelCase). Unificados para que um
#: sistema não vire duas linhas na tabela.
CANONICAL = {"dptoie": "DptOIE", "multi2oie": "MultiOIE", "portnoie": "PortNOIE"}


def load_sentences(dataset: str) -> list[str]:
    if dataset == "BIA":
        items = [json.loads(l) for l in
                 (REPO / "data/bia_gold_sentences.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        return [d["sentence"] for d in items]
    if dataset == "OIEC-PT":
        from src.benchmark.oiec import load_oiec
        path = REPO / "data/external/OIEC-PT/OIEC-PT.txt"
        return [s.sentence for s in load_oiec(path)]
    if dataset == "BIOMED":
        # resumos científicos em português (PubMed); ver scripts/fetch_biomed_pt.py.
        # Não tem gold e não precisa: a medida ancora na sintaxe da sentença.
        p = REPO / "data/biomed_pt_sentences.txt"
        return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if dataset.startswith("CUSTOM:"):
        # corpus arbitrário, uma sentença por linha
        return [l for l in Path(dataset[7:]).read_text(encoding="utf-8").splitlines() if l.strip()]
    p = BENCH / "Gold/Sentences" / f"{dataset}_Sentences.txt"
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def system_predictions(dataset: str) -> dict[str, dict[str, list[tuple]]]:
    """{sistema: {chave_da_sentença: [(arg0, rel, arg1), ...]}}

    Procura o arquivo do corpus nas DUAS origens (nossas predições e as do
    repositório do LREC), porque um mesmo sistema pode ter sido executado por nós
    num corpus e pelos autores do benchmark noutro.
    """
    candidates: dict[str, list[Path]] = defaultdict(list)
    ours = REPO / "outputs/external_benchmark/predictions/benchmark_format"
    if ours.exists():
        for d in sorted(ours.iterdir()):
            if d.is_dir():
                candidates[CANONICAL.get(d.name, d.name)].append(d)
    if (BENCH / "System_Results").exists():
        for d in sorted((BENCH / "System_Results").iterdir()):
            if d.is_dir():
                candidates[CANONICAL.get(d.name, d.name)].append(d)

    safe = dataset.replace("CUSTOM:", "").replace("/", "_")
    out: dict[str, dict[str, list[tuple]]] = {}
    for name, dirs in candidates.items():
        path = next((d / f"{safe}_results.csv" for d in dirs
                     if (d / f"{safe}_results.csv").exists()), None)
        if path is None:
            continue
        df = pd.read_csv(path, sep="\t").fillna("")
        by_sent: dict[str, list[tuple]] = defaultdict(list)
        for _, r in df.iterrows():
            by_sent[" ".join(toks(r.Sentence))].append((str(r.ARG0), str(r.V), str(r.ARG1)))
        out[name] = by_sent
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(LREC_DATASETS))
    p.add_argument("--output", default="outputs/polarity/polarity_fidelity.json")
    p.add_argument("--examples", type=int, default=8)
    args = p.parse_args(argv)

    import stanza

    nlp = stanza.Pipeline("pt", processors="tokenize,mwt,pos,lemma,depparse",
                          tokenize_no_ssplit=True, verbose=False,
                          download_method=stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES)

    # contadores: sistema -> tipo -> {covered_triples, preserved_triples, covered_preds, preds_ok}
    stats: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    examples: dict = defaultdict(list)
    marked_total = {"neg": 0, "mod": 0}

    for dataset in args.datasets:
        sentences = load_sentences(dataset)
        preds_by_system = system_predictions(dataset)
        if not preds_by_system:
            print(f"  {dataset}: nenhuma predição encontrada, pulando", file=sys.stderr)
            continue

        for sentence in sentences:
            marked = find_marked_predicates(nlp(sentence).sentences[0])
            if not marked["neg"] and not marked["mod"]:
                continue
            key = " ".join(toks(sentence))
            for kind in ("neg", "mod"):
                marked_total[kind] += len(marked[kind])
                for item in marked[kind]:
                    vtok = norm(item["verb"]).strip()
                    mtok = norm(item["marker"]).strip()
                    if not vtok or not mtok:
                        continue
                    for system, by_sent in preds_by_system.items():
                        triples = by_sent.get(key, [])
                        covering = [t for t in triples if vtok in toks(t[1])]
                        if not covering:
                            continue
                        stats[system][kind]["covered_preds"] += 1
                        any_ok = False
                        for t in covering:
                            stats[system][kind]["covered_triples"] += 1
                            if mtok in toks(t[1]):
                                stats[system][kind]["preserved_triples"] += 1
                                any_ok = True
                            elif len(examples[(system, kind)]) < args.examples:
                                examples[(system, kind)].append(
                                    {"dataset": dataset, "sentence": sentence,
                                     "marker": item["marker"], "verb": item["verb"],
                                     "triple": list(t)})
                        if any_ok:
                            stats[system][kind]["preds_ok"] += 1
        print(f"  {dataset}: processado", flush=True)

    rows = []
    for system, per_kind in stats.items():
        row = {"system": system}
        for kind in ("neg", "mod"):
            c = per_kind[kind]
            ct, pt = c["covered_triples"], c["preserved_triples"]
            cp, po = c["covered_preds"], c["preds_ok"]
            row[f"{kind}_covered_triples"] = ct
            row[f"{kind}_por_tripla_%"] = round(pt / ct * 100, 1) if ct else None
            row[f"{kind}_covered_preds"] = cp
            row[f"{kind}_por_predicado_%"] = round(po / cp * 100, 1) if cp else None
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("neg_por_tripla_%", ascending=False, na_position="last")

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {
        "created_at": now_iso(),
        "datasets": args.datasets,
        "marked_predicates_found": marked_total,
        "neg_lemmas": sorted(NEG_LEMMAS),
        "modal_lemmas": sorted(MODAL_LEMMAS),
        "results": rows,
        "examples_of_loss": {f"{s}|{k}": v for (s, k), v in examples.items()},
    })
    df.to_csv(out.with_suffix(".csv"), index=False)

    print(f"\npredicados marcados no texto: {marked_total['neg']} negados, "
          f"{marked_total['mod']} modalizados\n")
    print(f"{'sistema':<18}{'NEG n':>7}{'NEG/tripla':>12}{'NEG/pred':>10}"
          f"{'MOD n':>8}{'MOD/tripla':>12}{'MOD/pred':>10}")
    for r in df.to_dict("records"):
        f = lambda v: "—" if v is None or pd.isna(v) else f"{v:.1f}"  # noqa: E731
        print(f"{r['system']:<18}{r['neg_covered_triples']:>7}{f(r['neg_por_tripla_%']):>12}"
              f"{f(r['neg_por_predicado_%']):>10}{r['mod_covered_triples']:>8}"
              f"{f(r['mod_por_tripla_%']):>12}{f(r['mod_por_predicado_%']):>10}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
