"""Coleta sentenças biomédicas em português para o teste de fidelidade de polaridade.

Fonte: resumos em português (bloco ``OtherAbstract Language="por"``) de artigos
indexados no PubMed. São textos científicos públicos — **nenhum dado de paciente
é usado**.

Este corpus não tem gold e não precisa ter: a medida de fidelidade de polaridade
ancora na sintaxe da própria sentença (ver ``scripts/polarity_fidelity.py``), não
na anotação.

**Amostragem dirigida, e por que não enviesa.** Retemos apenas sentenças que
contêm marcador de negação ou de modalidade. A taxa medida é
``P(preserva | predicado marcado e coberto)``, condicional ao marcador; sobre-
amostrar sentenças marcadas aumenta o número de casos sem deslocar essa
probabilidade. O que este corpus **não** permite estimar é com que frequência
negação ocorre em texto biomédico.

**Reprodução.** A busca original (``--query``, ordem do PubMed em 2026-08-12) não é
estável no tempo. ``data/biomed_pt_pmids.txt`` lista, em ordem, os artigos de onde
vieram as 241 sentenças; com ``--pmids`` o script baixa só esses artigos e reconstrói
o arquivo idêntico (o texto dos resumos não é redistribuído no repositório).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.runtime import now_iso, write_json  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NEG = r"\b(não|nao|nunca|jamais|nenhum[ao]?s?|nem|sem)\b"
MOD = r"\b(pode[m]?|podia[m]?|poderia[m]?|deve[m]?|deveria[m]?|precisa[m]?|parece[m]?|tende[m]?)\b"


def _get(url: str, tries: int = 3) -> str:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3)
    return ""


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.;])\s+(?=[A-ZÀ-Ý])", text)
    return [p.strip() for p in parts if p.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-articles", type=int, default=400)
    p.add_argument("--max-sentences", type=int, default=400)
    p.add_argument("--query", default='portuguese[lang] AND hasabstract AND ("2015"[PDAT] : "2026"[PDAT])')
    p.add_argument("--pmids", default=None,
                   help="arquivo com um PMID por linha; substitui a busca (ex.: data/biomed_pt_pmids.txt)")
    p.add_argument("--output", default="data/biomed_pt_sentences.txt")
    args = p.parse_args(argv)

    if args.pmids:
        ids = Path(args.pmids).read_text().split()
    else:
        q = urllib.parse.quote(args.query)
        ids = re.findall(r"<Id>(\d+)</Id>",
                         _get(EUTILS + f"esearch.fcgi?db=pubmed&term={q}&retmax={args.n_articles}"))
    print(f"{len(ids)} artigos encontrados")

    abstracts: list[str] = []
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        xml = _get(EUTILS + f"efetch.fcgi?db=pubmed&id={','.join(batch)}&retmode=xml")
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        by_pmid = {art.findtext(".//PMID"): art for art in root.iter("PubmedArticle")}
        for art in (by_pmid[i] for i in batch if i in by_pmid):
            for oa in art.iter("OtherAbstract"):
                if oa.get("Language") == "por":
                    txt = " ".join("".join(t.itertext()) for t in oa.iter("AbstractText"))
                    if len(txt) > 200:
                        abstracts.append(txt)
        print(f"  {i + len(batch)}/{len(ids)} processados, {len(abstracts)} resumos em pt", flush=True)
        time.sleep(0.4)  # cortesia com a API

    seen: set[str] = set()
    kept: list[str] = []
    n_neg = n_mod = 0
    for ab in abstracts:
        for s in split_sentences(ab):
            if not (40 <= len(s) <= 320) or s in seen:
                continue
            has_neg = bool(re.search(NEG, s, re.I))
            has_mod = bool(re.search(MOD, s, re.I))
            if not (has_neg or has_mod):
                continue
            seen.add(s)
            kept.append(s)
            n_neg += has_neg
            n_mod += has_mod
            if len(kept) >= args.max_sentences:
                break
        if len(kept) >= args.max_sentences:
            break

    out = REPO / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(kept) + "\n"
    out.write_text(text, encoding="utf-8")

    write_json(out.with_suffix(".meta.json"), {
        "created_at": now_iso(), "source": "PubMed OtherAbstract[Language=por]",
        "query": None if args.pmids else args.query, "pmids_file": args.pmids,
        "n_articles_requested": len(ids),
        "n_abstracts_pt": len(abstracts), "n_sentences": len(kept),
        "n_with_negation_marker": n_neg, "n_with_modal_marker": n_mod,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "note": "texto científico público; nenhum dado de paciente",
    })
    print(f"\n{len(kept)} sentenças ({n_neg} com marcador de negação, {n_mod} com modal)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
