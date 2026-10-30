"""Baseline trivial verb-split, proposto por um revisor de uma versão anterior deste trabalho.

Para cada verbo da sentença gera uma "extração": tudo à esquerda do verbo é
arg1, o verbo é rel, tudo à direita é arg2. Serve como piso de sanidade dos
protocolos de matching: um protocolo que dá score alto a essas extrações está
creditando nonsense. Duas variantes de detecção (VERB; VERB+AUX) para que não
exista escolha conveniente.

Uso:
    python -m scripts.trivial_baseline \
        --gold data/bia_gold_sentences.jsonl \
        --output outputs/benchmark/metrics/trivial_verb_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.corpus import load_bia  # noqa: E402
from src.benchmark.evaluation import PROTOCOLS, evaluate_protocol  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="data/bia_gold_sentences.jsonl")
    p.add_argument("--output", default="outputs/benchmark/metrics/trivial_verb_baseline.json")
    args = p.parse_args(argv)

    import stanza

    sentences = load_bia(args.gold)
    nlp = stanza.Pipeline(
        "pt", processors="tokenize,mwt,pos", tokenize_no_ssplit=True, verbose=False
    )

    def trivial_preds(include_aux: bool):
        preds_by_sent = []
        for s in sentences:
            doc = nlp(s.sentence)
            words = [w for sent in doc.sentences for w in sent.words]
            allowed = {"VERB", "AUX"} if include_aux else {"VERB"}
            preds = []
            for i, w in enumerate(words):
                if w.upos in allowed:
                    preds.append({
                        "arg1": " ".join(x.text for x in words[:i]).strip(),
                        "rel": w.text,
                        "arg2": " ".join(x.text for x in words[i + 1:]).strip(),
                        "triple_id": f"{s.sentence_id}_trivial_{len(preds):03d}",
                    })
            preds_by_sent.append(preds)
        return preds_by_sent

    results = {}
    for label, include_aux in (("verb_only", False), ("verb_plus_aux", True)):
        preds = trivial_preds(include_aux)
        results[label] = {"n_triples": sum(len(x) for x in preds)}
        print(f"\n=== baseline trivial ({label}): {results[label]['n_triples']} extrações ===")
        for protocol in PROTOCOLS:
            r = evaluate_protocol(protocol, sentences, preds)
            results[label][protocol] = {
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                "precision": round(100 * r["precision"], 2),
                "recall": round(100 * r["recall"], 2),
                "f1": round(100 * r["f1"], 2),
            }
            print(f"  {protocol:>11}: P={100*r['precision']:6.2f} "
                  f"R={100*r['recall']:6.2f} F1={100*r['f1']:6.2f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsalvo em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
