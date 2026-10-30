import json
from pathlib import Path

from src.benchmark.corpus import (
    EXPECTED_GOLD_TRIPLES, EXPECTED_SENTENCES, load_bia, validate_corpus,
)

GOLD = Path(__file__).resolve().parents[1] / "data/bia_gold_sentences.jsonl"


def test_totais_do_corpus():
    sents = load_bia(GOLD)
    assert len(sents) == EXPECTED_SENTENCES == 262
    assert sum(len(s.gold) for s in sents) == EXPECTED_GOLD_TRIPLES == 427


def test_ids_unicos():
    sents = load_bia(GOLD)
    ids = [s.sentence_id for s in sents]
    assert len(set(ids)) == len(ids)


def test_sentencas_preservadas():
    """As sentenças carregadas são idênticas às do arquivo, byte a byte."""
    raw = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    sents = load_bia(GOLD)
    assert [s.sentence for s in sents] == [r["sentence"] for r in raw]
    assert [(g.arg1, g.rel, g.arg2) for s in sents for g in s.gold] == [
        (g["arg1"], g["rel"], g["arg2"]) for r in raw for g in r["gold"]
    ]


def test_validate_corpus_ok():
    report = validate_corpus(GOLD)
    assert report["ok"] is True
    assert report["unique_ids"] is True
    assert report["empty_sentences"] == 0
    assert len(report["sha256"]) == 64
