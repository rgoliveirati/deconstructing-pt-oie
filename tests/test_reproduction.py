"""Regressão: as predições salvas do benchmark reproduzem os números do artigo
sob o protocolo bia_legacy (Tabela 3 da submissão)."""

from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NORM = REPO / "outputs/benchmark/normalized"

PAPER = {
    # (TP, FP, FN) da Tabela 3
    "pt_oie": (265, 273, 162),        # rq3_attn_on_thr0
    "ud_baseline": (272, 333, 155),   # rq1_baseline_ud_puro (P=44.96 R=63.70)
}


@pytest.mark.slow
@pytest.mark.parametrize("system", sorted(PAPER))
def test_bia_legacy_reproduz_artigo(system):
    norm_path = NORM / f"{system}.jsonl"
    if not norm_path.exists():
        pytest.skip("benchmark ainda não executado")
    from src.benchmark.corpus import load_bia
    from src.benchmark.evaluation import evaluate_protocol
    from src.benchmark.schemas import read_jsonl

    sentences = load_bia(REPO / "data/bia_gold_sentences.jsonl")
    by = defaultdict(list)
    for r in read_jsonl(norm_path):
        by[r["sentence_id"]].append(r)
    preds = [by.get(s.sentence_id, []) for s in sentences]
    res = evaluate_protocol("bia_legacy", sentences, preds)
    assert (res["tp"], res["fp"], res["fn"]) == PAPER[system]
