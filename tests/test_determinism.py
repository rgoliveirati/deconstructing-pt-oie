import numpy as np

from src.benchmark.bootstrap import bootstrap_metrics, make_indices
from src.benchmark.evaluation import evaluate_protocol


def _preds():
    return [
        [{"arg1": "O Brasil", "rel": "exporta", "arg2": "soja", "triple_id": "a"}],
        [{"arg1": "capital", "rel": "é", "arg2": "Brasília", "triple_id": "b"},
         {"arg1": "capital", "rel": "é", "arg2": "Brasília", "triple_id": "c"}],
        [],
    ]


def test_avaliacao_deterministica(toy_sentences):
    for protocol in ("strict", "tolerant", "carb_style"):
        r1 = evaluate_protocol(protocol, toy_sentences, _preds())
        r2 = evaluate_protocol(protocol, toy_sentences, _preds())
        assert (r1["tp"], r1["fp"], r1["fn"]) == (r2["tp"], r2["fp"], r2["fn"])
        assert r1["per_sentence"] == r2["per_sentence"]


def test_bootstrap_deterministico():
    counts = [{"tp": 1, "fp": 0, "fn": 1}, {"tp": 2, "fp": 1, "fn": 0}]
    idx = make_indices(2, 500, seed=42)
    r1 = bootstrap_metrics(counts, idx)
    r2 = bootstrap_metrics(counts, make_indices(2, 500, seed=42))
    assert np.array_equal(r1["_f1_samples"], r2["_f1_samples"])
    assert r1["f1"] == r2["f1"]
