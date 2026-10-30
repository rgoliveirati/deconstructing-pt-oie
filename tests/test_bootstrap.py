import numpy as np

from src.benchmark.bootstrap import (
    bootstrap_metrics, make_indices, paired_f1_difference,
)


def _counts(rows):
    return [{"tp": t, "fp": f, "fn": n} for t, f, n in rows]


def test_indices_deterministicos_com_seed():
    a = make_indices(10, 100, seed=42)
    b = make_indices(10, 100, seed=42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make_indices(10, 100, seed=43))


def test_unidade_de_reamostragem_e_sentenca():
    idx = make_indices(5, 50, seed=1)
    assert idx.shape == (50, 5)
    assert idx.min() >= 0 and idx.max() < 5


def test_ci_cobre_valor_pontual():
    counts = _counts([(2, 1, 0), (1, 0, 1), (3, 2, 1), (0, 1, 2)])
    res = bootstrap_metrics(counts, make_indices(4, 2000, seed=42))
    assert res["f1"]["low"] <= res["point"]["f1"] <= res["f1"]["high"]
    assert 0 <= res["point"]["precision"] <= 1


def test_diferenca_pareada_de_sistemas_identicos_e_zero():
    counts = _counts([(2, 1, 0), (1, 0, 1), (3, 2, 1)])
    idx = make_indices(3, 1000, seed=42)
    a = bootstrap_metrics(counts, idx)
    b = bootstrap_metrics(counts, idx)
    d = paired_f1_difference(a, b)
    assert d["delta_point"] == 0.0
    assert d["ci_low"] == 0.0 and d["ci_high"] == 0.0


def test_diferenca_pareada_detecta_sistema_melhor():
    idx = make_indices(4, 2000, seed=42)
    melhor = bootstrap_metrics(_counts([(3, 0, 0), (2, 0, 0), (3, 1, 0), (2, 0, 1)]), idx)
    pior = bootstrap_metrics(_counts([(1, 2, 2), (1, 1, 1), (1, 3, 2), (0, 2, 3)]), idx)
    d = paired_f1_difference(melhor, pior)
    assert d["delta_point"] > 0
    assert d["prop_delta_gt_0"] > 0.99
