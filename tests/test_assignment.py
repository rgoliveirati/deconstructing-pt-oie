from src.benchmark.assignment import one_to_one_assignment


def test_um_para_um():
    # 2 golds, 3 preds; pred 0 casa bem com ambos os golds: só pode ser usada uma vez
    scores = {(0, 0): 0.9, (1, 0): 0.8, (0, 1): 0.7, (1, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0}
    m = one_to_one_assignment(2, 3, lambda g, p: scores[(g, p)], 0.5)
    golds = [g for g, _, _ in m]
    preds = [p for _, p, _ in m]
    assert len(set(golds)) == len(golds)
    assert len(set(preds)) == len(preds)
    assert (0, 0, 0.9) in m and (1, 1, 0.0) not in m
    # gold 1 fica com pred 1 (0.0 < threshold? não: 0.0 < 0.5 => sem par)
    assert m == [(0, 0, 0.9)] or (1, 1) not in [(g, p) for g, p, _ in m]


def test_empate_deterministico():
    # Todos os pares com o mesmo score: atribuição segue (gold_idx, pred_idx)
    m = one_to_one_assignment(2, 2, lambda g, p: 1.0, 0.5)
    assert m == [(0, 0, 1.0), (1, 1, 1.0)]


def test_threshold_exclui_pares():
    m = one_to_one_assignment(1, 1, lambda g, p: 0.4, 0.5)
    assert m == []


def test_determinismo_repetido():
    import random

    def score(g, p):
        return ((g * 7 + p * 13) % 10) / 10

    m1 = one_to_one_assignment(5, 6, score, 0.2)
    m2 = one_to_one_assignment(5, 6, score, 0.2)
    assert m1 == m2
