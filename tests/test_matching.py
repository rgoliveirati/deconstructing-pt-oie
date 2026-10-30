import pytest

from src.benchmark.evaluation import evaluate_protocol


def _preds(*triples, sid="s1"):
    return [
        {"arg1": a1, "rel": r, "arg2": a2, "triple_id": f"{sid}_{i}"}
        for i, (a1, r, a2) in enumerate(triples)
    ]


def test_strict_exige_igualdade_exata(toy_sentences):
    preds = [
        _preds(("o brasil", "EXPORTA", "Soja.")),  # s1: igual após lower+espaços? "soja." != "soja"
        _preds(("A capital do Brasil", "é", "Brasília")),  # s2: exato
        [],  # s3
    ]
    res = evaluate_protocol("strict", toy_sentences, preds)
    # s1 falha por causa do ponto final em "Soja."
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 3


def test_tolerant_aceita_variacao_pequena(toy_sentences):
    preds = [
        _preds(("O Brasil", "exporta", "soja")),
        _preds(("capital do Brasil", "é", "Brasília")),  # arg1 3/4 tokens
        [],
    ]
    res = evaluate_protocol("tolerant", toy_sentences, preds)
    assert res["tp"] == 2
    assert res["fn"] == 2  # os dois golds de s3


def test_carb_style_score_ponderado(toy_sentences):
    preds = [
        _preds(("O Brasil", "exporta", "soja")),
        [],
        _preds(("Maria", "comprou", "pão"), ("Maria", "comprou", "queijo")),
    ]
    res = evaluate_protocol("carb_style", toy_sentences, preds)
    assert res["tp"] == 3 and res["fp"] == 0 and res["fn"] == 1


def test_per_sentence_consistente_com_totais(toy_sentences):
    preds = [
        _preds(("O Brasil", "exporta", "soja")),
        _preds(("X", "y", "Z")),
        [],
    ]
    res = evaluate_protocol("tolerant", toy_sentences, preds)
    assert sum(s["tp"] for s in res["per_sentence"]) == res["tp"]
    assert sum(s["fp"] for s in res["per_sentence"]) == res["fp"]
    assert sum(s["fn"] for s in res["per_sentence"]) == res["fn"]


def test_gold_invalido_nao_conta(toy_sentences):
    toy_sentences[0].gold[0].valid = False
    res = evaluate_protocol("strict", toy_sentences, [[], [], []])
    assert res["fn"] == 3  # 4 golds - 1 inválido


@pytest.mark.slow
def test_bia_legacy_preserva_comportamento(toy_sentences):
    """O protocolo legado usa as heurísticas históricas: prefixo, sufixo,
    subconjunto de palavras e relação parcial (importa src.extractor)."""
    preds = [
        _preds(("Brasil", "exporta", "soja")),  # canon_arg remove determinante do gold
        _preds(("A capital", "é", "Brasília")),  # prefixo de "A capital do Brasil"
        [],
    ]
    res = evaluate_protocol("bia_legacy", toy_sentences, preds)
    assert res["tp"] == 2
    assert res["protocol"] == "bia_legacy"
