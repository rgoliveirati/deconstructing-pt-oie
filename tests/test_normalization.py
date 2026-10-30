from src.benchmark.normalization import (
    dedup_predictions, norm_space_lower, token_f1, tokenize,
)


def test_norm_space_lower():
    assert norm_space_lower("  O   Brasil ") == "o brasil"


def test_tokenize_remove_pontuacao_preserva_acentos():
    assert tokenize("Brasília, é!") == ["brasília", "é"]


def test_token_f1_igual():
    assert token_f1("o brasil", "O Brasil") == 1.0


def test_token_f1_parcial():
    # pred={a,b}, gold={a} -> P=0.5, R=1.0, F1=2/3
    assert abs(token_f1("a b", "a") - 2 / 3) < 1e-9


def test_token_f1_vazios():
    assert token_f1("", "") == 1.0
    assert token_f1("a", "") == 0.0


def test_dedup_mantem_primeira_ocorrencia():
    preds = [
        {"arg1": "A", "rel": "r", "arg2": "B", "triple_id": "1"},
        {"arg1": "a", "rel": "R", "arg2": "b", "triple_id": "2"},  # duplicata
        {"arg1": "A", "rel": "r", "arg2": "C", "triple_id": "3"},
    ]
    uniq, removed = dedup_predictions(preds)
    assert removed == 1
    assert [p["triple_id"] for p in uniq] == ["1", "3"]
