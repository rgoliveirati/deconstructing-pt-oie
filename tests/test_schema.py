from src.benchmark.schemas import Prediction, validate_prediction_dict


def _pred(**kw):
    base = dict(
        sentence_id="s1", sentence="x", system="sys", triple_id="t1",
        arg1="a", rel="r", arg2="b",
    )
    base.update(kw)
    return Prediction(**base)


def test_prediction_valida():
    row = _pred().to_dict()
    assert validate_prediction_dict(row) == []
    assert row["confidence"] is None
    assert row["status"] == "ok"


def test_campo_ausente_detectado():
    row = _pred().to_dict()
    del row["raw_output"]
    assert any("raw_output" in p for p in validate_prediction_dict(row))


def test_status_invalido_detectado():
    row = _pred(status="weird").to_dict()
    assert any("status" in p for p in validate_prediction_dict(row))


def test_slots_devem_ser_strings():
    row = _pred().to_dict()
    row["arg1"] = 123
    assert any("arg1" in p for p in validate_prediction_dict(row))
