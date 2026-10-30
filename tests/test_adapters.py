from pathlib import Path

from src.benchmark.registry import SYSTEM_NAMES, build_system
from src.benchmark.runner import run_system
from src.benchmark.schemas import Prediction, read_jsonl
from src.benchmark.systems.base import OpenIESystem, SystemUnavailable


def test_registry_constroi_todos_os_sistemas():
    for name in SYSTEM_NAMES:
        system = build_system(name, {})
        assert system.name == name


def test_externos_indisponiveis_sem_artefatos(tmp_path):
    for name in ("dptoie", "multi2oie", "portnoie"):
        system = build_system(name, {"external_dir": str(tmp_path / name)})
        try:
            system.setup()
            raise AssertionError(f"{name} deveria estar indisponível")
        except SystemUnavailable as exc:
            assert exc.reason
            assert exc.system == name


def test_runner_registra_indisponibilidade(tmp_path, toy_sentences):
    system = build_system("portnoie", {"external_dir": str(tmp_path / "vazio")})
    result = run_system(system, toy_sentences, tmp_path)
    assert result.status == "unavailable"
    assert result.reason
    assert result.n_ok == result.n_error == 0  # indisponível não vale zero


class _FlakySystem(OpenIESystem):
    """Falha na 2a sentença; usado para testar erros e retomada."""

    name = "flaky"

    def __init__(self):
        self.calls: list[str] = []

    def setup(self):
        pass

    def extract(self, sentence_id, sentence):
        self.calls.append(sentence_id)
        if sentence_id == "s2":
            raise RuntimeError("falha simulada")
        return [
            Prediction(
                sentence_id=sentence_id, sentence=sentence, system=self.name,
                triple_id=f"{sentence_id}_0", arg1="a", rel="r", arg2="b",
            )
        ]


def test_runner_preserva_erros_sem_converter_em_vazio(tmp_path, toy_sentences):
    system = _FlakySystem()
    result = run_system(system, toy_sentences, tmp_path)
    assert result.status == "ok_with_errors"
    assert result.n_ok == 2 and result.n_error == 1
    errs = read_jsonl(tmp_path / "errors/flaky.jsonl")
    assert len(errs) == 1 and errs[0]["sentence_id"] == "s2"
    assert errs[0]["status"] == "error" and "falha simulada" in errs[0]["error"]
    raws = read_jsonl(tmp_path / "raw/flaky.jsonl")
    assert {r["sentence_id"]: r["status"] for r in raws} == {
        "s1": "ok", "s2": "error", "s3": "ok",
    }


class _BatchSystem(OpenIESystem):
    """Sistema em lote: 1 processo para todas as sentenças (como o DptOIE)."""

    name = "batchy"
    supports_batch = True

    def setup(self):
        pass

    def extract_batch(self, sentences):
        out = {}
        for sid, sent in sentences:
            if sid == "s3":
                continue  # ausente do lote -> runner registra erro
            out[sid] = [
                Prediction(
                    sentence_id=sid, sentence=sent, system=self.name,
                    triple_id=f"{sid}_0", arg1="a", rel="r", arg2="b",
                )
            ]
        return out


def test_runner_modo_lote(tmp_path, toy_sentences):
    result = run_system(_BatchSystem(), toy_sentences, tmp_path)
    assert result.n_ok == 2 and result.n_error == 1
    raws = read_jsonl(tmp_path / "raw/batchy.jsonl")
    assert all(r["runtime"].get("amortized_from_batch") for r in raws)
    errs = read_jsonl(tmp_path / "errors/batchy.jsonl")
    assert errs[0]["sentence_id"] == "s3"
    norm = read_jsonl(tmp_path / "normalized/batchy.jsonl")
    assert {r["sentence_id"] for r in norm} == {"s1", "s2"}


def test_dptoie_parse_csv_autores(tmp_path, toy_sentences, monkeypatch):
    """O parser do CSV do DptOIE preserva slots e indexação por sentença."""
    from src.benchmark.systems.dptoie import DptOIESystem

    csv_text = (
        '"ID SENTENÇA";"SENTENÇA";"ID EXTRAÇÃO";"ARG1";"REL";"ARG2";'
        '"COERÊNCIA";"MINIMALIDADE";"MÓDULO SUJEITO";"MÓDULO RELAÇÃO";"MÓDULO ARG2"\n'
        '"1";"O Brasil exporta soja . ";"1.0";"O Brasil ";" exporta";"soja ";"";"";"1";"1";"1"\n'
        '"2";"A capital de o Brasil é Brasília . ";"";"";"";"";"";"";"";""\n'
    )

    system = DptOIESystem({"external_dir": str(tmp_path)})
    system.jar = tmp_path / "fake.jar"

    def fake_run(cmd, **kw):
        (tmp_path / "extractedFactsByDptOIE.csv").write_text(csv_text, encoding="utf-8")

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr("src.benchmark.systems.dptoie.subprocess.run", fake_run)
    result = system.extract_batch(
        [(s.sentence_id, s.sentence) for s in toy_sentences[:2]]
    )
    assert [
        (p.arg1, p.rel, p.arg2) for p in result[toy_sentences[0].sentence_id]
    ] == [("O Brasil", "exporta", "soja")]
    assert result[toy_sentences[1].sentence_id] == []  # sem extração, explícito


def test_retomada_pula_sentencas_concluidas(tmp_path, toy_sentences):
    system = _FlakySystem()
    run_system(system, toy_sentences, tmp_path)
    first_calls = list(system.calls)

    system2 = _FlakySystem()
    result = run_system(system2, toy_sentences, tmp_path, resume=True)
    assert first_calls == ["s1", "s2", "s3"]
    assert system2.calls == []  # tudo já registrado (inclusive o erro)
    assert result.n_ok == 2 and result.n_error == 1
