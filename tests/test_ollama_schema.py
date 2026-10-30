import json

import pytest

from src.benchmark.systems.base import SystemUnavailable
from src.benchmark.systems.ollama_gemma4 import (
    OllamaError, OllamaGemma4System, TRIPLES_JSON_SCHEMA,
)


def _system(**overrides):
    cfg = {
        "base_url": "http://localhost:11434",
        "model": "gemma4:latest",
        "generation": {"temperature": 0.0, "seed": 42},
        "structured_output": {"enabled": True, "max_repair_attempts": 1},
    }
    cfg.update(overrides)
    return OllamaGemma4System(cfg)


def test_schema_exige_tres_slots():
    props = TRIPLES_JSON_SCHEMA["properties"]["triples"]["items"]
    assert props["required"] == ["arg1", "rel", "arg2"]
    assert props["additionalProperties"] is False


def test_parse_valido():
    content = json.dumps({"triples": [{"arg1": "A", "rel": "é", "arg2": "B"}]})
    assert OllamaGemma4System._parse_triples(content) == [
        {"arg1": "A", "rel": "é", "arg2": "B"}
    ]


def test_parse_remove_cerca_markdown_sem_alterar_conteudo():
    fenced = '```json\n{\n"triples": [\n{\n"arg1": "A", "rel": "é", "arg2": "B"}\n]\n}\n```'
    assert OllamaGemma4System._parse_triples(fenced) == [
        {"arg1": "A", "rel": "é", "arg2": "B"}
    ]
    assert OllamaGemma4System._strip_markdown_fence('{"triples": []}') == '{"triples": []}'


def test_parse_lista_vazia_explicita_e_valida():
    assert OllamaGemma4System._parse_triples('{"triples": []}') == []


def test_parse_json_invalido_levanta():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        OllamaGemma4System._parse_triples("não é json")


def test_parse_fora_do_schema_levanta():
    with pytest.raises(ValueError):
        OllamaGemma4System._parse_triples('{"triples": [{"arg1": "só um slot"}]}')
    with pytest.raises(ValueError):
        OllamaGemma4System._parse_triples('{"outra_chave": []}')


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_reparo_unico_para_json_invalido(monkeypatch):
    system = _system()
    system.system_prompt = "prompt"
    system.digest = "sha256:abc"
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            return _FakeResponse(200, {"message": {"content": "texto solto"}})
        return _FakeResponse(200, {
            "message": {"content": '{"triples": [{"arg1": "A", "rel": "r", "arg2": "B"}]}'}
        })

    monkeypatch.setattr("src.benchmark.systems.ollama_gemma4.requests.post", fake_post)
    preds = system.extract("s1", "Sentença.")
    assert len(calls) == 2  # principal + 1 reparo
    assert len(preds) == 1
    assert preds[0].metadata["repaired"] is True
    assert preds[0].metadata["parse_error_before_repair"]
    # O reparo reenvia a resposta original e pede apenas conversão de formato
    repair_msgs = calls[1]["messages"]
    assert repair_msgs[-2]["content"] == "texto solto"
    assert "Não adicione, remova ou altere" in repair_msgs[-1]["content"]


def test_falha_apos_reparo_propaga(monkeypatch):
    system = _system()
    system.system_prompt = "prompt"

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(200, {"message": {"content": "sempre inválido"}})

    monkeypatch.setattr("src.benchmark.systems.ollama_gemma4.requests.post", fake_post)
    with pytest.raises(OllamaError) as exc:
        system.extract("s1", "Sentença.")
    # o conteúdo bruto das duas tentativas fica registrado no erro
    assert "sempre inválido" in str(exc.value)


def test_erro_http_levanta_ollama_error(monkeypatch):
    system = _system()
    system.system_prompt = "prompt"
    system.think_supported = True

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(500, {"error": "boom"})

    monkeypatch.setattr("src.benchmark.systems.ollama_gemma4.requests.post", fake_post)
    with pytest.raises(OllamaError):
        system.extract("s1", "Sentença.")


def test_timeout_propaga(monkeypatch):
    import requests as _requests

    system = _system()
    system.system_prompt = "prompt"

    def fake_post(url, json=None, timeout=None):
        raise _requests.exceptions.Timeout("timeout simulado")

    monkeypatch.setattr("src.benchmark.systems.ollama_gemma4.requests.post", fake_post)
    with pytest.raises(_requests.exceptions.Timeout):
        system.extract("s1", "Sentença.")


def test_mudanca_de_digest_interrompe(monkeypatch, tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("prompt fixo", encoding="utf-8")
    system = _system(
        expected_digest="sha256:esperado",
        prompt={"path": str(prompt)},
        reproducibility={"require_digest": True, "fail_on_digest_change": True},
    )
    monkeypatch.setattr(system, "_get", lambda path: {"models": [], "version": "x"})
    monkeypatch.setattr(
        system, "resolve_model",
        lambda: setattr(system, "digest", "sha256:diferente") or {},
    )
    with pytest.raises(SystemUnavailable) as exc:
        system.setup()
    assert "digest" in str(exc.value).lower()


def test_fallback_think_nao_suportado(monkeypatch):
    """HTTP 400 na 1a tentativa com `think` reenvia sem o campo e registra."""
    system = _system()
    system.system_prompt = "prompt"
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        if "think" in json:
            return _FakeResponse(400, {"error": "unknown field think"})
        return _FakeResponse(200, {"message": {"content": '{"triples": []}'}})

    monkeypatch.setattr("src.benchmark.systems.ollama_gemma4.requests.post", fake_post)
    preds = system.extract("s1", "Sentença.")
    assert preds == []
    assert system.think_supported is False
    assert "think" not in calls[-1]
