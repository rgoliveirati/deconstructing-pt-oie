"""Adaptador do Gemma 4 via API nativa do Ollama (/api/chat).

Regras: zero-shot, prompt fixo (SHA-256 registrado), saída estruturada por
JSON Schema, temperatura 0, seed 42, uma chamada principal por sentença e no
máximo uma tentativa de reparo — permitida somente para JSON inválido e
restrita à conversão da resposta original para o schema, sem consultar o
gold nem alterar conteúdo semântico. Falha nunca é convertida em lista
vazia. O digest de ``gemma4:latest`` é resolvido no setup e a execução é
interrompida caso mude (``fail_on_digest_change``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from ..schemas import Prediction
from ..runtime import now_iso
from .base import OpenIESystem, SystemUnavailable

TRIPLES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"},
                    "rel": {"type": "string"},
                    "arg2": {"type": "string"},
                },
                "required": ["arg1", "rel", "arg2"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}

REPAIR_INSTRUCTION = (
    "A resposta anterior não é um JSON válido no schema exigido. "
    "Converta exatamente o conteúdo da resposta anterior para o schema "
    '{"triples": [{"arg1": "...", "rel": "...", "arg2": "..."}]}. '
    "Não adicione, remova ou altere triplas. Não inclua texto fora do JSON."
)


class OllamaError(Exception):
    pass


class OllamaGemma4System(OpenIESystem):
    name = "ollama_gemma4"

    def __init__(self, config: dict[str, Any]):
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = config.get("model", "gemma4:latest")
        self.timeout = float(config.get("timeout_seconds", 300))
        gen = config.get("generation", {})
        self.options = {
            "temperature": float(gen.get("temperature", 0.0)),
            "seed": int(gen.get("seed", 42)),
            "num_predict": int(gen.get("num_predict", 384)),
            "num_ctx": int(gen.get("num_ctx", 8192)),
            "top_k": int(gen.get("top_k", 1)),
            "top_p": float(gen.get("top_p", 1.0)),
        }
        self.keep_alive = gen.get("keep_alive", "10m")
        self.think = bool(gen.get("think", False))
        so = config.get("structured_output", {})
        self.structured = bool(so.get("enabled", True))
        self.max_repair_attempts = int(so.get("max_repair_attempts", 1))
        self.prompt_path = config.get("prompt", {}).get("path", "configs/gemma4_prompt_pt.txt")
        repro = config.get("reproducibility", {})
        self.require_digest = bool(repro.get("require_digest", True))
        self.fail_on_digest_change = bool(repro.get("fail_on_digest_change", True))

        self.system_prompt = ""
        self.prompt_sha256 = ""
        self.digest: str | None = None
        self.ollama_version: str | None = None
        self.model_details: dict[str, Any] = {}
        self.think_supported: bool | None = None
        self.expected_digest: str | None = config.get("expected_digest")

    # -- descoberta do modelo -------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def resolve_model(self) -> dict[str, Any]:
        tags = self._get("/api/tags")
        entry = next(
            (m for m in tags.get("models", []) if m.get("name") == self.model), None
        )
        if entry is None:
            raise SystemUnavailable(
                self.name,
                f"modelo {self.model} não encontrado no Ollama",
                evidence=json.dumps([m.get("name") for m in tags.get("models", [])]),
            )
        try:
            self.ollama_version = self._get("/api/version").get("version")
        except Exception:
            self.ollama_version = None
        show: dict[str, Any] = {}
        try:
            resp = requests.post(
                f"{self.base_url}/api/show", json={"model": self.model}, timeout=30
            )
            resp.raise_for_status()
            show = resp.json()
            show.pop("tensors", None)  # volumoso e irrelevante para o manifesto
        except Exception as exc:
            show = {"error": str(exc)}
        manifest = {
            "checked_at": now_iso(),
            "name": entry.get("name"),
            "model": entry.get("model"),
            "digest": entry.get("digest"),
            "size": entry.get("size"),
            "modified_at": entry.get("modified_at"),
            "details": entry.get("details"),
            "ollama_version": self.ollama_version,
            "show": {
                k: show.get(k)
                for k in ("details", "model_info", "capabilities", "parameters",
                           "template", "license")
                if k in show
            },
        }
        self.digest = entry.get("digest")
        self.model_details = manifest
        return manifest

    # -- ciclo de vida ---------------------------------------------------------

    def setup(self) -> None:
        try:
            self._get("/api/tags")
        except Exception as exc:
            raise SystemUnavailable(
                self.name, f"servidor Ollama inacessível em {self.base_url}: {exc}"
            )
        manifest = self.resolve_model()
        if self.require_digest and not self.digest:
            raise SystemUnavailable(self.name, "digest do modelo não pôde ser obtido")
        if (
            self.expected_digest
            and self.fail_on_digest_change
            and self.digest != self.expected_digest
        ):
            raise SystemUnavailable(
                self.name,
                "digest de gemma4:latest mudou desde a inspeção registrada; "
                "não é permitido misturar resultados de digests diferentes",
                evidence=f"esperado={self.expected_digest} atual={self.digest}",
            )
        prompt_file = Path(self.prompt_path)
        if not prompt_file.exists():
            raise SystemUnavailable(self.name, f"prompt fixo ausente: {prompt_file}")
        raw = prompt_file.read_bytes()
        self.system_prompt = raw.decode("utf-8")
        self.prompt_sha256 = hashlib.sha256(raw).hexdigest()
        _ = manifest

    # -- chamada ---------------------------------------------------------------

    def _chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": dict(self.options),
            "keep_alive": self.keep_alive,
        }
        if self.structured:
            payload["format"] = TRIPLES_JSON_SCHEMA
        if self.think_supported is not False:
            payload["think"] = self.think
        resp = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        if resp.status_code == 400 and self.think_supported is None:
            # Versões antigas do Ollama rejeitam o campo `think`: registrar a
            # incompatibilidade e repetir a mesma requisição sem o campo.
            self.think_supported = False
            payload.pop("think", None)
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
        if resp.status_code != 200:
            raise OllamaError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        if self.think_supported is None:
            self.think_supported = True
        return resp.json()

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        """Remove cerca markdown (```json ... ```), quando presente.

        Tratamento puramente sintático de formato: o conteúdo interno não é
        alterado. Necessário porque o template do gemma4 no Ollama 0.20 emite
        a resposta cercada mesmo com `format` (JSON Schema) na requisição.
        A resposta bruta é preservada em raw_output.
        """
        text = content.strip()
        if text.startswith("```"):
            first_nl = text.find("\n")
            if first_nl != -1 and text.rstrip().endswith("```"):
                inner = text[first_nl + 1:]
                inner = inner.rstrip()
                if inner.endswith("```"):
                    inner = inner[: -3]
                return inner.strip()
        return content

    @classmethod
    def _parse_triples(cls, content: str) -> list[dict[str, str]]:
        data = json.loads(cls._strip_markdown_fence(content))
        if not isinstance(data, dict) or "triples" not in data:
            raise ValueError("JSON sem campo 'triples'")
        triples = data["triples"]
        if not isinstance(triples, list):
            raise ValueError("'triples' não é lista")
        out = []
        for t in triples:
            if not isinstance(t, dict) or not all(
                isinstance(t.get(k), str) for k in ("arg1", "rel", "arg2")
            ):
                raise ValueError(f"tripla fora do schema: {t!r}")
            out.append({"arg1": t["arg1"], "rel": t["rel"], "arg2": t["arg2"]})
        return out

    def extract(self, sentence_id: str, sentence: str) -> list[Prediction]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": sentence},
        ]
        response = self._chat(messages)
        content = (response.get("message") or {}).get("content", "")
        repaired_response: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            triples = self._parse_triples(content)
        except (ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
            if self.max_repair_attempts < 1:
                raise OllamaError(f"JSON inválido sem reparo permitido: {exc}")
            repair_messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": REPAIR_INSTRUCTION},
            ]
            repaired_response = self._chat(repair_messages)
            repaired_content = (repaired_response.get("message") or {}).get("content", "")
            try:
                triples = self._parse_triples(repaired_content)
            except (ValueError, json.JSONDecodeError) as exc2:
                raise OllamaError(
                    "JSON inválido após reparo único "
                    f"(done_reason={response.get('done_reason')!r}/"
                    f"{repaired_response.get('done_reason')!r}): {exc2}; "
                    f"conteúdo original: {content[:300]!r}; "
                    f"conteúdo reparado: {repaired_content[:300]!r}"
                ) from exc2

        def _ollama_metrics(r: dict[str, Any]) -> dict[str, Any]:
            return {
                k: r.get(k)
                for k in ("total_duration", "load_duration", "prompt_eval_count",
                           "prompt_eval_duration", "eval_count", "eval_duration",
                           "done_reason")
            }

        shared_meta = {
            "model": self.model,
            "digest": self.digest,
            "ollama_version": self.ollama_version,
            "quantization": (self.model_details.get("details") or {}).get("quantization_level"),
            "prompt_sha256": self.prompt_sha256,
            "generation_options": dict(self.options),
            "think": self.think if self.think_supported else None,
            "think_supported": self.think_supported,
            "structured_output": self.structured,
            "repaired": repaired_response is not None,
            "parse_error_before_repair": parse_error,
            "ollama_metrics": _ollama_metrics(response),
            "ollama_metrics_repair": (
                _ollama_metrics(repaired_response) if repaired_response else None
            ),
        }
        raw = {
            "response_content": content,
            "repaired_content": (
                (repaired_response.get("message") or {}).get("content")
                if repaired_response
                else None
            ),
        }
        preds = []
        for i, t in enumerate(triples):
            preds.append(
                Prediction(
                    sentence_id=sentence_id,
                    sentence=sentence,
                    system=self.name,
                    triple_id=f"{sentence_id}_{self.name}_{i:03d}",
                    arg1=t["arg1"],
                    rel=t["rel"],
                    arg2=t["arg2"],
                    confidence=None,
                    raw_output=raw,
                    metadata=shared_meta,
                )
            )
        if not preds:
            # Lista vazia explícita e válida: retornada pelo modelo no schema.
            self.last_empty_meta = {"raw_output": raw, "metadata": shared_meta}
        return preds

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "base_url": self.base_url,
            "model": self.model,
            "digest": self.digest,
            "ollama_version": self.ollama_version,
            "model_details": self.model_details,
            "prompt_path": str(self.prompt_path),
            "prompt_sha256": self.prompt_sha256,
            "generation_options": dict(self.options),
            "keep_alive": self.keep_alive,
            "think": self.think,
            "think_supported": self.think_supported,
            "structured_output": self.structured,
            "max_repair_attempts": self.max_repair_attempts,
            "zero_shot": True,
            "training_required": False,
        }
