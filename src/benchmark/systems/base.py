"""Interface comum dos sistemas OpenIE do benchmark."""

from __future__ import annotations

import time
from typing import Any

from ..schemas import Prediction
from ..runtime import now_iso


class SystemUnavailable(Exception):
    """Sistema não executável neste ambiente.

    `reason` e `evidence` são registrados em system_availability.json;
    sistemas indisponíveis não recebem métricas (não valem zero).
    """

    def __init__(self, system: str, reason: str, evidence: str = ""):
        super().__init__(f"{system}: {reason}")
        self.system = system
        self.reason = reason
        self.evidence = evidence


class OpenIESystem:
    """Contrato mínimo dos adaptadores.

    `setup()` deve levantar SystemUnavailable quando o sistema não puder ser
    executado. `extract()` deve levantar exceções em caso de falha por
    sentença — o runner registra o erro e continua; nunca converta falha em
    lista vazia. Lista vazia significa, exclusivamente, "o sistema respondeu
    e não extraiu triplas".
    """

    name: str = "base"
    supports_batch: bool = False  # sistemas com custo fixo alto por processo

    def setup(self) -> None:
        raise NotImplementedError

    def extract(self, sentence_id: str, sentence: str) -> list[Prediction]:
        raise NotImplementedError

    def extract_batch(
        self, sentences: list[tuple[str, str]]
    ) -> dict[str, list[Prediction]]:
        """Extração em lote para sistemas cujo executável processa um arquivo
        de sentenças por invocação (ex.: DptOIE). Recebe [(sentence_id,
        sentence)] na ordem do corpus e retorna sentence_id -> predições.
        Sentenças ausentes do retorno são registradas como erro."""
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {}

    def teardown(self) -> None:
        pass


class Timer:
    def __enter__(self):
        self.started_at = now_iso()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.finished_at = now_iso()
        self.duration_seconds = time.perf_counter() - self._t0
        return False

    def runtime(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 6),
        }
