"""Adaptador do PortNOIE (Cabral et al., 2022) — somente artefato dos autores.

Não é permitido construir uma versão aproximada. O adaptador procura um
artefato executável dos autores em ``.external/PortNOIE`` (código + modelo
treinado). Enquanto não houver artefato dos autores, o sistema é registrado como
``unavailable`` com a evidência da busca (ver
``outputs/benchmark/system_availability.json``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import Prediction
from .base import OpenIESystem, SystemUnavailable


class PortNOIESystem(OpenIESystem):
    name = "portnoie"

    def __init__(self, config: dict[str, Any]):
        self.external_dir = Path(config.get("external_dir", ".external/PortNOIE"))
        self.search_notes = config.get(
            "search_notes",
            "Busca (2026-07-12): o código dos autores existe em FORMAS/dptoie-neural "
            "(commit 770f29fe4a1f, clonado em .external/PortNOIE) e inclui um "
            "modelo treinado (saida_novo/model_final/model.th, formato AllenNLP). "
            "Porém o ambiente original de execução não é reconstituível de forma "
            "determinística: pyproject exige Python >=3.8,<3.10 com "
            "allennlp==2.7.0 (descontinuado, incompatível com o Python 3.12 do "
            "ambiente) e dependências git NÃO pinadas (sru@3.0.0-dev, "
            "flair@master); resolver essas branches hoje produziria um sistema "
            "diferente do publicado. Executar o modelo fora do stack original ou "
            "treinar substituto violaria as restrições do benchmark.",
        )

    def setup(self) -> None:
        raise SystemUnavailable(
            self.name,
            "ambiente de execução original do PortNOIE não é reconstituível "
            "(allennlp==2.7.0 exige Python <3.10; dependências git não pinadas); "
            "não é permitido construir substituto aproximado nem treinar novo modelo",
            evidence=self.search_notes,
        )

    def extract(self, sentence_id: str, sentence: str) -> list[Prediction]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"authors_artifact": None, "search_notes": self.search_notes}
