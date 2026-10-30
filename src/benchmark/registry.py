"""Registro dos sistemas do benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SYSTEM_NAMES = (
    "pt_oie", "pt_oie_op1", "ud_baseline", "dptoie", "multi2oie", "portnoie",
    "ollama_gemma4",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_system(name: str, config: dict[str, Any]):
    if name == "pt_oie":
        from .systems.pt_oie import PTOIESystem

        return PTOIESystem(config)
    if name == "pt_oie_op1":
        from .systems.pt_oie import PTOIEMainClauseSystem

        return PTOIEMainClauseSystem(config)
    if name == "ud_baseline":
        from .systems.pt_oie import UDBaselineSystem

        return UDBaselineSystem(config)
    if name == "dptoie":
        from .systems.dptoie import DptOIESystem

        return DptOIESystem(config)
    if name == "multi2oie":
        from .systems.multi2oie import Multi2OIESystem

        return Multi2OIESystem(config)
    if name == "portnoie":
        from .systems.portnoie import PortNOIESystem

        return PortNOIESystem(config)
    if name == "ollama_gemma4":
        from .systems.ollama_gemma4 import OllamaGemma4System

        return OllamaGemma4System(config)
    raise ValueError(f"sistema desconhecido: {name}")


def system_config_path(configs_dir: str | Path, name: str) -> Path:
    return Path(configs_dir) / "systems" / f"{name}.yaml"
