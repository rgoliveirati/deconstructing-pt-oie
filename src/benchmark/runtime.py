"""Captura de ambiente e utilitários de tempo."""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import platform
import subprocess
from typing import Any


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _pkg_version(name: str) -> str | None:
    try:
        return getattr(importlib.import_module(name), "__version__", None)
    except Exception:
        return None


def _read_first_match(path: str, prefix: str) -> str | None:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _nvidia_smi() -> dict[str, Any] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, mem, driver = [x.strip() for x in out.stdout.strip().split(",")[:3]]
            return {"name": name, "memory_total": mem, "driver": driver}
    except Exception:
        pass
    return None


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "collected_at": now_iso(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _read_first_match("/proc/cpuinfo", "model name"),
        "ram_total": _read_first_match("/proc/meminfo", "MemTotal"),
        "gpu": _nvidia_smi(),
        "packages": {
            name: _pkg_version(name)
            for name in ("torch", "transformers", "stanza", "numpy",
                          "pandas", "conllu", "requests", "yaml")
        },
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_torch"] = torch.cuda.get_device_name(0)
    except Exception:
        env["cuda_available"] = None
    return env


def git_state(repo_dir: str) -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", repo_dir, *args],
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def write_json(path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
