"""Construção do manifesto de execução."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .corpus import validate_corpus
from .runtime import collect_environment, git_state, now_iso


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_manifest(
    *,
    repo_dir: str,
    gold_path: str,
    prompt_path: str,
    seed: int,
    protocols: tuple[str, ...],
    systems_status: dict[str, Any],
    model_manifests: dict[str, Any],
    configs: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    corpus = validate_corpus(gold_path)
    return {
        "created_at": now_iso(),
        "started_at": started_at,
        "finished_at": now_iso(),
        "corpus": {
            "path": str(gold_path),
            "sha256": corpus["sha256"],
            "n_sentences": corpus["n_sentences"],
            "n_gold_triples": corpus["n_gold_triples"],
        },
        "git": git_state(repo_dir),
        "environment": collect_environment(),
        "seed": seed,
        "protocols": list(protocols),
        "matching_thresholds": {
            "tolerant_min_slot_f1": 0.70,
            "carb_style_threshold": 0.60,
            "carb_style_weights": {"arg1": 0.35, "rel": 0.30, "arg2": 0.35},
        },
        "prompt_sha256": file_sha256(prompt_path),
        "prompt_path": str(prompt_path),
        "systems": systems_status,
        "model_manifests": model_manifests,
        "system_configs": configs,
    }
