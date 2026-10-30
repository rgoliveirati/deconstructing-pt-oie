"""Inspeciona um modelo do Ollama e grava o manifesto (nome, tag, digest, etc.)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark.registry import load_yaml, system_config_path  # noqa: E402
from src.benchmark.systems.ollama_gemma4 import OllamaGemma4System  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma4:latest")
    p.add_argument("--base-url", default=None)
    p.add_argument("--output", default="outputs/benchmark/models/gemma4_latest_manifest.json")
    args = p.parse_args(argv)

    cfg = load_yaml(system_config_path("configs", "ollama_gemma4"))
    cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url
    system = OllamaGemma4System(cfg)
    manifest = system.resolve_model()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("name", "digest", "size", "modified_at", "ollama_version")}, indent=2))
    print(f"manifesto salvo em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
