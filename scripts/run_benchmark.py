"""Executa o benchmark: todos os sistemas sobre as mesmas 262 sentenças do BIA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.corpus import load_bia, validate_corpus  # noqa: E402
from src.benchmark.manifest import build_manifest  # noqa: E402
from src.benchmark.registry import build_system, load_yaml, system_config_path  # noqa: E402
from src.benchmark.runner import run_system  # noqa: E402
from src.benchmark.runtime import collect_environment, now_iso, write_json  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark comparativo OpenIE-PT sobre o BIA")
    p.add_argument("--config", default="configs/benchmark.yaml")
    p.add_argument("--systems", nargs="+", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="reexecuta ignorando saídas existentes")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--skip-unavailable", action="store_true", default=True)
    p.add_argument("--device", default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/benchmark"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    gold_path = str(REPO / cfg["gold_path"])
    system_names = args.systems or cfg.get("systems", [])
    started_at = now_iso()

    # 1. Corpus
    corpus_report = validate_corpus(gold_path)
    write_json(output_dir / "corpus_validation.json", corpus_report)
    if not corpus_report["ok"]:
        print("ERRO: corpus BIA fora do esperado:", corpus_report, file=sys.stderr)
        return 2
    sentences = load_bia(gold_path)
    print(f"corpus ok: {corpus_report['n_sentences']} sentenças, "
          f"{corpus_report['n_gold_triples']} triplas gold, sha256={corpus_report['sha256'][:12]}")

    write_json(output_dir / "environment.json", collect_environment())

    if args.dry_run:
        print("dry-run: sistemas selecionados:", system_names)
        return 0

    # 2. Digest do gemma4 antes da execução (exigência de reprodutibilidade)
    model_manifests: dict = {}
    if "ollama_gemma4" in system_names:
        gcfg = load_yaml(system_config_path(REPO / "configs", "ollama_gemma4"))
        from src.benchmark.systems.ollama_gemma4 import OllamaGemma4System
        from src.benchmark.systems.base import SystemUnavailable

        try:
            probe = OllamaGemma4System(gcfg)
            manifest = probe.resolve_model()
            model_manifests["gemma4_latest"] = manifest
            mpath = output_dir / "models" / "gemma4_latest_manifest.json"
            mpath.parent.mkdir(parents=True, exist_ok=True)
            write_json(mpath, manifest)
            print(f"gemma4:latest digest={manifest['digest']}")
        except (SystemUnavailable, Exception) as exc:  # registrado adiante no run
            print(f"aviso: inspeção do gemma4 falhou: {exc}", file=sys.stderr)

    # 3. Execução por sistema (falha de um não interrompe os demais).
    # Estado anterior é mesclado para permitir reexecução parcial (--systems X)
    # sem apagar o registro dos demais sistemas no manifesto.
    systems_status: dict = {}
    system_configs: dict = {}
    prev_manifest = output_dir / "manifest.json"
    if prev_manifest.exists():
        try:
            prev = json.loads(prev_manifest.read_text(encoding="utf-8"))
            systems_status.update(prev.get("systems", {}))
            system_configs.update(prev.get("system_configs", {}))
            model_manifests = {**prev.get("model_manifests", {}), **model_manifests}
        except Exception:
            pass
    for name in system_names:
        cfg_path = system_config_path(REPO / "configs", name)
        scfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        if args.timeout:
            scfg["timeout_seconds"] = args.timeout
        if name == "ollama_gemma4" and "gemma4_latest" in model_manifests:
            scfg["expected_digest"] = model_manifests["gemma4_latest"]["digest"]
        system_configs[name] = scfg
        print(f"\n=== {name} ===")
        try:
            system = build_system(name, scfg)
            result = run_system(
                system,
                sentences,
                output_dir,
                resume=args.resume and not args.force,
                fail_fast=args.fail_fast,
                limit=args.limit,
            )
            systems_status[name] = result.to_dict()
            print(json.dumps(
                {k: systems_status[name][k]
                 for k in ("status", "reason", "n_ok", "n_error", "n_triples")},
                ensure_ascii=False, default=str))
        except Exception as exc:
            import traceback

            systems_status[name] = {
                "system": name,
                "status": "runner_error",
                "reason": f"{type(exc).__name__}: {exc}",
                "evidence": traceback.format_exc()[-2000:],
            }
            print(f"ERRO no runner de {name}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    # 4. Disponibilidade e manifesto
    availability = [
        {
            "system": name,
            "status": (
                "unavailable"
                if st.get("status") in ("unavailable", "setup_error")
                else st.get("status")
            ),
            "reason": st.get("reason"),
            "evidence": st.get("evidence"),
            "checked_at": st.get("finished_at") or now_iso(),
        }
        for name, st in systems_status.items()
    ]
    write_json(output_dir / "system_availability.json", availability)

    manifest = build_manifest(
        repo_dir=str(REPO),
        gold_path=gold_path,
        prompt_path=str(REPO / "configs/gemma4_prompt_pt.txt"),
        seed=seed,
        protocols=tuple(cfg.get("protocols", [])),
        systems_status=systems_status,
        model_manifests=model_manifests,
        configs=system_configs,
        started_at=started_at,
    )
    write_json(output_dir / "manifest.json", manifest)
    print(f"\nmanifesto: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
