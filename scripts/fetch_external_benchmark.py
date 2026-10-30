"""Obtém o benchmark de OpenIE-PT de Silva et al. (LREC 2026) e registra proveniência.

Clona o repositório do PT-OIE-Benchmark em ``.external/PT-OIE-Benchmark`` (diretório já
ignorado pelo git, como os demais sistemas externos) e grava
``outputs/external_benchmark/provenance.json`` com URL, commit, licença,
contagens e SHA-256 de cada arquivo utilizado.

Os corpora NÃO são recriados nem corrigidos: usamos os arquivos distribuídos pelo benchmark.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.external.corpora import validate_datasets  # noqa: E402
from src.benchmark.registry import load_yaml  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def clone_or_update(url: str, dest: Path, commit: str | None) -> dict[str, str]:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"clonando {url} -> {dest}")
        subprocess.run(["git", "clone", url, str(dest)], check=True)
    else:
        print(f"clone já existe: {dest}")

    if commit:
        print(f"fixando commit {commit}")
        subprocess.run(["git", "checkout", commit], cwd=dest, check=True)

    return {
        "url": url,
        "commit": _git(["rev-parse", "HEAD"], dest),
        "commit_date": _git(["log", "-1", "--format=%cI"], dest),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], dest),
    }


def detect_license(bench_dir: Path) -> str:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        p = bench_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
    return "não declarada no repositório (verificar com os autores antes de redistribuir)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/external_benchmark.yaml")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    bench_cfg = cfg["benchmark_repo"]
    dest = REPO / bench_cfg["clone_dir"]
    out_dir = Path(args.output_dir or (REPO / cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    git_info = clone_or_update(bench_cfg["url"], dest, bench_cfg.get("commit"))
    report = validate_datasets(dest, [d["name"] for d in cfg["datasets"]])

    provenance = {
        "fetched_at": now_iso(),
        "paper": {
            "title": "Benchmarking Portuguese Open Information Extraction",
            "authors": "Silva, Rodrigues, Teixeira, Amorim",
            "venue": "LREC 2026, pages 7692-7700",
        },
        "repository": git_info,
        "license": detect_license(dest),
        "evaluator": {
            "path": str((dest / "Evaluation.py").relative_to(REPO)),
            "sha256": __import__("hashlib").sha256((dest / "Evaluation.py").read_bytes()).hexdigest(),
            "note": "avaliador do PT-OIE-Benchmark, usado sem modificação",
        },
        "datasets": report["datasets"],
        "total_sentences": report["total_sentences"],
        "total_gold_triples": report["total_gold_triples"],
        "conversion": {
            "performed": False,
            "note": "nenhuma conversão: os arquivos distribuídos já estão no formato "
                    "TSV Sentence/ARG0/V/ARG1 exigido pelo avaliador",
        },
    }
    write_json(out_dir / "provenance.json", provenance)

    print(f"\ncommit: {git_info['commit']}")
    print(f"licença: {provenance['license']}")
    for name, d in report["datasets"].items():
        flag = "ok" if d["ok"] else "DIVERGE"
        print(f"  {name:<10} {d['n_sentences']:>4} sentenças  {d['n_gold_triples']:>4} gold  "
              f"[{flag}] gold_sha256={d['gold_sha256'][:12]}")
        for w in d["warnings"]:
            print(f"      aviso: {w}")
    print(f"\ntotal: {report['total_sentences']} sentenças, {report['total_gold_triples']} triplas gold")
    print(f"proveniência -> {out_dir / 'provenance.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
