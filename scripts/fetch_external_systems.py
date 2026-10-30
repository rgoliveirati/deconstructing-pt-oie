"""Obtém os sistemas externos, implementações dos autores, em .external/ (idempotente).

Clona apenas os repositórios dos autores; não reimplementa sistemas nem baixa
artefatos de terceiros. Checkpoints que os autores distribuem fora do GitHub
(ex.: pesos do Multi²OIE) precisam ser colocados manualmente e apontados na
configuração YAML correspondente.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EXTERNAL = {
    "DptOIE": {
        "url": "https://github.com/FORMAS/DptOIE",
        "notes": "Sistema Java; construir com o build dos autores do repositório.",
    },
    "Multi2OIE": {
        "url": "https://github.com/youngbin-ro/Multi2OIE",
        "notes": "Checkpoint multilíngue distribuído pelos autores fora do GitHub; "
                 "colocar em .external/Multi2OIE e apontar em configs/systems/multi2oie.yaml.",
    },
    # PortNOIE: nenhum repositório dos autores público com artefato executável foi
    # localizado; não é clonado nada aqui. Ver system_availability.json.
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None)
    args = p.parse_args(argv)

    ext_dir = REPO / ".external"
    ext_dir.mkdir(exist_ok=True)
    report = []
    for name, spec in EXTERNAL.items():
        if args.only and name not in args.only:
            continue
        dest = ext_dir / name
        if dest.exists() and any(dest.iterdir()):
            commit = subprocess.run(
                ["git", "-C", str(dest), "rev-parse", "HEAD"],
                capture_output=True, text=True,
            ).stdout.strip()
            report.append({"system": name, "status": "present", "commit": commit})
            print(f"{name}: já presente ({commit[:12]})")
            continue
        print(f"{name}: clonando {spec['url']} ...")
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", spec["url"], str(dest)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            commit = subprocess.run(
                ["git", "-C", str(dest), "rev-parse", "HEAD"],
                capture_output=True, text=True,
            ).stdout.strip()
            report.append({"system": name, "status": "cloned", "commit": commit,
                           "url": spec["url"], "notes": spec["notes"]})
            print(f"{name}: ok ({commit[:12]}). {spec['notes']}")
        else:
            report.append({"system": name, "status": "fetch_failed",
                           "url": spec["url"], "error": proc.stderr[-500:]})
            print(f"{name}: FALHA — {proc.stderr[-200:]}", file=sys.stderr)

    out = REPO / "outputs/benchmark/cache/fetch_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"relatório: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
