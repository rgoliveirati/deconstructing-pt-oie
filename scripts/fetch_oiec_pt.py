"""Baixa o OIEC-PT (Souza et al., 2025) do Zenodo para ``data/external/OIEC-PT``.

Registro 10.5281/zenodo.17438351, licença CC BY 4.0. O corpus não é redistribuído
neste repositório; cada arquivo é conferido pelo MD5 publicado no registro.
Usado só como corpus de desenvolvimento (auditoria das regras, Op.1, polaridade).

Uso:
    python scripts/fetch_oiec_pt.py
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = "https://zenodo.org/api/records/17438351/files/{}/content"
FILES = {
    "OIEC-PT.txt": "57f5e7cd893157789f55d2bdcd35b4e8",
    "OIEC-PT-GOLD.txt": "acd848f5f400b900d0ce115d4e031848",
    "OIEC-PT-SILVER.txt": "d2a95506a4748769d085c14331e15dc4",
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=REPO / "data/external/OIEC-PT")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        path = args.output_dir / name
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != md5:
            with urllib.request.urlopen(RECORD.format(name), timeout=120) as r:
                path.write_bytes(r.read())
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != md5:
            raise SystemExit(f"{name}: MD5 {got} != {md5}")
        print(f"ok  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
