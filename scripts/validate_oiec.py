"""Valida o OIEC-PT e comprova a semântica da flag de validade.

Reproduz a Tabela 2 de Souza et al. (2025) a partir dos arquivos distribuídos.
Se as contagens baterem nos dois subconjuntos independentemente, a coluna de
validade está corretamente identificada.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.oiec import PAPER_TABLE2, validate_oiec  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402

DEFAULT_DIR = REPO / "data/external/OIEC-PT"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", default=str(DEFAULT_DIR))
    p.add_argument("--output", default="outputs/dev_oiec/corpus_validation.json")
    args = p.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        print(f"ERRO: diretório do OIEC-PT não encontrado: {corpus_dir}", file=sys.stderr)
        return 2

    reports = {}
    ok = True
    print(f"{'arquivo':<22}{'sent':>6}{'cand':>7}{'validos':>9}{'rejeit':>8}"
          f"{'val/sent':>10}{'artigo':>8}{'':>4}")
    for name in PAPER_TABLE2:
        path = corpus_dir / name
        if not path.exists():
            print(f"{name:<22} AUSENTE")
            ok = False
            continue
        r = validate_oiec(path)
        reports[name] = r
        flag = "ok" if r["ok"] else "DIVERGE"
        if not r["ok"]:
            ok = False
        print(f"{name:<22}{r['n_sentences']:>6}{r['n_candidate_triples']:>7}"
              f"{r['n_valid_triples']:>9}{r['n_rejected_triples']:>8}"
              f"{r['valid_per_sentence']:>10.2f}{r['expected_valid']:>8}{flag:>8}")

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"created_at": now_iso(), "corpus_dir": str(corpus_dir),
                     "validity_flag_index": 1, "files": reports, "ok": ok})
    print(f"\n{'CONFIRMADO' if ok else 'FALHOU'}: a flag de índice 1 (2ª das três) marca validade.")
    print(f"relatório -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
