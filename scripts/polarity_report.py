"""Agrega as medições de fidelidade de polaridade e gera as tabelas do artigo.

Lê os JSON produzidos por ``scripts/polarity_fidelity.py`` (um por grupo de
corpora), soma os contadores por sistema e reporta a taxa agregada com intervalo
de confiança de Wilson — apropriado para proporções com amostra pequena, ao
contrário do intervalo normal, que produz limites fora de [0, 1] em taxas
extremas como 0 % e 100 %.

Saídas: CSV, LaTeX e um resumo em texto.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.runtime import now_iso, write_json  # noqa: E402

#: rótulo legível -> arquivo produzido por polarity_fidelity.py
DEFAULT_SOURCES = {
    "LREC (WIKI200+CETEN200+Gamalho)": "polarity_fidelity.json",
    "BIA + OIEC-PT": "bia_oiec.json",
    "BIOMED (PubMed pt)": "biomed.json",
}

DISPLAY = {"pt_oie": "PT-OIE-EXTRACTOR", "pt_oie_op1": "PT-OIE + Op.1",
           "ud_baseline": "PT-OIE (UD puro)", "ollama_gemma4": "Gemma 4 (8B)",
           "MultiOIE": "Multi²OIE", "PortOIE-LLama": "PortOIE-Llama"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """IC de Wilson para proporção. Bem definido em k=0 e k=n."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="outputs/polarity")
    p.add_argument("--kind", choices=["neg", "mod"], default="neg")
    args = p.parse_args(argv)

    base = REPO / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)
    kind = args.kind

    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"k": 0, "n": 0})
    per_corpus: dict[str, dict[str, tuple[int, int]]] = {}
    marked = {"neg": 0, "mod": 0}
    missing = []

    for label, fname in DEFAULT_SOURCES.items():
        path = base / fname
        if not path.exists():
            missing.append(fname)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        for k2 in ("neg", "mod"):
            marked[k2] += d["marked_predicates_found"].get(k2, 0)
        per_corpus[label] = {}
        for r in d["results"]:
            n = r[f"{kind}_covered_triples"]
            rate = r[f"{kind}_por_tripla_%"]
            if not n or rate is None:
                continue
            k = round(rate / 100 * n)
            totals[r["system"]]["k"] += k
            totals[r["system"]]["n"] += n
            per_corpus[label][r["system"]] = (k, n)

    if missing:
        print(f"aviso: fontes ausentes ({', '.join(missing)}) — "
              "rode scripts/polarity_fidelity.py para elas", file=sys.stderr)
    if not totals:
        print("ERRO: nenhuma medição encontrada", file=sys.stderr)
        return 2

    rows = []
    for system, t in totals.items():
        k, n = t["k"], t["n"]
        lo, hi = wilson(k, n)
        row = {"system": DISPLAY.get(system, system), "system_key": system,
               "preserved": k, "covered": n, "rate_pct": round(k / n * 100, 1),
               "ci_low": round(lo, 1), "ci_high": round(hi, 1)}
        for label in DEFAULT_SOURCES:
            cell = per_corpus.get(label, {}).get(system)
            row[label] = f"{cell[0]}/{cell[1]}" if cell else None
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("rate_pct", ascending=False)

    out_csv = base / f"aggregate_{kind}.csv"
    df.to_csv(out_csv, index=False)

    # ---------------------------------------------------------------- LaTeX
    lines = [
        "% Gerado por scripts/polarity_report.py — não editar à mão.",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Preserva\\c{c}\\~ao de negação: entre as triplas que cobrem um predicado "
        "negado no texto, propor\\c{c}\\~ao que mant\\'em o marcador no descritor de rela\\c{c}\\~ao. "
        "IC de Wilson a 95\\%.}",
        "\\label{tab:polarity}",
        "\\begin{tabular}{l r r l}", "\\toprule",
        "Sistema & Preservadas/Cobertas & \\% & IC95\\% \\\\", "\\midrule",
    ]
    for r in df.to_dict("records"):
        lines.append(f"{r['system']} & {r['preserved']}/{r['covered']} & "
                     f"{r['rate_pct']:.1f} & [{r['ci_low']:.0f}, {r['ci_high']:.0f}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (base / f"aggregate_{kind}.tex").write_text("\n".join(lines), encoding="utf-8")

    write_json(base / f"aggregate_{kind}.json", {
        "created_at": now_iso(), "kind": kind,
        "sources": {k: v for k, v in DEFAULT_SOURCES.items() if v not in missing},
        "marked_predicates_total": marked, "rows": rows,
    })

    label = "NEGAÇÃO" if kind == "neg" else "MODALIDADE"
    print(f"{label} — agregado ({marked[kind]} predicados marcados no texto)\n")
    hdr = f"{'sistema':<22}{'preserv/cobert':>16}{'taxa':>8}{'IC95':>14}   "
    hdr += " | ".join(f"{l.split(' ')[0]:>12}" for l in DEFAULT_SOURCES)
    print(hdr)
    for r in df.to_dict("records"):
        cells = " | ".join(f"{(r.get(l) or '—'):>12}" for l in DEFAULT_SOURCES)
        print(f"{r['system']:<22}{f'{r['preserved']}/{r['covered']}':>16}"
              f"{r['rate_pct']:7.1f}%{f'[{r['ci_low']:.0f}, {r['ci_high']:.0f}]':>14}   {cells}")
    print(f"\n-> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
