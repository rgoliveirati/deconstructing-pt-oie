"""Gera as tabelas (CSV/LaTeX) e o relatório da avaliação externa.

Distingue explicitamente a proveniência de cada número:
  reproduced  — pontuado localmente com o avaliador do PT-OIE-Benchmark sobre as saídas
                por sentença distribuídas pelos autores;
  paper       — copiado da Tabela 5 do artigo;
  no_output   — o repositório do PT-OIE-Benchmark não traz saída do sistema para o corpus;
  n/a         — não aplicável.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.registry import load_yaml  # noqa: E402
from src.benchmark.runtime import now_iso, write_json  # noqa: E402

DATASETS = ["WIKI200", "CETEN200", "Gamalho"]

#: Tabela 5 do artigo (P, R, F1 por corpus). Transcrita, não recalculada.
PAPER_TABLE5: dict[str, dict[str, tuple[float, float, float] | None]] = {
    "PragmaticOIE":  {"WIKI200": (74.2, 82.0, 77.9), "CETEN200": (72.7, 80.4, 76.3), "Gamalho": (37.0, 42.6, 39.6)},
    "DptOIE":        {"WIKI200": (23.5, 47.6, 31.4), "CETEN200": (19.7, 39.5, 26.3), "Gamalho": (0.0, 0.0, 0.0)},
    "TransAlign":    {"WIKI200": (77.6, 40.2, 53.0), "CETEN200": (65.1, 37.1, 47.3), "Gamalho": (71.5, 35.2, 47.2)},
    "PortOIE-LLama": {"WIKI200": (9.4, 53.3, 16.0),  "CETEN200": (27.7, 35.9, 31.3), "Gamalho": (39.5, 25.3, 30.8)},
    "MultiOIE":      {"WIKI200": (52.5, 50.7, 51.6), "CETEN200": (46.2, 49.7, 47.9), "Gamalho": (55.1, 52.3, 53.7)},
    "DetIE":         {"WIKI200": (53.1, 51.0, 52.0), "CETEN200": (44.0, 47.6, 45.7), "Gamalho": (50.4, 50.5, 50.4)},
    "GraphIE":       {"WIKI200": (35.6, 37.0, 36.3), "CETEN200": (35.1, 33.6, 34.3), "Gamalho": (44.9, 37.4, 40.8)},
    "ArgOE":         {"WIKI200": (26.5, 19.6, 22.5), "CETEN200": (34.4, 40.0, 37.0), "Gamalho": (34.6, 18.1, 23.7)},
}

DISPLAY = {"pt_oie": "PT-OIE-EXTRACTOR", "ud_baseline": "PT-OIE (UD puro, RQ1)"}


def build_tables(cfg, out_dir: Path):
    metrics = json.loads((out_dir / "evaluations/metrics.json").read_text(encoding="utf-8"))
    ci = pd.read_csv(out_dir / "bootstrap/confidence_intervals.csv")
    ci_idx = {(r.dataset, r.system): r for r in ci.itertuples()}

    ours = list(cfg["systems"])
    rows = []
    for system in ours + list(cfg["reference_systems"]):
        for ds in DATASETS:
            entry = metrics["systems"][system]["datasets"].get(ds, {})
            paper = PAPER_TABLE5.get(system, {}).get(ds)
            rec = {
                "system": DISPLAY.get(system, system),
                "system_key": system,
                "dataset": ds,
                "is_ours": system in ours,
            }
            if entry.get("status") == "ok":
                c = ci_idx.get((ds, system))
                rec.update({
                    "provenance": "reproduced",
                    "precision": round(entry["precision"] * 100, 1),
                    "recall": round(entry["recall"] * 100, 1),
                    "f1": round(entry["f1_score"] * 100, 1),
                    "f1_ci_low": round(c.f1_ci_low, 1) if c is not None else None,
                    "f1_ci_high": round(c.f1_ci_high, 1) if c is not None else None,
                    "n_pred": entry["total_extractions"],
                    "extractions_per_sentence": round(entry["extractions_per_sentence"], 2),
                    "pct_sentences_without_extraction": round(entry["pct_sentences_without_extraction"], 1),
                })
            else:
                rec.update({"provenance": "no_output", "precision": None, "recall": None,
                            "f1": None, "f1_ci_low": None, "f1_ci_high": None, "n_pred": None,
                            "extractions_per_sentence": None,
                            "pct_sentences_without_extraction": None})
            rec["paper_precision"] = paper[0] if paper else None
            rec["paper_recall"] = paper[1] if paper else None
            rec["paper_f1"] = paper[2] if paper else None
            rec["paper_provenance"] = "paper" if paper else "n/a"
            if paper and rec["f1"] is not None:
                rec["f1_abs_diff_vs_paper"] = round(abs(rec["f1"] - paper[2]), 1)
            else:
                rec["f1_abs_diff_vs_paper"] = None
            rows.append(rec)

    df = pd.DataFrame(rows)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "comparison_full.csv", index=False)

    # Tabela principal: F1 por corpus + média (média só quando há os 3 corpora)
    wide = df.pivot(index="system", columns="dataset", values="f1")
    wide = wide.reindex(columns=DATASETS).reset_index()
    meta = df[["system", "system_key", "is_ours"]].drop_duplicates("system")
    wide = meta.merge(wide, on="system", how="right")
    wide["average"] = wide[DATASETS].mean(axis=1, skipna=False).round(1)
    wide = wide.sort_values("average", ascending=False, na_position="last")
    wide.to_csv(tables_dir / "comparison_f1.csv", index=False)
    return df, wide, metrics


def latex_table(df: pd.DataFrame, wide: pd.DataFrame) -> str:
    lines = [
        "% Gerado por scripts/report_external_benchmark.py — não editar à mão.",
        "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{PT-OIE-EXTRACTOR e os sistemas de \\citet{silva2026benchmarking} nos mesmos "
        "tr\\^es corpora, pontuados com o avaliador do PT-OIE-Benchmark. Todos os valores "
        "foram reproduzidos localmente. Intervalos de confian\\c{c}a e compara\\c{c}\\~oes pareadas "
        "(bootstrap por senten\\c{c}a, 10.000 r\\'eplicas, seed 42) est\\~ao no texto. "
        "$\\dagger$ = sem sa\\'ida no reposit\\'orio do PT-OIE-Benchmark.}",
        "\\label{tab:external_benchmark}",
        "\\begin{tabular}{l rrr rrr rrr r}", "\\toprule",
        "& \\multicolumn{3}{c}{WIKI200} & \\multicolumn{3}{c}{CETEN200} & "
        "\\multicolumn{3}{c}{Gamalho} & \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Sistema & P & R & $F_1$ & P & R & $F_1$ & P & R & $F_1$ & M\\'edia \\\\",
        "\\midrule",
    ]
    idx = {(r.system, r.dataset): r for r in df.itertuples()}
    for w in wide.itertuples():
        cells = []
        for ds in DATASETS:
            r = idx.get((w.system, ds))
            if r is None or r.provenance != "reproduced":
                cells += ["$\\dagger$", "$\\dagger$", "$\\dagger$"]
            else:
                cells += [f"{r.precision:.1f}", f"{r.recall:.1f}", f"{r.f1:.1f}"]
        avg = "--" if pd.isna(w.average) else f"{w.average:.1f}"
        name = f"\\textbf{{{w.system}}}" if w.is_ours else w.system
        lines.append(f"{name} & " + " & ".join(cells) + f" & {avg} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/external_benchmark.yaml")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args(argv)

    cfg = load_yaml(REPO / args.config)
    out_dir = Path(args.output_dir or (REPO / cfg["output_dir"]))
    df, wide, metrics = build_tables(cfg, out_dir)

    tables_dir = out_dir / "tables"
    (tables_dir / "comparison.tex").write_text(latex_table(df, wide), encoding="utf-8")

    # Divergências entre nossa reprodução e a tabela publicada.
    rep = df[(df.provenance == "reproduced") & df.f1_abs_diff_vs_paper.notna()]
    diverging = rep[rep.f1_abs_diff_vs_paper > 0.15][
        ["system", "dataset", "f1", "paper_f1", "f1_abs_diff_vs_paper"]
    ]
    diverging.to_csv(tables_dir / "reproduction_divergences.csv", index=False)

    write_json(out_dir / "tables/summary.json", {
        "created_at": now_iso(),
        "n_rows": len(df),
        "n_reproduced": int((df.provenance == "reproduced").sum()),
        "n_no_output": int((df.provenance == "no_output").sum()),
        "max_abs_diff_vs_paper": float(rep.f1_abs_diff_vs_paper.max()),
        "n_diverging_gt_0_15": int(len(diverging)),
    })

    print(wide[["system", *DATASETS, "average"]].to_string(index=False))
    print(f"\nreprodução vs. artigo: diferença absoluta máxima em F1 = "
          f"{rep.f1_abs_diff_vs_paper.max():.1f} ponto(s)")
    if len(diverging):
        print("linhas divergentes (> 0.15):")
        print(diverging.to_string(index=False))
    print(f"\ntabelas -> {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
