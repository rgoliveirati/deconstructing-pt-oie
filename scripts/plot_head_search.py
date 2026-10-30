"""Figura da busca aberta de cabeças: ganho de cada uma das 144 cabeças do BERTimbau.

O ganho é o de ``bosque_heads_gain_roleaware`` (atenção verbo -> dependente menos a
média da matriz), calculado sem janela de camadas sobre 1.000 sentenças do UD Bosque.
As 10 melhores dentro das janelas fixas do extrator (sujeito 2–5, objeto 7–12) são
as cabeças que o ``rq3_attn_on_thr0`` usa.

Uso:
    python scripts/plot_head_search.py \
        --gains outputs/head_search/head_gains_open_search.json \
        --output paper/jiis/figures/head_search.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

WINDOWS = {"subj": (2, 5), "obj": (7, 12)}
TITLES = {"subj": "Subject evidence", "obj": "Object evidence"}
# Rampa sequencial azul (100 -> 700) e cinza neutro para ganho <= 0.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#f0efec"
INK, MUTED = "#1f1f1f", "#6b6b6b"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gains", default="outputs/head_search/head_gains_open_search.json")
    p.add_argument("--output", default="paper/jiis/figures/head_search.pdf")
    args = p.parse_args(argv)

    gains = json.loads(Path(args.gains).read_text(encoding="utf-8"))["gains"]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    cmap.set_under(NEUTRAL)
    vmax = max(g for role in gains.values() for _, _, g in role)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)
    for ax, role in zip(axes, ("subj", "obj")):
        grid = np.zeros((12, 12))
        for layer, head, g in gains[role]:
            grid[layer - 1, head - 1] = g
        im = ax.imshow(grid, cmap=cmap, vmin=1e-9, vmax=vmax, origin="upper", aspect="equal")

        ranked = sorted(gains[role], key=lambda x: -x[2])
        lo, hi = WINDOWS[role]
        selected = [x for x in ranked if lo <= x[0] <= hi][:10]
        open_top = ranked[:10]
        ax.add_patch(Rectangle((-0.5, lo - 1.5), 12, hi - lo + 1, fill=False, ec=INK, lw=1.4))
        ax.scatter([h - 1 for _, h, _ in selected], [l - 1 for l, _, _ in selected],
                   s=14, c=INK, lw=0, zorder=3, label="selected (top 10 in window)")
        ax.scatter([h - 1 for _, h, _ in open_top], [l - 1 for l, _, _ in open_top],
                   s=70, facecolors="none", edgecolors="#d9480f", lw=1.2, zorder=4,
                   label="top 10 of open search")
        best = ranked[0]
        # O rótulo vai para o lado com célula vizinha neutra, para não cobrir dado.
        right = grid[best[0] - 1, best[1]] <= 0 if best[1] < 12 else False
        ax.annotate(f"best: L{best[0]}H{best[1]}", (best[1] - 1, best[0] - 1),
                    xytext=(8 if right else -8, 1), textcoords="offset points", fontsize=7,
                    color=INK, va="center", ha="left" if right else "right")
        ax.set_title(f"{TITLES[role]} (window L{lo}–L{hi})", fontsize=9, color=INK)
        ax.set_xticks(range(12), [str(i) for i in range(1, 13)])
        ax.set_yticks(range(12), [f"L{i}" for i in range(1, 13)])
        ax.set_xlabel("head")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("layer")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, extend="min", pad=0.02)
    cb.set_label("gain (gray: gain ≤ 0)", color=MUTED)
    cb.outline.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
