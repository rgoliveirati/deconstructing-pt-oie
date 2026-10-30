"""Busca aberta de cabeças: ganho das 144 cabeças do BERTimbau, sem janela de camadas.

Usa ``bosque_heads_gain_roleaware``, a mesma função com que o extrator escolhe as
cabeças, com a janela aberta (camadas 1–12) para sujeito e objeto, sobre as 1.000
primeiras sentenças do UD Bosque. ``plot_head_search.py`` desenha a figura a partir
desta saída.

Uso:
    python scripts/head_search.py --output outputs/head_search/head_gains_open_search.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.runtime import write_json  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bert-model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--bosque", default=str(REPO / "data/pt_bosque-ud-train.conllu"))
    p.add_argument("--n-sent", type=int, default=1000)
    p.add_argument("--output", type=Path, default=REPO / "outputs/head_search/head_gains_open_search.json")
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModel, AutoTokenizer

    from src.extractor import bosque_heads_gain_roleaware, load_bosque

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    model = AutoModel.from_pretrained(args.bert_model, output_attentions=True, attn_implementation="eager")
    model.to(device).eval()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    pairs = load_bosque(args.bosque, max_sent=args.n_sent)
    gains = {role: [list(t) for t in bosque_heads_gain_roleaware(tokenizer, model, pairs, n_layers, n_heads,
                                                                  (1, n_layers), role, device)]
             for role in ("subj", "obj")}
    write_json(args.output, {"bert_model": args.bert_model, "bosque_path": Path(args.bosque).name,
                             "n_sent_rank": args.n_sent, "n_layers": n_layers, "n_heads": n_heads,
                             "gains": gains})
    for role, g in gains.items():
        print(role, "best", g[0])
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
