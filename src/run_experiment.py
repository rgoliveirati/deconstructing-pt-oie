"""Command line entry point for PT OIE EXTRACTOR experiments."""

from __future__ import annotations

from typing import Optional, Sequence

from .extractor import Config, run_experiment as _run_experiment

run_experiment = _run_experiment


def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Run PT OIE EXTRACTOR experiment")
    p.add_argument("--gold", required=True, help="JSONL gold file with the field 'sentence'")
    p.add_argument("--output-dir", required=True, help="Directory where outputs will be written")
    p.add_argument("--dataset-name", default="dataset")
    p.add_argument("--bert-model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--bosque", default=None, help="UD Portuguese Bosque .conllu file used for attention head ranking")
    p.add_argument("--heads-mode", default="rank", choices=["rank", "all", "forced", "random"])
    p.add_argument("--top-k-heads", type=int, default=10)
    p.add_argument("--attn-threshold", type=float, default=0.0)
    p.add_argument("--no-attn", action="store_true")
    p.add_argument("--theory-mode", choices=["off", "annotate", "filter"], default="filter")
    p.add_argument("--s2-span-policy", choices=["minimal", "extensive", "both"], default="both")
    p.add_argument("--strict-theory", action="store_true")
    p.add_argument("--block-reported-belief", action="store_true")
    p.add_argument("--no-e8-clausal", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = Config(
        bert_model=args.bert_model,
        bosque_path=args.bosque,
        heads_mode=args.heads_mode,
        top_k_heads=args.top_k_heads,
        attn_threshold=args.attn_threshold,
        no_attn=args.no_attn,
        theory_mode=args.theory_mode,
        s2_span_policy=args.s2_span_policy,
        strict_theory=args.strict_theory,
        s1_block_reported_belief=args.block_reported_belief,
        e8_allow_clausal_args=not args.no_e8_clausal,
    )
    run_experiment(
        config=config,
        gold_path=args.gold,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
