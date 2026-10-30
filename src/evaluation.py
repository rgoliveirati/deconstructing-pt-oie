"""Evaluation and reporting utilities for PT OIE EXTRACTOR."""

from .extractor import (
    args_match,
    rels_match,
    triple_matches,
    evaluate_dataset,
    flatten_pred_rows,
    flatten_gold_rows,
    build_triples_table,
    build_triples_table_legacy,
    jaccard_accuracy,
    canon_arg,
    canon_rel,
    write_csv,
    write_jsonl,
    read_gold_jsonl,
)

__all__ = [
    "args_match",
    "rels_match",
    "triple_matches",
    "evaluate_dataset",
    "flatten_pred_rows",
    "flatten_gold_rows",
    "build_triples_table",
    "build_triples_table_legacy",
    "jaccard_accuracy",
    "canon_arg",
    "canon_rel",
    "write_csv",
    "write_jsonl",
    "read_gold_jsonl",
]
