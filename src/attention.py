"""Attention utilities for PT OIE EXTRACTOR.

The functions are imported from the validated experimental implementation in
`extractor.py` to preserve the published behavior while exposing a clearer module boundary.
"""

from .extractor import (
    _calc_offsets_in_text,
    _map_tokens_to_wp_simple,
    attn_score_pair,
    average_selected_heads,
    load_bosque,
    bosque_heads_gain_roleaware,
    aggregate_head_usage,
    parse_forced_heads,
    validate_heads,
)

__all__ = [
    "_calc_offsets_in_text",
    "_map_tokens_to_wp_simple",
    "attn_score_pair",
    "average_selected_heads",
    "load_bosque",
    "bosque_heads_gain_roleaware",
    "aggregate_head_usage",
    "parse_forced_heads",
    "validate_heads",
]
