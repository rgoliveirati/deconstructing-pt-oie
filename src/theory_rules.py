"""Theoretical validation data structures and rule summaries.

Rule implementations are methods of `OpenIEExtractorBIA` in `extractor.py`. This module
exposes the rule related dataclasses and summary helpers as a stable import surface.
"""

from .extractor import (
    CandidateTriple,
    TheoryValidation,
    summarize_by_rule,
    summarize_by_factuality,
    summarize_by_variant,
    summarize_by_pattern_and_rule,
)

__all__ = [
    "CandidateTriple",
    "TheoryValidation",
    "summarize_by_rule",
    "summarize_by_factuality",
    "summarize_by_variant",
    "summarize_by_pattern_and_rule",
]
