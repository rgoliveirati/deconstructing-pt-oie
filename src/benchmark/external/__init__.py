"""Avaliação externa: PT-OIE-EXTRACTOR nos corpora do benchmark de Silva et al. (LREC 2026).

Este subpacote NÃO altera o comportamento experimental do BIA. Ele apenas
executa o extrator, na configuração congelada do artigo, sobre corpora de
terceiros, e pontua com o avaliador do PT-OIE-Benchmark.
"""

from .corpora import DATASETS, load_dataset, validate_datasets
from .carb import score_dataset, per_sentence_stats, aggregate

__all__ = [
    "DATASETS",
    "load_dataset",
    "validate_datasets",
    "score_dataset",
    "per_sentence_stats",
    "aggregate",
]
