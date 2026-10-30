"""Benchmark comparativo de sistemas OpenIE sobre o corpus BIA.

Subpacote independente do extrator experimental: `src.extractor` é importado
apenas pelos adaptadores internos e pelo protocolo `bia_legacy`.
"""

__all__ = ["schemas", "corpus", "normalization", "assignment", "evaluation", "bootstrap"]
