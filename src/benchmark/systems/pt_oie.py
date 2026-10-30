"""Adaptador do PT-OIE-EXTRACTOR na melhor configuração do artigo.

Configuração idêntica a ``rq3_attn_on_thr0`` de ``src/run_all_ablations.py``:
UD + atenção (heads_mode="rank", top_k_heads=10, attn_threshold=0.0),
validação linguística desativada, cop_mode="full", seed 13, BERTimbau.
A implementação original não é duplicada: usa-se a API pública
``src.extractor.Config`` / ``OpenIEExtractorBIA``.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Prediction
from .base import OpenIESystem, SystemUnavailable


def _paper_config_kwargs(bosque_path: str, seed: int, *, attention: bool) -> dict[str, Any]:
    base: dict[str, Any] = {
        "bert_model": "neuralmind/bert-base-portuguese-cased",
        "bosque_path": bosque_path,
        "seed": seed,
        "cop_mode": "full",
        "theory_mode": "off",
        "apply_theoretical_rules": False,
    }
    if attention:
        base.update(
            no_attn=False,
            attn_decision_enabled=True,
            attn_rerank_enabled=True,
            heads_mode="rank",
            top_k_heads=10,
            attn_threshold=0.0,
        )
    else:  # RQ1: UD puro
        base.update(no_attn=True)
    return base


class PTOIESystem(OpenIESystem):
    name = "pt_oie"
    use_attention = True
    #: Overrides aplicados sobre a configuração do artigo. Vazio = artigo.
    extra_config: dict[str, Any] = {}

    def __init__(self, config: dict[str, Any]):
        self.bosque_path = config.get("bosque_path", "data/pt_bosque-ud-train.conllu")
        self.seed = int(config.get("seed", 13))
        self.extractor = None
        self.config_kwargs: dict[str, Any] = {}
        self.overrides = dict(self.extra_config)
        self.overrides.update(config.get("config_overrides") or {})

    def setup(self) -> None:
        try:
            from src.extractor import Config, OpenIEExtractorBIA
        except Exception as exc:  # pragma: no cover
            raise SystemUnavailable(self.name, f"falha ao importar src.extractor: {exc}")

        self.config_kwargs = _paper_config_kwargs(
            self.bosque_path, self.seed, attention=self.use_attention
        )
        self.config_kwargs.update(self.overrides)
        config = Config(**self.config_kwargs)
        self.extractor = OpenIEExtractorBIA(config=config, verbose=False)
        # Força a inicialização (Stanza, BERT e ranking de cabeças) no setup,
        # para que o custo não seja atribuído à primeira sentença.
        self.extractor.extract("O Brasil exporta soja.")

    def extract(self, sentence_id: str, sentence: str) -> list[Prediction]:
        rows, _ = self.extractor.extract(sentence)
        preds = []
        for i, row in enumerate(rows):
            preds.append(
                Prediction(
                    sentence_id=sentence_id,
                    sentence=sentence,
                    system=self.name,
                    triple_id=f"{sentence_id}_{self.name}_{i:03d}",
                    arg1=str(row.get("arg1", "")),
                    rel=str(row.get("rel", "")),
                    arg2=str(row.get("arg2", "")),
                    confidence=row.get("conf_att"),
                    raw_output=row,
                    metadata={"pattern_ud": row.get("pattern_ud")},
                )
            )
        return preds

    def metadata(self) -> dict[str, Any]:
        import stanza, torch, transformers  # noqa: E401

        meta: dict[str, Any] = {
            "config": dict(self.config_kwargs),
            "stanza_version": stanza.__version__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "bert_checkpoint": self.config_kwargs.get("bert_model"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "training_required": False,
            "approx_parameters": "110M (BERTimbau base)" if self.use_attention else "0 (simbólico) + parser Stanza",
        }
        if self.extractor is not None:
            meta["heads_meta"] = getattr(self.extractor, "heads_meta", None)
        return meta

    def teardown(self) -> None:
        self.extractor = None


class UDBaselineSystem(PTOIESystem):
    """Baseline UD puro — configuração RQ1 (atenção e validação desativadas)."""

    name = "ud_baseline"
    use_attention = False


class PTOIEMainClauseSystem(PTOIESystem):
    """Configuração do artigo + cobertura de oração principal (Oportunidade 1).

    Ativa apenas o que foi validado nos corpora de DESENVOLVIMENTO (BIA e
    OIEC-PT): oblíquo regido não-locativo (1a) e passiva com agente (1c).
    Sujeito nulo (1b) ficou de fora — no BIA acrescenta 41 predições sem elevar
    a revocação (62,06 -> 62,06) e no OIEC-PT o ganho é marginal.
    """

    name = "pt_oie_op1"
    extra_config = {
        "obl_core_policy": "governed",
        "obl_allow_sole_temporal": True,
        "allow_passive_agent": True,
        "allow_null_subject": False,
    }
