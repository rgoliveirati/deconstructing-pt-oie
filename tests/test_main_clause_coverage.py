"""Oportunidade 1 — cobertura de oração principal.

Trava dois compromissos:

1. os defaults das flags novas reproduzem EXATAMENTE o comportamento do artigo
   (as 26 ablações não podem mudar de valor por causa deste trabalho);
2. cada flag, quando ligada, cobre o padrão que motivou sua criação.

Os testes que instanciam o extrator são marcados ``slow`` (torch/stanza/BERT).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Sentenças que motivaram cada regra, com o padrão UD esperado.
CASES = {
    "obl_governed": (
        "As mesmas divergências acontecem com o PP .",
        {"obl_core_policy": "governed"},
        "verb+obl",
    ),
    "obl_sole_temporal": (
        "As filmagens começaram em 11 de setembro de 2011 .",
        {"obl_core_policy": "governed", "obl_allow_sole_temporal": True},
        "verb+obl",
    ),
    "passive_agent": (
        "A queda de as vendas em relação a março já era prevista por a indústria .",
        {"allow_passive_agent": True},
        "aux:pass+agent",
    ),
    "null_subject": (
        "Até mesmo en a Organização por a Libertação de a Palestina houve críticas a o líder .",
        {"allow_null_subject": True},
        "verb+null-subj",
    ),
}


# --------------------------------------------------------------- defaults


def test_defaults_da_config_preservam_o_artigo():
    """Nenhuma flag nova pode vir ligada por default."""
    from src.extractor import Config

    cfg = Config()
    assert cfg.obl_core_policy == "locative_only"
    assert cfg.obl_allow_sole_temporal is False
    assert cfg.allow_null_subject is False
    assert cfg.allow_passive_agent is False


def test_sistema_pt_oie_nao_traz_overrides():
    """`pt_oie` continua sendo a configuração do artigo; só `pt_oie_op1` difere."""
    from src.benchmark.systems.pt_oie import PTOIEMainClauseSystem, PTOIESystem

    assert PTOIESystem({}).overrides == {}
    ov = PTOIEMainClauseSystem({}).overrides
    assert ov["obl_core_policy"] == "governed"
    assert ov["allow_passive_agent"] is True
    # 1b foi rejeitado no desenvolvimento: no BIA soma predições sem mover a
    # revocação (62,06 -> 62,06). Tem de continuar desligado.
    assert ov["allow_null_subject"] is False


def test_registry_conhece_o_sistema_novo():
    from src.benchmark.registry import SYSTEM_NAMES

    assert "pt_oie_op1" in SYSTEM_NAMES


# ----------------------------------------------------------------- slow


@pytest.fixture(scope="module")
def base_kwargs():
    return dict(
        bert_model="neuralmind/bert-base-portuguese-cased",
        bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"),
        seed=13, cop_mode="full", theory_mode="off", apply_theoretical_rules=False,
        no_attn=True,  # atenção desligada: estes testes checam GERAÇÃO de candidatos
    )


def _patterns(sentence, kwargs):
    from src.extractor import Config, OpenIEExtractorBIA

    ex = OpenIEExtractorBIA(config=Config(**kwargs), verbose=False)
    cands, _, _ = ex._generate_all_candidates(sentence)
    return {c.pattern_ud for c in cands}, cands


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_flag_ligada_gera_o_padrao(case, base_kwargs):
    sentence, overrides, expected = CASES[case]
    pats, cands = _patterns(sentence, {**base_kwargs, **overrides})
    assert expected in pats, f"{case}: padrões gerados = {pats}"
    assert all(c.arg1 and c.rel and c.arg2 for c in cands if c.pattern_ud == expected)


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_flag_desligada_nao_gera_nada(case, base_kwargs):
    """Com os defaults do artigo, estas sentenças não produzem candidato algum."""
    sentence, _, _ = CASES[case]
    pats, _ = _patterns(sentence, base_kwargs)
    assert pats == set(), f"{case}: esperado nenhum candidato, veio {pats}"


@pytest.mark.slow
def test_passiva_sem_agente_nao_dispara(base_kwargs):
    """`allow_passive_agent` exige obl:agent explícito — não é passiva genérica."""
    pats, _ = _patterns("A casa foi construída rapidamente .",
                        {**base_kwargs, "allow_passive_agent": True})
    assert "aux:pass+agent" not in pats


@pytest.mark.slow
def test_obl_governed_nao_captura_adjunto_pre_verbal(base_kwargs):
    """O ramo 'governed' só aceita oblíquo pós-verbal; introdutório continua fora."""
    from src.extractor import Config, OpenIEExtractorBIA

    ex = OpenIEExtractorBIA(config=Config(**{**base_kwargs, "obl_core_policy": "governed"}),
                            verbose=False)
    sentence = "Em fevereiro de 2013 , Beyoncé cantou o hino nacional ."
    cands, ud, kids = ex._generate_all_candidates(sentence)
    verbs = [t for t in ud if (t.upos or "").upper() == "VERB"]
    for v in verbs:
        for c in kids.get(v.id, []):
            if (c.deprel or "").startswith("obl") and c.id < v.id:
                assert not ex._obl_is_core_for_attachment(c, v, ud, kids)


@pytest.mark.slow
def test_reproducao_bit_exata_do_artigo_no_bia(base_kwargs):
    """Trava a Tabela 3: os defaults têm de dar TP=265, FP=273, FN=162."""
    import json

    from src.extractor import Config, OpenIEExtractorBIA, evaluate_dataset

    gold_path = REPO / "data/bia_gold_sentences.jsonl"
    if not gold_path.exists():
        pytest.skip("corpus BIA ausente")
    items = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    kwargs = dict(base_kwargs)
    kwargs.update(no_attn=False, attn_decision_enabled=True, attn_rerank_enabled=True,
                  heads_mode="rank", top_k_heads=10, attn_threshold=0.0)
    ex = OpenIEExtractorBIA(config=Config(**kwargs), verbose=False)
    preds = [ex.extract(d["sentence"])[0] for d in items]
    m = evaluate_dataset(preds, items, ex.config)
    assert (m["TP"], m["FP"], m["FN"]) == (265, 273, 162)
    assert sum(len(p) for p in preds) == 538
