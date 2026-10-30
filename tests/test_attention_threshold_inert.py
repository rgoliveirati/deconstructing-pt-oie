"""Regressão: documenta que `attn_threshold` é INERTE no pipeline ativo.

Descoberta em 2026-07-16 ao executar o sweep de tau: variar
tau de 0.00 a 0.30 (inclusive acima do maior conf_att observado, 0.1737) não
altera nenhuma contagem. A causa é estrutural, não estatística:

- `_passes_attention_gate` está definido (duas vezes) mas nunca é chamado;
- o filtro por limiar do `extract()` legado (`attn_filter_in_legacy`) está
  desativado por padrão E esse `extract()` é sombreado pela camada
  theory-aware;
- o `extract()` ativo (camada theory-aware) aplica apenas
  `_apply_theory_policy` e `_apply_attention_rerank`.

Consequência: a atenção atua exclusivamente como RERANKER (escolhe a melhor
variante por grupo de competição), nunca como filtro por limiar. O ganho de
+2.21 F1 vem do reranking; o limiar não participa.

Estes testes travam esse comportamento para que qualquer mudança futura
(intencional ou acidental) seja detectada.
"""

import inspect

import pytest


def test_gate_de_threshold_nunca_e_chamado():
    """`_passes_attention_gate` existe mas nenhum caminho de execução o usa."""
    from src import extractor as ex

    source = inspect.getsource(ex)
    definicoes = source.count("def _passes_attention_gate")
    chamadas = source.count("self._passes_attention_gate(")
    assert definicoes >= 1, "o método deveria existir (legado)"
    assert chamadas == 0, (
        "o gate passou a ser chamado; se isso for intencional, o sweep de "
        "thresholds e a descrição do artigo precisam ser refeitos"
    )


def test_extract_ativo_nao_aplica_threshold():
    """O extract() efetivamente usado não referencia limiar de atenção."""
    from src.extractor import OpenIEExtractorBIA

    linhas, _ = inspect.getsourcelines(OpenIEExtractorBIA.extract)
    corpo = "".join(linhas)
    assert "_apply_attention_rerank" in corpo, "o reranking é o mecanismo ativo"
    assert "attn_threshold" not in corpo
    assert "_passes_attention_gate" not in corpo


def test_filtro_legado_desativado_por_padrao():
    """O único filtro por limiar existente está desligado na configuração padrão."""
    from src.extractor import Config

    assert Config().attn_filter_in_legacy is False


@pytest.mark.slow
def test_sweep_registrado_e_constante():
    """Os resultados salvos do sweep são idênticos em todos os tau."""
    import json
    from pathlib import Path

    caminho = (
        Path(__file__).resolve().parents[1]
        / "outputs/benchmark/metrics/attention_threshold_sweep.json"
    )
    if not caminho.exists():
        pytest.skip("sweep ainda não executado")
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    assert len(dados) >= 4, "o sweep deve cobrir vários valores de tau"
    contagens = {
        (e["bia_legacy"]["tp"], e["bia_legacy"]["fp"], e["bia_legacy"]["fn"])
        for e in dados.values()
    }
    assert len(contagens) == 1, (
        "o limiar passou a ter efeito; revisar a descrição do artigo"
    )
