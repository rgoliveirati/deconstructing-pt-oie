"""Testes da avaliação externa (benchmark de Silva et al., LREC 2026).

Os testes que dependem do clone do PT-OIE-Benchmark são pulados quando ele não existe,
para que a suíte continue verde num checkout limpo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.benchmark.external.bootstrap import bootstrap_carb, make_indices, paired_delta
from src.benchmark.external.carb import (
    aggregate,
    assert_equivalent_to_benchmark_evaluator,
    per_sentence_stats,
    score_dataset,
)
from src.benchmark.external.corpora import DATASETS, load_dataset, validate_datasets
from src.benchmark.external.runner import sentence_id

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / ".external/PT-OIE-Benchmark"
OUT = REPO / "outputs/external_benchmark"

needs_bench = pytest.mark.skipif(
    not (BENCH / "Evaluation.py").exists(),
    reason="clone do PT-OIE-Benchmark ausente; rode scripts/fetch_external_benchmark.py",
)
needs_preds = pytest.mark.skipif(
    not (OUT / "predictions/benchmark_format/pt_oie/WIKI200_results.csv").exists(),
    reason="predições ausentes; rode scripts/run_external_benchmark.py",
)


# --------------------------------------------------------------------------- IDs


def test_sentence_id_estavel_e_ordenavel():
    assert sentence_id("WIKI200", 0) == "WIKI200_0000"
    assert sentence_id("WIKI200", 7) == "WIKI200_0007"
    ids = [sentence_id("X", i) for i in range(12)]
    assert ids == sorted(ids), "ordenação lexicográfica deve seguir a ordem do corpus"


# ------------------------------------------------------------------- agregação


def test_aggregate_reproduz_formula_do_avaliador():
    m = aggregate(n_pred_total=10, n_gold_total=20, prec_num=5.0, rec_num=8.0)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.4)
    assert m["f1_score"] == pytest.approx(2 * 0.5 * 0.4 / 0.9)


def test_aggregate_sem_predicoes_nao_divide_por_zero():
    m = aggregate(0, 20, 0.0, 0.0)
    assert m == {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}


# --------------------------------------------------------------------- corpora


@needs_bench
def test_corpora_batem_com_as_contagens_do_artigo():
    report = validate_datasets(BENCH)
    assert report["ok"], report
    assert report["total_sentences"] == 506
    assert report["total_gold_triples"] == 1151
    for name, meta in DATASETS.items():
        d = report["datasets"][name]
        assert d["n_sentences"] == meta["expected_sentences"]
        assert d["n_gold_triples"] == meta["expected_gold"]


@needs_bench
def test_gold_nao_e_modificado_na_carga():
    """A carga preserva o arquivo do PT-OIE-Benchmark byte a byte (nenhuma correção silenciosa)."""
    ds = load_dataset("Gamalho", BENCH)
    raw = pd.read_csv(ds.gold_path, sep="\t")
    pd.testing.assert_frame_equal(ds.gold, raw)


@needs_bench
def test_ordem_das_sentencas_preservada():
    ds = load_dataset("WIKI200", BENCH)
    raw = [l for l in ds.sentences_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert ds.sentences == raw


# ------------------------------------------------- equivalência com o avaliador do PT-OIE-Benchmark


@needs_bench
@needs_preds
@pytest.mark.parametrize("dataset", ["WIKI200", "CETEN200", "Gamalho"])
def test_decomposicao_equivale_ao_avaliador_do_benchmark(dataset):
    ds = load_dataset(dataset, BENCH)
    pred = pd.read_csv(
        OUT / "predictions/benchmark_format/pt_oie" / f"{dataset}_results.csv", sep="\t"
    )
    check = assert_equivalent_to_benchmark_evaluator(ds.gold, pred, str(BENCH))
    assert check["ok"] and check["max_abs_diff"] < 1e-9


@needs_bench
def test_equivalencia_tambem_para_sistema_do_artigo():
    ds = load_dataset("WIKI200", BENCH)
    pred = pd.read_csv(BENCH / "System_Results/PragmaticOIE/WIKI200_results.csv", sep="\t")
    check = assert_equivalent_to_benchmark_evaluator(ds.gold, pred, str(BENCH))
    assert check["ok"]


@needs_bench
def test_numeros_publicados_do_pragmaticoie_sao_reproduzidos():
    """Sanidade do arcabouço: reproduz a Tabela 5 do artigo."""
    ds = load_dataset("WIKI200", BENCH)
    pred = pd.read_csv(BENCH / "System_Results/PragmaticOIE/WIKI200_results.csv", sep="\t")
    s = score_dataset(ds.gold, pred, str(BENCH))
    assert s["precision"] * 100 == pytest.approx(74.2, abs=0.1)
    assert s["recall"] * 100 == pytest.approx(82.0, abs=0.1)
    assert s["f1_score"] * 100 == pytest.approx(77.9, abs=0.1)


@needs_bench
@needs_preds
def test_soma_por_sentenca_reproduz_denominadores():
    ds = load_dataset("Gamalho", BENCH)
    pred = pd.read_csv(OUT / "predictions/benchmark_format/pt_oie/Gamalho_results.csv", sep="\t")
    stats = per_sentence_stats(ds.gold, pred, str(BENCH))
    assert sum(s["n_gold"] for s in stats["sentences"]) == stats["n_gold_total"]
    assert sum(s["n_pred"] for s in stats["sentences"]) + stats["n_pred_orphan"] == stats["n_pred_total"]


@needs_bench
@needs_preds
def test_nenhuma_predicao_orfa():
    """Toda sentença prevista casa com uma sentença gold — alinhamento correto."""
    for dataset in ("WIKI200", "CETEN200", "Gamalho"):
        ds = load_dataset(dataset, BENCH)
        pred = pd.read_csv(
            OUT / "predictions/benchmark_format/pt_oie" / f"{dataset}_results.csv", sep="\t"
        )
        assert per_sentence_stats(ds.gold, pred, str(BENCH))["n_pred_orphan"] == 0


# -------------------------------------------------------------------- bootstrap


def _toy(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"prec_num": float(rng.random()), "rec_num": float(rng.random()),
         "n_pred": int(rng.integers(1, 4)), "n_gold": int(rng.integers(1, 4))}
        for _ in range(n)
    ]


def test_bootstrap_e_deterministico_com_seed():
    ps = _toy()
    a = bootstrap_carb(ps, make_indices(len(ps), 500, seed=42))
    b = bootstrap_carb(ps, make_indices(len(ps), 500, seed=42))
    assert a["point"] == b["point"]
    assert a["f1"]["low"] == b["f1"]["low"] and a["f1"]["high"] == b["f1"]["high"]


def test_ci_contem_o_ponto_e_unidade_e_sentenca():
    ps = _toy(12)
    idx = make_indices(len(ps), 2000, seed=42)
    assert idx.shape == (2000, len(ps))
    res = bootstrap_carb(ps, idx)
    assert res["f1"]["low"] <= res["point"]["f1"] <= res["f1"]["high"]
    assert res["n_sentences"] == len(ps)


def test_delta_pareado_de_sistemas_identicos_e_zero():
    ps = _toy(8)
    idx = make_indices(len(ps), 1000, seed=7)
    a = bootstrap_carb(ps, idx)
    d = paired_delta(a, bootstrap_carb(ps, idx), "a", "b")
    assert d["delta_point_f1"] == 0.0
    assert d["ci_low"] == 0.0 and d["ci_high"] == 0.0
    assert d["significant_95"] is False


def test_delta_pareado_exige_mesmas_replicas():
    ps = _toy(8)
    a = bootstrap_carb(ps, make_indices(8, 100, seed=1))
    b = bootstrap_carb(ps, make_indices(8, 200, seed=1))
    with pytest.raises(ValueError):
        paired_delta(a, b, "a", "b")


def test_orfas_entram_no_denominador_da_precisao():
    ps = _toy(5)
    idx = make_indices(5, 300, seed=3)
    sem = bootstrap_carb(ps, idx, n_pred_orphan=0)
    com = bootstrap_carb(ps, idx, n_pred_orphan=50)
    assert com["point"]["precision"] < sem["point"]["precision"]
    assert com["point"]["recall"] == pytest.approx(sem["point"]["recall"])
