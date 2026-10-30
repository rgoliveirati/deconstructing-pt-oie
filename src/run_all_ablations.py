#!/usr/bin/env python3
"""Run the 26 PT-OIE-EXTRACTOR ablation configurations.

This script is the command-line counterpart of
``notebooks/openie_pt_experiment_final.ipynb``. It preserves the notebook's
configuration labels, uses the same ``Config`` and ``run_experiment`` APIs,
reuses completed runs from disk, and writes a consolidated result table.

Run from the repository root:

    python -m src.run_all_ablations

or, while the file is outside ``src``:

    python run_all_ablations.py --project-root /path/to/repository
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import inspect
import json
import logging
import os
import random
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOGGER = logging.getLogger("pt_oie_ablations")

DEFAULT_BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
BASELINE_LABEL = "rq1_baseline_ud_puro"

S_RULES = ("S1", "S2", "S3", "S4", "S5")
E_RULES = ("E1", "E2", "E3", "E4", "E4_1", "E4_2", "E5", "E6", "E7", "E8", "E9")

RESULT_COLUMNS = (
    "label",
    "status",
    "source",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "delta_f1_vs_baseline",
    "no_attn",
    "theory_mode",
    "apply_theoretical_rules",
    "cop_mode",
    "extract_cop",
    "attn_threshold",
    "attn_decision_enabled",
    "attn_rerank_enabled",
    "heads_mode",
    "top_k_heads",
    "metrics_path",
    "error",
    "config_json",
)


def _jsonable(value: Any) -> Any:
    """Convert common Python objects into deterministic JSON-compatible data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        raw = asdict(config)
    else:
        raw = vars(config).copy()
    return _jsonable(raw)


def _metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _normalise_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tp": _metric(metrics, "TP", "tp"),
        "fp": _metric(metrics, "FP", "fp"),
        "fn": _metric(metrics, "FN", "fn"),
        "precision": _metric(metrics, "precision", "P", "p"),
        "recall": _metric(metrics, "recall", "R", "r"),
        "f1": _metric(metrics, "f1", "F1"),
    }


def _set_global_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _detect_input(project_root: Path, filename: str) -> Path:
    """Find an input file in the documented and legacy repository locations."""
    candidates = (
        project_root / "data" / filename,
        project_root / "notebooks" / filename,
        project_root / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _supports_var_keyword(callable_obj: Any) -> bool:
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(callable_obj).parameters.values()
    )


def _make_config(config_class: Any, kwargs: Mapping[str, Any]) -> Any:
    """Construct Config and fail clearly when notebook/source APIs diverge."""
    signature = inspect.signature(config_class)
    supports_kwargs = _supports_var_keyword(config_class)
    unsupported = sorted(
        key for key in kwargs if key not in signature.parameters and not supports_kwargs
    )
    if unsupported:
        raise RuntimeError(
            "The repository's Config class does not expose fields required by the "
            "ablation notebook: "
            + ", ".join(unsupported)
            + ". Synchronise src/extractor.py with the notebook before running the suite."
        )
    return config_class(**dict(kwargs))


def build_ablation_specs(
    *,
    bert_model: str,
    bosque_path: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Return the 26 configurations in the same order used by the notebook."""

    common = {
        "bert_model": bert_model,
        "bosque_path": bosque_path,
        "seed": seed,
        "cop_mode": "full",
    }

    no_attention_no_theory = {
        **common,
        "no_attn": True,
        "theory_mode": "off",
        "apply_theoretical_rules": False,
    }

    all_theory = {
        **common,
        "no_attn": True,
        "theory_mode": "filter",
        "apply_theoretical_rules": True,
    }

    specs: list[dict[str, Any]] = []

    def add(label: str, **overrides: Any) -> None:
        base = dict(overrides.pop("_base", no_attention_no_theory))
        base.update(overrides)
        specs.append({"label": label, "config": base})

    # RQ1 — pure UD baseline.
    add(BASELINE_LABEL)

    # RQ2 — copular extraction.
    add("rq2_cop_full", cop_mode="full", extract_cop=True)
    add("rq2_cop_restricted", cop_mode="restricted", extract_cop=True)
    add("rq2_cop_off", cop_mode="off", extract_cop=False)

    # RQ3 — attention-based candidate selection.
    attention_base = {
        **common,
        "no_attn": False,
        "attn_decision_enabled": True,
        "attn_rerank_enabled": True,
        "heads_mode": "rank",
        "top_k_heads": 10,
        "theory_mode": "off",
        "apply_theoretical_rules": False,
    }
    add("rq3_attn_on_thr0", _base=attention_base, attn_threshold=0.0)
    add("rq3_attn_on_thr15", _base=attention_base, attn_threshold=0.15)

    # RQ4 — all structural and semantic/discourse rules.
    add("rq4_theory_all_on", _base=all_theory)

    # RQ5 — structural rules only.
    only_e = dict(all_theory)
    only_e.update({f"apply_{rule}": False for rule in S_RULES})
    add("rq5_only_E", _base=only_e)

    # RQ5 — semantic/discourse rules only.
    only_s = dict(all_theory)
    only_s.update({f"apply_{rule}": False for rule in E_RULES})
    add("rq5_only_S", _base=only_s)

    # RQ6 — leave one semantic/discourse rule out.
    for rule in S_RULES:
        add(f"rq6_no_{rule}", _base=all_theory, **{f"apply_{rule}": False})

    # RQ6b — leave one structural rule out.
    for rule in E_RULES:
        add(f"rq6b_no_{rule}", _base=all_theory, **{f"apply_{rule}": False})

    # RQ8 — attention + theory, excluding E4.
    best_combined = {
        **common,
        "no_attn": False,
        "attn_threshold": 0.0,
        "attn_decision_enabled": True,
        "attn_rerank_enabled": True,
        "heads_mode": "rank",
        "top_k_heads": 10,
        "theory_mode": "filter",
        "apply_theoretical_rules": True,
        "apply_E4": False,
    }
    add("rq8_best_config_no_E4", _base=best_combined)

    if len(specs) != 26:
        raise AssertionError(f"Expected 26 ablations, built {len(specs)}")

    labels = [spec["label"] for spec in specs]
    if len(labels) != len(set(labels)):
        raise AssertionError("Ablation labels must be unique")

    return specs


def _select_specs(
    specs: Sequence[dict[str, Any]], patterns: Sequence[str] | None
) -> list[dict[str, Any]]:
    if not patterns:
        return list(specs)

    selected = [
        spec
        for spec in specs
        if any(fnmatch.fnmatch(spec["label"], pattern) for pattern in patterns)
    ]
    if not selected:
        raise ValueError(
            "No ablation matched --only patterns: " + ", ".join(patterns)
        )
    return selected


def _load_cached_metrics(metrics_path: Path) -> Mapping[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid metrics file: {metrics_path}")
    return payload


def _read_metrics_from_result(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        nested = result.get("metrics")
        if isinstance(nested, Mapping):
            return nested
        # Compatibility with a runner that returns the metrics directly.
        if any(key in result for key in ("TP", "tp", "f1", "F1")):
            return result
    raise ValueError("run_experiment() did not return a metrics mapping")


def run_one(
    *,
    spec: Mapping[str, Any],
    config_class: Any,
    run_experiment: Any,
    gold_path: Path,
    output_base: Path,
    dataset_prefix: str,
    force: bool,
) -> dict[str, Any]:
    label = str(spec["label"])
    config = _make_config(config_class, spec["config"])
    config_dict = _config_to_dict(config)

    output_dir = output_base / f"abl_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = f"{dataset_prefix}_{label}"
    metrics_path = output_dir / f"{dataset_name}_metrics.json"

    summary_fields = (
        "no_attn",
        "theory_mode",
        "apply_theoretical_rules",
        "cop_mode",
        "extract_cop",
        "attn_threshold",
        "attn_decision_enabled",
        "attn_rerank_enabled",
        "heads_mode",
        "top_k_heads",
    )
    row: dict[str, Any] = {
        "label": label,
        "status": None,
        "source": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "delta_f1_vs_baseline": None,
        "metrics_path": str(metrics_path),
        "error": None,
        "config_json": json.dumps(config_dict, ensure_ascii=False, sort_keys=True),
        **{field: config_dict.get(field) for field in summary_fields},
    }

    if metrics_path.exists() and not force:
        try:
            metrics = _load_cached_metrics(metrics_path)
            row.update(_normalise_metrics(metrics))
            row.update(status="cached", source="disk")
            LOGGER.info(
                "[CACHE] %-30s F1=%s P=%s R=%s",
                label,
                _format_number(row["f1"]),
                _format_number(row["precision"]),
                _format_number(row["recall"]),
            )
            return row
        except Exception as exc:  # corrupted or incompatible cache
            LOGGER.warning("Ignoring invalid cache for %s: %s", label, exc)

    LOGGER.info("\n%s\nRunning %s\n%s", "=" * 72, label, "=" * 72)
    for key, value in sorted(spec["config"].items()):
        LOGGER.info("  %-30s = %s", key, value)

    try:
        result = run_experiment(
            config=config,
            gold_path=str(gold_path),
            output_dir=str(output_dir),
            dataset_name=dataset_name,
        )
        try:
            metrics = _read_metrics_from_result(result)
        except ValueError:
            if not metrics_path.exists():
                raise
            metrics = _load_cached_metrics(metrics_path)
        row.update(_normalise_metrics(metrics))
        row.update(status="ok", source="fresh")
        LOGGER.info(
            "[OK] %s F1=%s P=%s R=%s TP/FP/FN=%s/%s/%s",
            label,
            _format_number(row["f1"]),
            _format_number(row["precision"]),
            _format_number(row["recall"]),
            row["tp"],
            row["fp"],
            row["fn"],
        )
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}"
        row["source"] = "fresh"
        row["error"] = str(exc)
        LOGGER.error("[ERROR] %s: %s", label, exc)
        LOGGER.debug("%s", traceback.format_exc())
    finally:
        _release_accelerator_memory()

    return row


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def _calculate_deltas(rows: list[dict[str, Any]]) -> None:
    baseline_rows = [row for row in rows if row["label"] == BASELINE_LABEL]
    if not baseline_rows or baseline_rows[0].get("f1") is None:
        return
    baseline_f1 = float(baseline_rows[0]["f1"])
    for row in rows:
        if row.get("f1") is not None:
            row["delta_f1_vs_baseline"] = float(row["f1"]) - baseline_f1


def _write_manifest(
    *,
    specs: Sequence[Mapping[str, Any]],
    path: Path,
    gold_path: Path,
    bosque_path: Path,
    bert_model: str,
    seed: int,
) -> None:
    payload = {
        "gold_path": str(gold_path),
        "bosque_path": str(bosque_path),
        "bert_model": bert_model,
        "seed": seed,
        "number_of_configurations": len(specs),
        "configurations": [
            {"label": spec["label"], "config": _jsonable(spec["config"])}
            for spec in specs
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def _write_results(rows: list[dict[str, Any]], output_base: Path, write_xlsx: bool) -> None:
    output_base.mkdir(parents=True, exist_ok=True)

    json_path = output_base / "ablation_results.json"
    csv_path = output_base / "ablation_results.csv"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf-8",
    )

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to consolidate ablation results") from exc

    frame = pd.DataFrame(rows)
    ordered = [column for column in RESULT_COLUMNS if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    frame = frame[ordered + extras]
    frame.to_csv(csv_path, index=False, sep=";")

    if write_xlsx:
        xlsx_path = output_base / "ablation_results.xlsx"
        try:
            frame.to_excel(xlsx_path, index=False)
        except ImportError:
            LOGGER.warning("openpyxl is not installed; XLSX output was skipped")

    LOGGER.info("Consolidated results: %s", csv_path.resolve())
    LOGGER.info("JSON results:       %s", json_path.resolve())


def _build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and consolidate the 26 PT-OIE-EXTRACTOR ablations."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Repository root used to import src and resolve default paths.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="BIA JSONL file. Auto-detected under data/, notebooks/, or root.",
    )
    parser.add_argument(
        "--bosque",
        type=Path,
        default=None,
        help="UD Bosque CONLLU file. Auto-detected under data/, notebooks/, or root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to outputs/ablation_experiments.",
    )
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--dataset-prefix", default="bia_abl")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--only",
        action="append",
        metavar="GLOB",
        help="Run only labels matching a shell-style pattern; repeatable.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore metrics cache.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true", help="List labels and exit.")
    parser.add_argument("--no-xlsx", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_project_root() -> Path:
    script_path = Path(__file__).resolve()
    # Installed in src/run_all_ablations.py.
    if script_path.parent.name == "src":
        return script_path.parents[1]
    return Path.cwd().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    initial_root = _resolve_project_root()
    parser = _build_parser(initial_root)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    project_root = args.project_root.expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    gold_path = (
        args.gold.expanduser()
        if args.gold is not None
        else _detect_input(project_root, "bia_gold_sentences.jsonl")
    )
    bosque_path = (
        args.bosque.expanduser()
        if args.bosque is not None
        else _detect_input(project_root, "pt_bosque-ud-train.conllu")
    )
    output_base = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else project_root / "outputs" / "ablation_experiments"
    )

    # Relative paths are interpreted from the repository root, not from the
    # caller's current working directory.
    if not gold_path.is_absolute():
        gold_path = project_root / gold_path
    if not bosque_path.is_absolute():
        bosque_path = project_root / bosque_path
    if not output_base.is_absolute():
        output_base = project_root / output_base

    specs = build_ablation_specs(
        bert_model=args.bert_model,
        bosque_path=str(bosque_path),
        seed=args.seed,
    )
    selected_specs = _select_specs(specs, args.only)

    if args.list:
        for index, spec in enumerate(specs, start=1):
            print(f"{index:02d}  {spec['label']}")
        return 0

    missing = [path for path in (gold_path, bosque_path) if not path.exists()]
    if missing:
        parser.error("Input file(s) not found: " + ", ".join(str(path) for path in missing))

    try:
        from src.extractor import Config
        from src.run_experiment import run_experiment
    except Exception as exc:
        LOGGER.error("Could not import the repository experiment modules: %s", exc)
        return 2

    _set_global_seed(args.seed)
    output_base.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        specs=selected_specs,
        path=output_base / "ablation_manifest.json",
        gold_path=gold_path,
        bosque_path=bosque_path,
        bert_model=args.bert_model,
        seed=args.seed,
    )

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Gold corpus:  %s", gold_path)
    LOGGER.info("Bosque file:  %s", bosque_path)
    LOGGER.info("Output:       %s", output_base)
    LOGGER.info("Runs:         %d", len(selected_specs))

    rows: list[dict[str, Any]] = []
    for spec in selected_specs:
        try:
            row = run_one(
                spec=spec,
                config_class=Config,
                run_experiment=run_experiment,
                gold_path=gold_path,
                output_base=output_base,
                dataset_prefix=args.dataset_prefix,
                force=args.force,
            )
        except Exception as exc:
            LOGGER.error("Cannot build/run %s: %s", spec["label"], exc)
            row = {
                "label": spec["label"],
                "status": f"error:{type(exc).__name__}",
                "source": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "delta_f1_vs_baseline": None,
                "metrics_path": None,
                "error": str(exc),
                "config_json": json.dumps(
                    _jsonable(spec["config"]), ensure_ascii=False, sort_keys=True
                ),
            }
        rows.append(row)

        if args.fail_fast and str(row.get("status", "")).startswith("error"):
            break

    _calculate_deltas(rows)
    _write_results(rows, output_base, write_xlsx=not args.no_xlsx)

    failed = [row for row in rows if str(row.get("status", "")).startswith("error")]
    if failed:
        LOGGER.error("Completed with %d failed run(s).", len(failed))
        return 1

    LOGGER.info("All selected ablations completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
