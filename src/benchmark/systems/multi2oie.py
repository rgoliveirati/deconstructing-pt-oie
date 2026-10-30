"""Adaptador do Multi²OIE (Ro et al., EMNLP 2020) — código e checkpoint dos autores.

Execução multilíngue zero-shot, exatamente como descrita no artigo original
(mBERT treinado apenas com dados em inglês e testado em português sem
ajuste): usa o repositório dos autores ``youngbin-ro/Multi2OIE`` e o checkpoint
multilíngue distribuído pelos autores (Google Drive do README).
Nenhuma linha do código dos autores é modificada; este adaptador invoca
``dataset.load_data`` e ``extract.extract`` do próprio repositório.

O modelo NÃO é treinado nem ajustado com o BIA.

Notas de compatibilidade (registradas no manifesto):
- O ambiente original é torch 1.4/transformers 2.10; aqui o checkpoint é
  carregado com torch 2.5/transformers 4.48. ``load_state_dict`` retorna
  zero chaves faltantes e zero inesperadas — a carga é exata.
- A extração produz tuplas n-árias [pred, arg0, arg1, ...]; para o esquema
  binário do benchmark, arg1 = arg0, rel = pred e arg2 = concatenação dos
  demais argumentos, preservando a saída bruta integral.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..schemas import Prediction
from .base import OpenIESystem, SystemUnavailable

AUTHORS_REPO = "https://github.com/youngbin-ro/Multi2OIE"
AUTHORS_CKPT = (
    "https://drive.google.com/file/d/1lHQeetbacFOqvyPQ3ZzVUGPgn-zwTRA_/view"
)
BERT_CONFIG = "bert-base-multilingual-cased"


class Multi2OIESystem(OpenIESystem):
    name = "multi2oie"
    supports_batch = True

    def __init__(self, config: dict[str, Any]):
        self.external_dir = Path(config.get("external_dir", ".external/Multi2OIE"))
        self.checkpoint = config.get("checkpoint") or str(
            self.external_dir / "multilingual_model.bin"
        )
        self.device = config.get("device", "cpu")
        self.batch_size = int(config.get("batch_size", 32))
        self.max_len = int(config.get("max_len", 64))
        self.mode = config.get("mode", "multilingual_zero_shot")
        self.timeout = float(config.get("timeout_seconds", 3600))
        self.commit: str | None = None
        self.model = None
        self.load_report: dict[str, Any] = {}

    def setup(self) -> None:
        if not self.external_dir.exists():
            raise SystemUnavailable(
                self.name,
                "repositório dos autores não está presente em .external/Multi2OIE; "
                "execute scripts/fetch_external_systems.py (requer rede)",
                evidence=f"esperado em {self.external_dir}; repo dos autores: {AUTHORS_REPO}",
            )
        ckpt = Path(self.checkpoint)
        if not ckpt.exists():
            raise SystemUnavailable(
                self.name,
                "checkpoint multilíngue dos autores ausente; distribuído pelos autores em "
                f"{AUTHORS_CKPT}",
                evidence=f"checkpoint configurado: {self.checkpoint!r}",
            )
        import torch

        sys.path.insert(0, str(self.external_dir.resolve()))
        try:
            from utils import utils as m2o_utils
        except Exception as exc:
            raise SystemUnavailable(
                self.name, f"falha ao importar o código dos autores: {exc}"
            )
        self.model = m2o_utils.get_models(bert_config=BERT_CONFIG, device=self.device)
        state = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemUnavailable(
                self.name,
                "checkpoint dos autores incompatível com o stack instalado "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                evidence=f"missing={missing[:5]} unexpected={unexpected[:5]}",
            )
        self.load_report = {"missing_keys": 0, "unexpected_keys": 0}
        self.model.zero_grad()
        self.model.eval()
        try:
            out = subprocess.run(
                ["git", "-C", str(self.external_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            self.commit = out.stdout.strip() or None
        except Exception:
            self.commit = None

    def extract(self, sentence_id: str, sentence: str) -> list[Prediction]:
        return self.extract_batch([(sentence_id, sentence)])[sentence_id]

    def extract_batch(
        self, sentences: list[tuple[str, str]]
    ) -> dict[str, list[Prediction]]:
        from dataset import load_data
        from extract import extract as m2o_extract

        with tempfile.TemporaryDirectory() as tmp:
            pkl = Path(tmp) / "sentences.pkl"
            with open(pkl, "wb") as fh:
                pickle.dump([s for _, s in sentences], fh)
            loader = load_data(
                data_path=str(pkl),
                batch_size=self.batch_size,
                max_len=self.max_len,
                train=False,
                tokenizer_config=BERT_CONFIG,
            )
            args = SimpleNamespace(device=self.device, binary=False,
                                    bert_config=BERT_CONFIG)
            outdir = Path(tmp) / "out"
            m2o_extract(args, self.model, loader, str(outdir))
            lines = (outdir / "extraction.txt").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()

        result: dict[str, list[Prediction]] = {sid: [] for sid, _ in sentences}
        sid_by_text: dict[str, str] = {}
        for sid, text in sentences:
            sid_by_text.setdefault(text, sid)
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            sentence, confidence, pred = parts[0], parts[1], parts[2]
            argset = parts[3:]
            sid = sid_by_text.get(sentence)
            if sid is None:
                continue
            preds = result[sid]
            preds.append(
                Prediction(
                    sentence_id=sid,
                    sentence=sentence,
                    system=self.name,
                    triple_id=f"{sid}_{self.name}_{len(preds):03d}",
                    arg1=argset[0] if argset else "",
                    rel=pred,
                    arg2=" ".join(argset[1:]).strip(),
                    confidence=float(confidence) if confidence else None,
                    raw_output=line,
                    metadata={"n_ary_args": argset},
                )
            )
        return result

    def metadata(self) -> dict[str, Any]:
        import torch, transformers  # noqa: E401

        return {
            "authors_repo": AUTHORS_REPO,
            "authors_checkpoint": AUTHORS_CKPT,
            "commit": self.commit,
            "checkpoint": str(self.checkpoint),
            "bert_config": BERT_CONFIG,
            "mode": self.mode,
            "zero_shot": True,
            "trained_on": "OpenIE4 (inglês); nenhum ajuste com o BIA",
            "device": self.device,
            "batch_size": self.batch_size,
            "max_len": self.max_len,
            "state_dict_load": self.load_report,
            "compat_note": (
                "checkpoint torch 1.4 carregado com torch "
                f"{torch.__version__}/transformers {transformers.__version__}; "
                "0 chaves faltantes, 0 inesperadas"
            ),
            "training_required": True,
        }

    def teardown(self) -> None:
        self.model = None
