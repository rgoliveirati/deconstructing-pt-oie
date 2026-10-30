"""Adaptador do DptOIE (Oliveira et al., 2023) — implementação dos autores apenas.

Usa exclusivamente os artefatos distribuídos pelos autores: o repositório FORMAS/DptOIE e os
modelos/JAR distribuídos pelos autores na pasta "Models" (Google Drive) do
README dos autores. Nada é reimplementado. O DptOIE é executado em lote (um
processo Java para o arquivo com todas as sentenças, na ordem do corpus),
porque a carga do parser de dependências domina o custo por invocação;
a saída ``extractedFactsByDptOIE.csv`` indexa as sentenças por posição.

Configuração linguística: módulos documentados no README dos autores
(``-SC true -CC true -appositive 1``), correspondentes ao sistema completo
avaliado no artigo do DptOIE. As regras não são alteradas.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..schemas import Prediction
from .base import OpenIESystem, SystemUnavailable

AUTHORS_REPO = "https://github.com/FORMAS/DptOIE"
AUTHORS_MODELS = "https://drive.google.com/drive/folders/1U7p3o2dvWMN0xecocCcsHh7uPmaW1Zmh"


class DptOIESystem(OpenIESystem):
    name = "dptoie"
    supports_batch = True

    def __init__(self, config: dict[str, Any]):
        self.external_dir = Path(config.get("external_dir", ".external/DptOIE"))
        self.jar_path = config.get("jar_path")
        self.flags = list(config.get("flags", ["-SC", "true", "-CC", "true",
                                                "-appositive", "1"]))
        self.timeout = float(config.get("timeout_seconds", 3600))
        self.commit: str | None = None
        self.hashes: dict[str, str] = {}

    def setup(self) -> None:
        if not self.external_dir.exists():
            raise SystemUnavailable(
                self.name,
                "repositório dos autores não está presente em .external/DptOIE; "
                "execute scripts/fetch_external_systems.py (requer rede)",
                evidence=f"esperado em {self.external_dir}; repo dos autores: {AUTHORS_REPO}",
            )
        import shutil

        if shutil.which("java") is None:
            raise SystemUnavailable(
                self.name,
                "runtime Java não encontrado no PATH (DptOIE é um sistema Java)",
                evidence="shutil.which('java') retornou None",
            )
        candidates = (
            [Path(self.jar_path)] if self.jar_path
            else [self.external_dir / "DptOIE-drive.jar", self.external_dir / "DptOIE.jar"]
        )
        jar = next((c for c in candidates if c.exists()), None)
        if jar is None:
            raise SystemUnavailable(
                self.name,
                "JAR dos autores do DptOIE não encontrado",
                evidence=f"procurado: {[str(c) for c in candidates]}",
            )
        dep_model = self.external_dir / "pt-models/pt-dep-parser.gz"
        pos_model = self.external_dir / "pt-models/pt-pos-tagger.model"
        if not dep_model.exists() or not pos_model.exists():
            raise SystemUnavailable(
                self.name,
                "modelos dos autores (pt-dep-parser.gz / pt-pos-tagger.model) ausentes; "
                f"distribuídos pelos autores em {AUTHORS_MODELS}",
                evidence=f"dep={dep_model.exists()} pos={pos_model.exists()}",
            )
        self.jar = jar.resolve()  # o subprocesso roda com cwd=external_dir
        for f in (jar, dep_model, pos_model):
            self.hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        try:
            out = subprocess.run(
                ["git", "-C", str(self.external_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            self.commit = out.stdout.strip() or None
        except Exception:
            self.commit = None

    def extract(self, sentence_id: str, sentence: str) -> list[Prediction]:
        result = self.extract_batch([(sentence_id, sentence)])
        return result[sentence_id]

    def extract_batch(
        self, sentences: list[tuple[str, str]]
    ) -> dict[str, list[Prediction]]:
        """Lote com bisseção: uma sentença que derruba o processo Java não
        pode silenciar as demais. Em falha, o lote é dividido recursivamente
        até isolar as sentenças problemáticas, registradas em
        ``self.batch_errors`` (o runner as grava como erro individual)."""
        self.batch_errors: dict[str, str] = {}
        result: dict[str, list[Prediction]] = {}
        self._extract_chunk(sentences, result)
        return result

    def _extract_chunk(
        self, chunk: list[tuple[str, str]], result: dict[str, list[Prediction]]
    ) -> None:
        try:
            result.update(self._run_dptoie(chunk))
        except Exception as exc:
            if len(chunk) == 1:
                self.batch_errors[chunk[0][0]] = f"{type(exc).__name__}: {exc}"
                return
            mid = len(chunk) // 2
            self._extract_chunk(chunk[:mid], result)
            self._extract_chunk(chunk[mid:], result)

    def _run_dptoie(
        self, sentences: list[tuple[str, str]]
    ) -> dict[str, list[Prediction]]:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "sentences.txt"
            # Uma sentença por linha, na ordem do corpus; sem linha final em
            # branco para não criar sentença fantasma na indexação do DptOIE.
            inp.write_text(
                "\n".join(s.replace("\n", " ") for _, s in sentences),
                encoding="utf-8",
            )
            out_csv = self.external_dir / "extractedFactsByDptOIE.csv"
            out_csv.unlink(missing_ok=True)
            proc = subprocess.run(
                ["java", "-jar", str(self.jar), "-sentencesIN", str(inp), *self.flags],
                capture_output=True, text=True, timeout=self.timeout,
                cwd=str(self.external_dir),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"DptOIE retornou código {proc.returncode}: {proc.stderr[-500:]}"
                )
            if not out_csv.exists():
                raise RuntimeError(
                    "DptOIE não produziu extractedFactsByDptOIE.csv; "
                    f"stderr: {proc.stderr[-500:]}"
                )
            rows = list(csv.reader(
                out_csv.read_text(encoding="utf-8", errors="replace").splitlines(),
                delimiter=";", quotechar='"',
            ))

        result: dict[str, list[Prediction]] = {sid: [] for sid, _ in sentences}
        sid_by_index = {i + 1: sid for i, (sid, _) in enumerate(sentences)}
        sent_by_id = dict(sentences)
        current: str | None = None
        for row in rows[1:]:  # pula cabeçalho
            if not row or all(not c.strip() for c in row):
                continue
            id_sent = row[0].strip()
            if id_sent:
                try:
                    current = sid_by_index.get(int(float(id_sent)))
                except ValueError:
                    current = None
            if current is None:
                continue
            id_extr = row[2].strip() if len(row) > 2 else ""
            if not id_extr:
                continue  # linha da sentença sem extração
            arg1 = row[3].strip() if len(row) > 3 else ""
            rel = row[4].strip() if len(row) > 4 else ""
            arg2 = row[5].strip() if len(row) > 5 else ""
            if not (arg1 or rel or arg2):
                continue
            preds = result[current]
            preds.append(
                Prediction(
                    sentence_id=current,
                    sentence=sent_by_id[current],
                    system=self.name,
                    triple_id=f"{current}_{self.name}_{len(preds):03d}",
                    arg1=arg1, rel=rel, arg2=arg2,
                    raw_output=";".join(row),
                    metadata={
                        "id_extracao": id_extr,
                        "coerencia": row[6].strip() if len(row) > 6 else "",
                        "minimalidade": row[7].strip() if len(row) > 7 else "",
                    },
                )
            )
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "authors_repo": AUTHORS_REPO,
            "authors_models": AUTHORS_MODELS,
            "commit": self.commit,
            "jar": str(getattr(self, "jar", None)),
            "artifact_sha256": self.hashes,
            "flags": self.flags,
            "language": "java",
            "batch_execution": True,
            "training_required": False,
        }
