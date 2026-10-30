#!/usr/bin/env bash
# Executa o PragmaticOIE (Sena & Claro, 2020) sobre o corpus BIA.
#
# Motivação: o PragmaticOIE lidera o benchmark do LREC nos dois corpora que o
# próprio repositório dele distribui (WIKI200 e CETEN200, em Sentences/). Rodá-lo
# num corpus de outra tradição de anotação testa se a vantagem viaja. Não viaja:
# 77,9 / 76,3 nos corpora próprios contra 39,6 no Gamalho e 33,9 no BIA.
#
# DUAS ARMADILHAS, ambas resolvidas aqui:
#
#   1. R >= 4.2 QUEBRA o código. Ele usa `||` sobre vetor de comprimento maior
#      que 1 (em contextExtraction), o que virou erro naquela versão. Usamos
#      R 4.1.3, a versão contemporânea ao código, em vez de corrigir a semântica
#      de código de terceiros.
#   2. rJava não acha libjvm.so sem LD_LIBRARY_PATH apontando para o server/ da
#      JVM da env.
#
# Preparação da env (uma vez):
#   conda create -y -n pragmaticoie41 -c conda-forge "r-base=4.1" r-rjava \
#       r-stringr r-nlp r-stringi r-snowballc openjdk=8
#
# Use uma env separada: o R 4.1 e o Java 8 não devem entrar no ambiente Python do artigo.
#
# Uso:  bash scripts/run_pragmaticoie_bia.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRAGMATIC="$REPO/.external/PragmaticOIE"
ENV_DIR="${PRAGMATIC_ENV:-$HOME/miniconda3/envs/pragmaticoie41}"

[ -d "$PRAGMATIC" ] || { echo "ERRO: clone o repositório primeiro:"; \
  echo "  git clone https://github.com/FORMAS/PragmaticOIE $PRAGMATIC"; exit 1; }
[ -x "$ENV_DIR/bin/Rscript" ] || { echo "ERRO: env pragmaticoie41 ausente (ver cabeçalho)"; exit 1; }

export JAVA_HOME="$ENV_DIR/jre"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/amd64/server:${LD_LIBRARY_PATH:-}"
export PRAGMATIC_WS="$PRAGMATIC"
export PRAGMATIC_IN="Sentences/bia.txt"

# 1. Entrada: uma sentença do BIA por linha, na ordem original.
python - "$REPO" "$PRAGMATIC" <<'PY'
import json, re, sys
from pathlib import Path
repo, pragmatic = Path(sys.argv[1]), Path(sys.argv[2])
items = [json.loads(l) for l in
         (repo / "data/bia_gold_sentences.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sents = [re.sub(r"\s+", " ", d["sentence"]).strip() for d in items]
(pragmatic / "Sentences/bia.txt").write_text("\n".join(sents) + "\n", encoding="utf-8")
print(f"entrada: {len(sents)} sentenças -> Sentences/bia.txt")
PY

# 2. main.r com workspace e arquivo de entrada parametrizados (a lógica de
#    extração fica intacta; só os dois caminhos fixos viram variáveis de ambiente).
python - "$PRAGMATIC" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
main = (p / "R/main.r").read_text(encoding="utf-8")
main = main.replace('work_space <- "/home/cflsena/Documentos/Github/R/PragmaticOIE"',
                    'work_space <- Sys.getenv("PRAGMATIC_WS")')
main = main.replace('"Sentences/test.txt"', 'Sys.getenv("PRAGMATIC_IN")')
main = main.replace("if(length(new.packages))\n  install.packages(new.packages)",
                    "if(length(new.packages)) stop(paste('faltam pacotes:', paste(new.packages, collapse=', ')))")
(p / "R/main_bia.r").write_text(main, encoding="utf-8")
print("R/main_bia.r gerado")
PY

# 3. Execução (~4 min para 262 sentenças).
cd "$PRAGMATIC"
rm -rf Output
"$ENV_DIR/bin/Rscript" R/main_bia.r 2>&1 | tail -5
echo "saída bruta: $PRAGMATIC/Output/output.csv ($(wc -l < Output/output.csv) linhas)"
echo
echo "Para pontuar com o avaliador do PT-OIE-Benchmark, ver a conversão em"
echo "paper/tabela_consolidada_proveniencia.md §2b (';' como separador; linhas"
echo "sem índice continuam a sentença anterior; linhas com '.' são vazias)."
