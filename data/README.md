# Data

| File | Content | License | SHA-256 |
|---|---|---|---|
| `bia_gold_sentences.jsonl` | BIA: 262 sentences, 427 reference triples | CC BY-SA 4.0 | `ddd882e93b3d7f9f5778f1a740e0f3854a193e00327082acc3d658bd6550b19e` |
| `pt_bosque-ud-train.conllu` | UD Portuguese Bosque, training split | CC BY-SA 4.0 | `b56d7124af821e2968302c38298904ec67af342304a33ae8a9dc4fa57df89648` |
| `biomed_pt_pmids.txt` | PubMed IDs of the 139 articles behind the 241 sentences of the polarity analysis | — | — |

Downloaded by scripts and not redistributed:

| Path | Source | Script |
|---|---|---|
| `.external/PT-OIE-Benchmark/` | WIKI200, CETEN200, Gamalho, system outputs, and `Evaluation.py`; https://github.com/gabrielrsilva11/PT-OIE-Benchmark, commit `f69b6bb` | `scripts/fetch_external_benchmark.py` |
| `data/external/OIEC-PT/` | OIEC-PT, Zenodo 10.5281/zenodo.17438351, CC BY 4.0 | `scripts/fetch_oiec_pt.py` |
| `data/biomed_pt_sentences.txt` | Portuguese abstracts (`OtherAbstract`) of the PubMed articles above | `scripts/fetch_biomed_pt.py --pmids data/biomed_pt_pmids.txt` |

## BIA

BIA (Queiroz, Cavalcante, and Claro, 2023) is built on Portuguese machine
translations of English Wikipedia sentences from SQuAD 2.0 (Rajpurkar, Jia, and
Liang, 2018), revised and annotated with OpenIE triples. Only reference triples
whose elements occur as contiguous spans of the sentence are kept; this file
holds the 262 of 360 sentences that retain at least one triple. SQuAD 2.0 is
distributed under CC BY-SA 4.0, and BIA is distributed here under the same
license.

- B. Queiroz, R. Cavalcante, D. Claro. Desafios da tarefa de Extração de
  Informação Aberta: uma abordagem metodológica de um corpus automatizado até o
  corpus manual. STIL 2023, pp. 388–392. https://doi.org/10.5753/stil.2023.234606
- P. Rajpurkar, R. Jia, P. Liang. Know What You Don't Know: Unanswerable
  Questions for SQuAD. ACL 2018, pp. 784–789. https://doi.org/10.18653/v1/P18-2124

## UD Bosque

Used only to rank attention heads, never as an evaluation set. From
https://github.com/UniversalDependencies/UD_Portuguese-Bosque.

- A. Rademaker, F. Chalub, L. Real, C. Freitas, E. Bick, V. de Paiva. Universal
  Dependencies for Portuguese. Depling 2017, pp. 197–206.
  https://aclanthology.org/W17-6523/
