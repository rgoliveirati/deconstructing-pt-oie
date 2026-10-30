# Third-party systems and resources

No third-party code is included in this repository. Systems are run from their
authors' implementations, without modification, cloned into `.external/` by
`scripts/fetch_external_systems.py`. For the BIA runs, the commit, the hashes of
the files used, and the library versions of each system are recorded in
`outputs/bia_external_systems/provenance/`.

## DptOIE (Oliveira, Claro, and Souza, 2023)

- Repository: FORMAS/DptOIE, commit `1a5ef708b1ed`.
- The dependency-parser model `pt-dep-parser.gz` is distributed by the authors in
  the "Models" folder linked from their README, not in the repository. Files used:
  - `pt-models/pt-dep-parser.gz`, SHA-256 `341e6b5bfc2288b8e7991ee746b2517b02dfb2ef7e540b3666f2b969bd4db30d`;
  - `DptOIE.jar` from the same folder, saved as `DptOIE-drive.jar`, SHA-256 `9c260049ad0ff03bcd795c5837ea71f62308bdf5fa088eada75cf2b3206a6dde`;
  - the jar in the repository, for reference, SHA-256 `5ff8246cc170537ce7ccb7e1640872564b1534f2ab6c0dbb00ea264468b36d52`.
- Invocation: `java -jar DptOIE-drive.jar -sentencesIN <file> -SC true -CC true -appositive 1`,
  one batch for all sentences, in corpus order.
- DptOIE splits some inputs into segments. Mapping its outputs back by position
  attaches triples to the wrong sentence (913 of 2,381 triples in a first run,
  `outputs/benchmark/normalized/dptoie.jsonl`, kept for the record). The
  predictions used in the article are aligned by sentence text
  (`outputs/bia_external_systems/DptOIE.jsonl`, 3 of 2,402 misaligned).

## Multi²OIE (Ro, Lee, and Kang, 2020)

- Repository: youngbin-ro/Multi2OIE, commit `4a73a3c37412`, MIT license.
- Checkpoint: the authors' multilingual model (`multilingual_model.bin`,
  980,501,266 bytes) from the link in their README, saved under `.external/Multi2OIE/`.
- Zero-shot, as in the original article: multilingual BERT trained on English
  OpenIE4 data and applied to Portuguese with no training on BIA.
- The original requirements pin PyTorch 1.4.0 and Transformers 2.10.0. The runs
  used PyTorch 2.5.1 and Transformers 4.48.2; loading the checkpoint reports 0
  missing and 0 unexpected keys.
- Mapping to binary triples: the output `[pred, arg0, arg1, ...]` becomes
  arg1 = arg0, rel = pred, arg2 = the remaining arguments joined. The raw line is
  kept in `raw_output`. Multi²OIE leaves a slot empty in 110 of its 672 BIA
  triples; the evaluator counts them.

## ArgOE in LinguaKit (Gamallo et al., 2018)

Repository citiususc/Linguakit, commit `349bb41ed103`. LinguaKit joins the words of multiword expressions with `@`, which the
PT-OIE-Benchmark evaluator deletes, fusing the words. `@` is replaced by a space,
as in the ArgOE outputs distributed with the benchmark. In the benchmark
repository at commit `f69b6bb`, `System_Results/ArgOE/` is identical to
`System_Results/GraphIE/`, so ArgOE's published external scores cannot be
reproduced.

## PragmaticOIE (Sena and Claro, 2020)

Repository FORMAS/PragmaticOIE, commit `f4852e05ebaa`. Runs in R 4.1.3; R ≥ 4.2 raises an error on the code's use of `||` over vectors.
`scripts/run_pragmaticoie_bia.sh` documents the environment.

## DetIE (Vasilkovsky et al., 2022)

Repository commit `a11a638f7c0a`, checkpoint `results/logs/default/version_243/checkpoints/best.ckpt`
(SHA-256 `bec36fa131315daa3e2e008b17fc7a428181aac3da8d5ef6770c8fd6d09918f3`), loaded with
`strict=True`. Predictions on BIA: `outputs/bia_external_systems/DetIE.jsonl`.

## PortNOIE (Cabral et al., 2022)

Not run: the released environment requires Python < 3.10 with AllenNLP 2.7.0 and
unpinned Git dependencies, so it cannot be rebuilt deterministically
(`src/benchmark/systems/portnoie.py` records the check).

## PT-OIE-Benchmark (Silva et al., 2026)

Corpora, system outputs, and `Evaluation.py` from
https://github.com/gabrielrsilva11/PT-OIE-Benchmark at commit
`f69b6bb3bbc1636d07365c2b28012842a8d21a83`, used unmodified.
