"""Auditoria de subsunção da camada de validação (regras S1–S5, E1–E9).

Responde à questão deixada em aberto na §6.1 do artigo: as regras que nunca
rejeitam candidato são silenciosas por **subsunção** (a precondição já é
garantida a montante, na geração) ou por **inadequação ao corpus** (a construção
que elas visam não ocorre no BIA)?

Método:

* **empírico** — instrumenta o dispatcher e conta, por regra, quantas vezes foi
  avaliada e quantas rejeitou, em CINCO corpora de sintaxe e anotação bem
  diferentes. Uma regra que nunca rejeita em nenhum deles dificilmente é
  silenciosa por peculiaridade de um corpus;
* **estático** — a classificação em ``STATIC_VERDICT`` registra, por leitura do
  código, POR QUE a regra não pode rejeitar (tautologia, flag de configuração,
  ou precondição garantida pela geração).

As duas evidências são reportadas lado a lado; divergência entre elas é sinal de
erro na classificação estática.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.benchmark.runtime import now_iso, write_json  # noqa: E402

RULES = ["E1", "E2", "E3", "E4", "E4_1", "E4_2", "E5", "E6", "E7", "E8", "E9",
         "S1", "S2", "S3", "S4", "S5"]

#: Veredito por leitura do código, com o mecanismo. Confrontado com o empírico.
STATIC_VERDICT: dict[str, tuple[str, str]] = {
    "E6": ("tautologia", "todos os ramos retornam True"),
    "E7": ("tautologia", "expressão `True if cond else True`"),
    "E9": ("tautologia", "_detect_prepositional_locution só devolve a locução se ela "
                         "já estiver contida em rel; o ramo else é sempre verdadeiro"),
    "E4_2": ("tautologia", "idêntica a E9"),
    "S5": ("tautologia", "retorna True: não há inferência externa nesta implementação"),
    "E4_1": ("inerte por configuração", "gated por strict_theory, que é False por default"),
    "E8": ("inerte por configuração", "gated por e8_allow_clausal_args, True por default"),
    "S2": ("inerte por configuração", "s2_span_policy='both' por default retorna True"),
    # ATENÇÃO: a hipótese inicial de que E4 estaria subsumida por
    # _attach_governed_preposition_to_rel foi REFUTADA pela auditoria — E4 é a
    # regra que mais rejeita (6,2%). A anexação da preposição só ocorre nos
    # padrões baseados em obl; nos demais (verb+obj[+obl], nary_split, clausais)
    # o objeto pode ter um `case` que a geração não move para rel.
    "E4": ("ativa", "rejeita quando o objeto tem preposição regente que não foi "
                    "movida para rel; a anexação só cobre os padrões via obl"),
    "E1": ("ativa", "além da realização verbal, exige vínculo de sujeito/objeto e "
                    "forma copular/passiva compatível"),
    "E5": ("ativa", "rejeita âncora argumental pobre (pronome relativo isolado)"),
    "E2": ("dependente do corpus (não observada)", "concordância Number/Person"),
    "E3": ("dependente do corpus (não observada)", "oração relativa não essencial no argumento"),
    "S1": ("ativa", "factualidade/modalidade: rejeita hipotéticos e condicionais"),
    "S3": ("ativa", "oblíquo periférico anexado / nary_split não central"),
    "S4": ("ativa", "forma copular/passiva compatível com o padrão"),
}


def corpora(bench_dir: Path, oiec_path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}

    bia = REPO / "data/bia_gold_sentences.jsonl"
    if bia.exists():
        out["BIA"] = [json.loads(l)["sentence"]
                      for l in bia.read_text(encoding="utf-8").splitlines() if l.strip()]

    for name in ("WIKI200", "CETEN200", "Gamalho"):
        p = bench_dir / "Gold" / "Sentences" / f"{name}_Sentences.txt"
        if p.exists():
            out[name] = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    if oiec_path.exists():
        from src.benchmark.oiec import load_oiec
        out["OIEC-PT"] = [s.sentence for s in load_oiec(oiec_path)]

    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-dir", default=".external/PT-OIE-Benchmark")
    p.add_argument("--oiec", default=str(REPO / "data/external/OIEC-PT/OIEC-PT.txt"))
    p.add_argument("--output", default="outputs/rule_audit/subsumption.json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    from src.extractor import Config, OpenIEExtractorBIA

    # Camada teórica LIGADA e em modo 'annotate': queremos observar as regras,
    # não deixá-las alterar a saída.
    cfg = Config(
        bert_model="neuralmind/bert-base-portuguese-cased",
        bosque_path=str(REPO / "data/pt_bosque-ud-train.conllu"),
        seed=13, cop_mode="full",
        theory_mode="annotate", apply_theoretical_rules=True,
        no_attn=True,  # atenção é irrelevante para a validação; acelera a auditoria
    )
    ex = OpenIEExtractorBIA(config=cfg, verbose=False)

    evaluated: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rejected: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Instrumenta cada _validate_X sem alterar o comportamento.
    def wrap(rule: str, fn):
        def inner(cand, ud, kids):
            res = fn(cand, ud, kids)
            ok = res[0] if isinstance(res, tuple) else res
            evaluated[current["corpus"]][rule] += 1
            if not ok:
                rejected[current["corpus"]][rule] += 1
            return res
        return inner

    current = {"corpus": "?"}
    for rule in RULES:
        attr = f"_validate_{rule}"
        if hasattr(ex, attr):
            setattr(ex, attr, wrap(rule, getattr(ex, attr)))

    sets = corpora(REPO / args.bench_dir, Path(args.oiec))
    if not sets:
        print("ERRO: nenhum corpus encontrado", file=sys.stderr)
        return 2

    for name, sentences in sets.items():
        current["corpus"] = name
        if args.limit:
            sentences = sentences[: args.limit]
        for s in sentences:
            try:
                ex.extract(s)
            except Exception:
                continue
        print(f"  {name}: {len(sentences)} sentenças processadas", flush=True)

    # ------------------------------------------------------------------ tabela
    names = list(sets)
    rows = []
    for rule in RULES:
        tot_e = sum(evaluated[c][rule] for c in names)
        tot_r = sum(rejected[c][rule] for c in names)
        static_kind, static_why = STATIC_VERDICT.get(rule, ("?", ""))
        empirical = "rejeita" if tot_r > 0 else ("nunca rejeita" if tot_e else "nunca avaliada")
        # o veredito estático prevê disparo apenas para regras classificadas "ativa"
        agree = (tot_r > 0) == (static_kind == "ativa")
        rows.append({
            "rule": rule,
            "evaluated": tot_e,
            "rejected": tot_r,
            "reject_rate_pct": (tot_r / tot_e * 100) if tot_e else None,
            "per_corpus_rejected": {c: rejected[c][rule] for c in names},
            "empirical": empirical,
            "static_verdict": static_kind,
            "static_mechanism": static_why,
            "agreement": agree,
        })

    result = {
        "created_at": now_iso(),
        "corpora": {c: len(s) for c, s in sets.items()},
        "config": {"theory_mode": "annotate", "strict_theory": cfg.strict_theory,
                   "e8_allow_clausal_args": cfg.e8_allow_clausal_args,
                   "s2_span_policy": cfg.s2_span_policy},
        "rules": rows,
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)

    print(f"\n{'regra':<6}{'avaliada':>10}{'rejeita':>9}{'taxa':>8}  {'veredito estático':<26}{'confere?':>9}")
    for r in rows:
        rate = f"{r['reject_rate_pct']:.2f}%" if r["reject_rate_pct"] is not None else "—"
        print(f"{r['rule']:<6}{r['evaluated']:>10}{r['rejected']:>9}{rate:>8}  "
              f"{r['static_verdict']:<26}{'sim' if r['agreement'] else 'NÃO':>9}")

    silent = [r["rule"] for r in rows if r["rejected"] == 0]
    print(f"\nregras silenciosas em {len(sets)} corpora ({sum(len(s) for s in sets.values())} sentenças): "
          f"{len(silent)}/{len(RULES)} -> {', '.join(silent)}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
