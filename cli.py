"""
Textome na linha de comando — roda sem clicar e em lote.

  python cli.py run corpus.txt --config pesquisa.yaml --saida resultados/
  python cli.py run pasta_com_txts/ --k 6 --modelo llama3.1:8b
  python cli.py traduzir entrevistas_en.txt --saida traduzidos/
  python cli.py limpar corpus.txt --termos termos.txt --saida limpos/
  python cli.py importar respostas.csv --texto resposta --variaveis sexo idade --saida corpus.txt
  python cli.py transcrever audios/ --variaveis participantes.csv --saida corpus.txt
  python cli.py validar resultados/ --ia
  python cli.py config-exemplo > pesquisa.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional

from corpus import to_iramuteq
from limpeza import clean_documents, load_compound_terms, read_table, table_to_documents
from pipeline import Config, load_documents, run_pipeline, write_outputs
from translator import LANGUAGE_NAMES, CorpusTranslator, language_summary

EXAMPLE_CONFIG = """\
# Configuração do Textome (todas as chaves são opcionais).
# Guarde este arquivo junto com os resultados: ele documenta o método.

# --- Classificação (rainette / método Reinert) ---
k: 5                   # número de classes
segment_size: 40       # tamanho preferido do segmento (palavras)
min_segment_size: 12   # mínimo de formas por segmento
min_split_members: 10
min_docfreq: 3         # frequência mínima de um termo
n_terms: 20            # formas características por classe
seed: 42               # semente (reprodutibilidade)
language: pt           # stopwords, quando translate: false

# --- Limpeza (regras do IRaMuTeQ) ---
clean: true
compound_terms:        # expressões unidas por "_" (ex.: sistema_único_de_saúde)
  # - sistema único de saúde
  # - bem estar

# --- Tradução para português ---
translate: true
source_lang: auto      # auto, en, es, fr, de, it ou outro

# --- LLM local (Ollama) ---
use_llm: true
model: qwen2.5:7b
temperature: 0.3
deep: false            # interpretação aprofundada
mock: false            # true = respostas simuladas (teste sem Ollama)
# host: http://localhost:11434   # padrão: variável OLLAMA_HOST

# --- Cache ---
use_cache: true
force_refresh: false
cache_db: .cache/interpretations.db
"""


def find_inputs(paths: List[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.glob("*.txt")))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"Não encontrado: {raw}")
    if not found:
        raise FileNotFoundError("Nenhum arquivo .txt encontrado nas entradas.")
    return found


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if getattr(args, "termos", None):
        cfg = cfg.with_overrides(
            compound_terms=list(cfg.compound_terms) + load_compound_terms(args.termos)
        )
    return cfg.with_overrides(
        clean=False if getattr(args, "sem_limpeza", False) else None,
        k=getattr(args, "k", None),
        model=getattr(args, "modelo", None),
        source_lang=getattr(args, "idioma_origem", None),
        translate=False if getattr(args, "sem_traducao", False) else None,
        use_llm=False if getattr(args, "sem_llm", False) else None,
        mock=True if getattr(args, "mock", False) else None,
        deep=True if getattr(args, "profunda", False) else None,
        force_refresh=True if getattr(args, "forcar", False) else None,
    )


def cmd_run(args: argparse.Namespace, bridge_factory: Optional[Callable] = None) -> int:
    cfg = build_config(args)
    inputs = find_inputs(args.entradas)
    out_root = Path(args.saida)
    failures = 0
    for path in inputs:
        out_dir = out_root / path.stem if len(inputs) > 1 else out_root
        print(f"\n=== {path} → {out_dir}")
        try:
            kwargs = {"bridge_factory": bridge_factory} if bridge_factory else {}
            result = run_pipeline(load_documents(path), cfg, source_name=path.name, **kwargs)
            for f in write_outputs(result, out_dir):
                print(f"  ✔ {f}")
            for w in result.warnings:
                print(f"  ⚠ {w}")
        except Exception as e:  # segue para o próximo arquivo do lote
            failures += 1
            print(f"  ✘ Falhou: {e}", file=sys.stderr)
    print(f"\n{len(inputs) - failures}/{len(inputs)} arquivo(s) processado(s).")
    if failures < len(inputs) and cfg.use_llm:
        print(f"Próximo passo: revise os nomes e meça a concordância com "
              f"'python cli.py validar {out_dir if len(inputs) == 1 else out_root / '<arquivo>'}'")
    return 1 if failures else 0


def cmd_traduzir(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    translator = CorpusTranslator(
        model=cfg.model, host=cfg.host, db_path=cfg.cache_db,
        use_cache=cfg.use_cache, mock=cfg.mock,
    )
    out_root = Path(args.saida)
    out_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for path in find_inputs(args.entradas):
        try:
            docs = load_documents(path)
            results = translator.translate_corpus(
                [d.text for d in docs], source_lang=cfg.source_lang,
                force_refresh=cfg.force_refresh,
            )
            out = out_root / f"{path.stem}_pt.txt"
            out.write_text(
                to_iramuteq(
                    [r.text for r in results],
                    headers=[d.header for d in docs],
                    extra_vars=[f"*lang_{r.source_lang}" for r in results],
                ),
                encoding="utf-8",
            )
            resumo = ", ".join(
                f"{LANGUAGE_NAMES.get(k, k)}: {v}" for k, v in language_summary(results).items()
            )
            print(f"✔ {path} → {out}  ({resumo})")
        except Exception as e:
            failures += 1
            print(f"✘ {path}: {e}", file=sys.stderr)
    return 1 if failures else 0


def _print_report(report) -> None:
    for line in report.summary_lines() or ["nenhuma alteração necessária"]:
        print(f"    · {line}")


def cmd_limpar(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    out_root = Path(args.saida)
    out_root.mkdir(parents=True, exist_ok=True)
    for path in find_inputs(args.entradas):
        docs, report = clean_documents(load_documents(path), cfg.compound_terms)
        out = out_root / f"{path.stem}_limpo.txt"
        out.write_text(to_iramuteq([d.text for d in docs], [d.header for d in docs]), encoding="utf-8")
        print(f"✔ {path} → {out}")
        _print_report(report)
    return 0


def cmd_importar(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    docs = table_to_documents(read_table(args.planilha), args.texto, args.variaveis or [])
    report = None
    if not args.sem_limpeza:
        docs, report = clean_documents(docs, cfg.compound_terms)
    out = Path(args.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_iramuteq([d.text for d in docs], [d.header for d in docs]), encoding="utf-8")
    print(f"✔ {len(docs)} resposta(s) → {out}")
    if report:
        _print_report(report)
    return 0


def cmd_transcrever(args: argparse.Namespace, transcriber=None) -> int:
    from transcricao import WhisperTranscriber, find_audio, load_variables, transcribe_to_documents

    cfg = build_config(args)
    files = find_audio(args.entradas)
    variables = load_variables(args.variaveis) if args.variaveis else {}
    if transcriber is None:
        print(f"Carregando Whisper ({args.modelo_whisper})… na 1ª vez o modelo é baixado.")
        transcriber = WhisperTranscriber(
            args.modelo_whisper, language=None if args.idioma == "auto" else args.idioma
        )
    docs, transcripts, warnings = transcribe_to_documents(
        files, transcriber, variables,
        progress=lambda i, n, p: print(f"  [{i}/{n}] {p.name}"),
    )
    for tr in transcripts:
        print(f"    {tr.source.name}: idioma {tr.language}, {tr.duration_s / 60:.1f} min")
    report = None
    if not args.sem_limpeza:
        docs, report = clean_documents(docs, cfg.compound_terms)
    out = Path(args.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_iramuteq([d.text for d in docs], [d.header for d in docs]), encoding="utf-8")
    print(f"✔ {len(docs)} entrevista(s) → {out}")
    for w in warnings:
        print(f"  ⚠ {w}")
    if report:
        _print_report(report)
    print("Revise a transcrição antes da análise (nomes próprios, falas do entrevistador).")
    if any(tr.language != "pt" for tr in transcripts):
        print(f"Há áudio em outro idioma: rode 'python cli.py traduzir {out}' ou 'run' (traduz sozinho).")
    return 0


def cmd_validar(args: argparse.Namespace, ask: Callable[[str], str] = input,
                interpreter=None) -> int:
    """Validação interativa: revisão dos nomes + atribuição às cegas + kappa."""
    import json as _json

    from validacao import (
        NONE_LABEL, assign_with_ai, decide_name, session_from_outputs, write_validation,
    )

    result_dir = Path(args.resultados)
    if not (result_dir / "classes.json").exists():
        raise FileNotFoundError(f"classes.json não encontrado em {result_dir}. Rode 'run' antes.")
    session = session_from_outputs(result_dir, per_class=args.por_classe, seed=args.semente)
    data = _json.loads((result_dir / "classes.json").read_text(encoding="utf-8"))
    forms = {c["id"]: [f["forma"] for f in c.get("formas", [])][:12] for c in data["classes"]}

    print("\n=== Etapa 1/2 — Revisão dos nomes sugeridos pela IA ===")
    print("Enter = aceitar · digite um novo nome = editar · '-' = rejeitar\n")
    for i, d in enumerate(session.names):
        print(f"Classe {d.class_id}: {d.ai_name}")
        if session.descriptions.get(d.class_id):
            print(f"  {session.descriptions[d.class_id]}")
        print(f"  formas: {', '.join(forms.get(d.class_id, []))}")
        session.names[i] = decide_name(d.class_id, d.ai_name, ask("  nome final> "))

    names = session.final_names()
    valid = set(names) | {NONE_LABEL}
    print(f"\n=== Etapa 2/2 — Atribuição às cegas ({len(session.items)} segmentos) ===")
    print("Para cada segmento, escolha a classe que melhor o descreve (0 = nenhuma/não sei):")
    for cid, name in sorted(names.items()):
        print(f"  {cid}. {name}")
    for it in session.items:
        print(f"\n[{it.item_id}/{len(session.items)}] {it.text}")
        while True:
            raw = ask("  classe> ").strip()
            if raw.isdigit() and int(raw) in valid:
                it.human = int(raw)
                break
            print(f"  Digite um destes números: {', '.join(map(str, sorted(valid)))}")

    if args.ia:
        if interpreter is None:
            from llm_interpreter import ClassInterpreter

            cfg = Config.from_yaml(args.config) if args.config else Config()
            interpreter = ClassInterpreter(model=args.modelo or cfg.model, host=cfg.host,
                                           use_cache=False, mock=args.mock)
        session.ai_model = getattr(interpreter, "model_key", None)
        print("\nIA classificando os mesmos segmentos às cegas…")
        assign_with_ai(session, interpreter.atribuir_segmento)

    print("\n=== Resultado ===")
    acc = session.acceptance()
    print(f"Nomes: {acc['aceito']} aceitos, {acc['editado']} editados, {acc['rejeitado']} rejeitados")
    for key, ag in session.agreements().items():
        k = "indefinido" if ag.kappa is None else f"{ag.kappa:.2f}"
        print(f"{key}: kappa = {k} ({ag.interpretation}), concordância {100 * ag.observed:.0f}% (n={ag.n})")
    for p in write_validation(session, result_dir):
        print(f"  ✔ {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="textome", description="Textome — CHD (Reinert) + tradução + interpretação com LLM local."
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("entradas", nargs="+", help="arquivos .txt ou pastas com .txt")
        p.add_argument("--config", help="arquivo YAML de configuração")
        p.add_argument("--saida", default="resultados", help="pasta de saída (padrão: resultados)")
        p.add_argument("--modelo", help="modelo Ollama (sobrepõe a config)")
        p.add_argument("--idioma-origem", choices=["auto", "en", "es", "fr", "de", "it", "outro"])
        p.add_argument("--sem-traducao", action="store_true", help="não traduzir para português")
        p.add_argument("--mock", action="store_true", help="LLM simulado (teste sem Ollama)")
        p.add_argument("--forcar", action="store_true", help="ignorar o cache")
        p.add_argument("--k", type=int, help="número de classes")
        p.add_argument("--termos", help="arquivo com expressões compostas, uma por linha")
        p.add_argument("--sem-limpeza", action="store_true", help="não aplicar a limpeza IRaMuTeQ")

    run = sub.add_parser("run", help="análise completa (tradução → CHD → LLM → relatório)")
    common(run)
    run.add_argument("--sem-llm", action="store_true", help="não interpretar classes com LLM")
    run.add_argument("--profunda", action="store_true", help="interpretação aprofundada")

    tr = sub.add_parser("traduzir", help="só traduzir corpus para português (formato IRaMuTeQ)")
    common(tr)

    lp = sub.add_parser("limpar", help="limpar corpus segundo as regras do IRaMuTeQ")
    lp.add_argument("entradas", nargs="+", help="arquivos .txt ou pastas com .txt")
    lp.add_argument("--config")
    lp.add_argument("--termos", help="arquivo com expressões compostas, uma por linha")
    lp.add_argument("--saida", default="limpos", help="pasta de saída (padrão: limpos)")

    im = sub.add_parser("importar", help="planilha (CSV/XLSX) de respostas → corpus IRaMuTeQ")
    im.add_argument("planilha")
    im.add_argument("--texto", required=True, help="coluna com o texto")
    im.add_argument("--variaveis", nargs="*", help="colunas que viram variáveis (*coluna_valor)")
    im.add_argument("--config")
    im.add_argument("--termos")
    im.add_argument("--sem-limpeza", action="store_true")
    im.add_argument("--saida", default="corpus.txt")

    ts = sub.add_parser("transcrever", help="áudios de entrevistas → corpus IRaMuTeQ (Whisper local)")
    ts.add_argument("entradas", nargs="+", help="arquivos de áudio/vídeo ou pastas")
    ts.add_argument("--variaveis", help="planilha com coluna 'arquivo' + variáveis de cada participante")
    ts.add_argument("--modelo-whisper", default="small",
                    help="tiny, base, small (padrão), medium, large-v3")
    ts.add_argument("--idioma", default="auto", help="auto (padrão) ou código, ex.: pt")
    ts.add_argument("--config")
    ts.add_argument("--termos")
    ts.add_argument("--sem-limpeza", action="store_true")
    ts.add_argument("--saida", default="corpus_entrevistas.txt")

    va = sub.add_parser("validar", help="revisar nomes das classes e medir concordância (kappa)")
    va.add_argument("resultados", help="pasta de resultados de 'run' (com classes.json)")
    va.add_argument("--por-classe", type=int, default=5, help="segmentos sorteados por classe (padrão 5)")
    va.add_argument("--semente", type=int, default=42)
    va.add_argument("--ia", action="store_true", help="a IA também classifica os segmentos às cegas")
    va.add_argument("--modelo", help="modelo Ollama para --ia")
    va.add_argument("--config")
    va.add_argument("--mock", action="store_true")

    sub.add_parser("config-exemplo", help="imprime um YAML de configuração comentado")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.comando == "config-exemplo":
            print(EXAMPLE_CONFIG, end="")
            return 0
        if args.comando == "traduzir":
            return cmd_traduzir(args)
        if args.comando == "limpar":
            return cmd_limpar(args)
        if args.comando == "importar":
            return cmd_importar(args)
        if args.comando == "validar":
            return cmd_validar(args)
        if args.comando == "transcrever":
            return cmd_transcrever(args)
        return cmd_run(args)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
