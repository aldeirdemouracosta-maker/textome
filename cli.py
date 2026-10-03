"""
Textome na linha de comando — roda sem clicar e em lote.

  python cli.py run corpus.txt --config pesquisa.yaml --saida resultados/
  python cli.py run pasta_com_txts/ --k 6 --modelo llama3.1:8b
  python cli.py traduzir entrevistas_en.txt --saida traduzidos/
  python cli.py config-exemplo > pesquisa.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional

from corpus import to_iramuteq
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
    return cfg.with_overrides(
        k=args.k,
        model=args.modelo,
        source_lang=args.idioma_origem,
        translate=False if args.sem_traducao else None,
        use_llm=False if getattr(args, "sem_llm", False) else None,
        mock=True if args.mock else None,
        deep=True if getattr(args, "profunda", False) else None,
        force_refresh=True if args.forcar else None,
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

    run = sub.add_parser("run", help="análise completa (tradução → CHD → LLM → relatório)")
    common(run)
    run.add_argument("--sem-llm", action="store_true", help="não interpretar classes com LLM")
    run.add_argument("--profunda", action="store_true", help="interpretação aprofundada")

    tr = sub.add_parser("traduzir", help="só traduzir corpus para português (formato IRaMuTeQ)")
    common(tr)

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
        return cmd_run(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
