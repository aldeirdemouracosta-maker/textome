"""
Servidor MCP do Textome: permite que o Claude (Desktop, Code ou outro cliente
MCP) rode as análises por linguagem natural, ex.:
  "Rode a CHD em entrevistas.txt com k=6 e me diga o que cada classe representa."

Instalação no Claude Code:
  claude mcp add textome -e TEXTOME_DADOS=/caminho/dos/corpora -- python /caminho/textome/mcp_server.py

Segurança: as ferramentas só leem e escrevem dentro da pasta TEXTOME_DADOS
(padrão: pasta atual). Caminhos fora dela são recusados.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "textome",
    instructions=(
        "Análise textual no estilo IRaMuTeQ: Classificação Hierárquica Descendente (Reinert), "
        "tradução para português, limpeza de corpus, AFC, similitude e interpretação das classes "
        "por LLM local. Os nomes das classes são sugestões da IA: apresente-os como tal e "
        "lembre o pesquisador de validá-los (comando 'validar' do Textome)."
    ),
)


# Fábrica da ponte com o R; os testes podem trocar por uma versão simulada.
BRIDGE_FACTORY = None


def data_dir() -> Path:
    return Path(os.environ.get("TEXTOME_DADOS", os.getcwd())).resolve()


def safe_path(relative: str, must_exist: bool = True) -> Path:
    """Resolve um caminho dentro de TEXTOME_DADOS; recusa qualquer coisa fora dela."""
    base = data_dir()
    path = (base / relative).resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"Caminho fora da pasta de dados ({base}): {relative}")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Não encontrado em {base}: {relative}")
    return path


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _summary(result_dir: Path, max_forms: int = 10) -> Dict[str, Any]:
    data = json.loads((result_dir / "classes.json").read_text(encoding="utf-8"))
    total = sum(c["segmentos_total"] for c in data["classes"]) or 1
    classes = []
    for c in data["classes"]:
        interp = c.get("interpretacao") or {}
        classes.append({
            "classe": c["id"],
            "nome_sugerido_pela_ia": c["nome"],
            "descricao": interp.get("descricao", ""),
            "resumo": interp.get("resumo", ""),
            "segmentos": c["segmentos_total"],
            "pct_corpus": round(100 * c["segmentos_total"] / total, 1),
            "formas_caracteristicas": [f["forma"] for f in c.get("formas", [])][:max_forms],
            "exemplo_de_segmento": (c.get("segmentos") or [""])[0],
        })
    out: Dict[str, Any] = {
        "pasta": str(result_dir.relative_to(data_dir())),
        "classes": classes,
        "avisos": data.get("avisos", []),
        "arquivos": sorted(str(p.relative_to(result_dir)) for p in result_dir.rglob("*") if p.is_file()),
    }
    tri = result_dir / "triangulacao.json"
    if tri.exists():
        t = json.loads(tri.read_text(encoding="utf-8"))
        out["triangulacao"] = {"ari": round(t["ari"], 3), "nmi": round(t["nmi"], 3), "metodo": t["method"]}
    val = result_dir / "validacao.json"
    if val.exists():
        out["validacao"] = json.loads(val.read_text(encoding="utf-8")).get("resultados")
    return out


@mcp.tool()
def listar_corpora() -> List[str]:
    """Lista os arquivos .txt, .csv e .xlsx disponíveis na pasta de dados."""
    base = data_dir()
    return sorted(str(p.relative_to(base)) for ext in ("*.txt", "*.csv", "*.xlsx")
                  for p in base.rglob(ext) if "resultados" not in p.parts)


@mcp.tool()
def analisar_corpus(
    arquivo: str,
    k: int = 5,
    traduzir: bool = True,
    limpar: bool = True,
    interpretar: bool = True,
    topicos: bool = False,
    termos_compostos: Optional[List[str]] = None,
    modelo: Optional[str] = None,
    saida: Optional[str] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """
    Análise completa de um corpus (.txt livre ou formato IRaMuTeQ ****): tradução para
    português, limpeza IRaMuTeQ, CHD (método Reinert, k classes), interpretação das
    classes pela IA local, AFC, similitude, nuvem e relatório Word. Com topicos=True,
    também triangula com BERTopic. Pode levar alguns minutos.
    Retorna as classes (nome sugerido, formas, % do corpus) e a lista de arquivos gerados.
    """
    from pipeline import Config, load_documents, run_pipeline, write_outputs

    src = safe_path(arquivo)
    out = safe_path(saida or f"resultados/{src.stem}", must_exist=False)
    cfg = Config(k=k, translate=traduzir, clean=limpar, use_llm=interpretar, topics=topicos,
                 compound_terms=list(termos_compostos or []), mock=mock,
                 **({"model": modelo} if modelo else {}))
    kwargs = {"bridge_factory": BRIDGE_FACTORY} if BRIDGE_FACTORY else {}
    result = run_pipeline(load_documents(src), cfg, source_name=src.name, log=_log, **kwargs)
    write_outputs(result, out)
    return _summary(out)


@mcp.tool()
def ler_resultado(pasta: str) -> Dict[str, Any]:
    """Resume uma pasta de resultados já gerada (classes, triangulação, validação, arquivos)."""
    return _summary(safe_path(pasta))


@mcp.tool()
def segmentos_da_classe(pasta: str, classe: int, limite: int = 10) -> List[str]:
    """Devolve segmentos de texto de uma classe, para examinar o conteúdo com o pesquisador."""
    data = json.loads((safe_path(pasta) / "classes.json").read_text(encoding="utf-8"))
    todos = [s["texto"] for s in data.get("todos_segmentos", []) if s["classe"] == classe]
    if not todos:
        todos = next((c.get("segmentos", []) for c in data["classes"] if c["id"] == classe), [])
    return todos[: max(1, min(limite, 100))]


@mcp.tool()
def traduzir_corpus(arquivo: str, idioma_origem: str = "auto", modelo: Optional[str] = None,
                    mock: bool = False) -> Dict[str, Any]:
    """Traduz para português os textos que não estão em português; grava <arquivo>_pt.txt."""
    from corpus import to_iramuteq
    from pipeline import load_documents
    from translator import CorpusTranslator, language_summary

    src = safe_path(arquivo)
    docs = load_documents(src)
    tr = CorpusTranslator(**({"model": modelo} if modelo else {}), mock=mock)
    results = tr.translate_corpus([d.text for d in docs], source_lang=idioma_origem)
    out = src.with_name(f"{src.stem}_pt.txt")
    out.write_text(to_iramuteq([r.text for r in results], [d.header for d in docs],
                               [f"*lang_{r.source_lang}" for r in results]), encoding="utf-8")
    return {"arquivo": str(out.relative_to(data_dir())), "idiomas": language_summary(results),
            "traduzidos": sum(r.translated for r in results)}


@mcp.tool()
def limpar_corpus(arquivo: str, termos_compostos: Optional[List[str]] = None) -> Dict[str, Any]:
    """Aplica a limpeza IRaMuTeQ (aspas, hífens, expressões compostas…); grava <arquivo>_limpo.txt."""
    from corpus import to_iramuteq
    from limpeza import clean_documents
    from pipeline import load_documents

    src = safe_path(arquivo)
    docs, report = clean_documents(load_documents(src), termos_compostos or [])
    out = src.with_name(f"{src.stem}_limpo.txt")
    out.write_text(to_iramuteq([d.text for d in docs], [d.header for d in docs]), encoding="utf-8")
    return {"arquivo": str(out.relative_to(data_dir())), "alteracoes": report.summary_lines()}


def main() -> None:
    """
    No transporte stdio, a saída padrão é o canal do protocolo. Tudo o que o pipeline
    (ou o R) imprimir vai para stderr; só o MCP escreve no stdout original.
    """
    import anyio
    from mcp.server.stdio import stdio_server

    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())   # saída de C/R → stderr
    sys.stdout = sys.stderr                              # print() → stderr
    try:
        from rpy2.rinterface_lib import callbacks

        callbacks.consolewrite_print = lambda s: sys.stderr.write(s)
        callbacks.consolewrite_warnerror = lambda s: sys.stderr.write(s)
    except Exception:
        pass

    async def run() -> None:
        out = anyio.wrap_file(io.TextIOWrapper(protocol_out, encoding="utf-8", write_through=True))
        async with stdio_server(stdout=out) as (read_stream, write_stream):
            server = mcp._mcp_server
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()
