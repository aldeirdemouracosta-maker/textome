"""
Cliente MCP de exemplo/teste: sobe o servidor do Textome via stdio e chama ferramentas.

  python exemplos/cliente_mcp.py exemplos/corpus_demo.txt [--mock] [--k 4]

Usado no CI (dentro do Docker, com R) para testar a análise completa pelo protocolo MCP.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

RAIZ = Path(__file__).resolve().parents[1]


async def main(arquivo: str, k: int, mock: bool) -> int:
    params = StdioServerParameters(
        command=sys.executable, args=[str(RAIZ / "mcp_server.py")],
        env={**os.environ, "TEXTOME_DADOS": str(Path(arquivo).resolve().parent)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(t.name for t in (await session.list_tools()).tools)
            print("ferramentas:", tools)
            res = await session.call_tool("analisar_corpus", {
                "arquivo": Path(arquivo).name, "k": k, "mock": mock, "saida": "resultados_mcp",
            })
            if res.isError:
                print("ERRO:", res.content[0].text)
                return 1
            resumo = json.loads(res.content[0].text)
            for c in resumo["classes"]:
                print(f"  classe {c['classe']}: {c['pct_corpus']}% · {', '.join(c['formas_caracteristicas'][:5])}")
            print("arquivos:", len(resumo["arquivos"]))
            assert resumo["classes"] and "relatorio.docx" in resumo["arquivos"]
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.arquivo, a.k, a.mock)))
