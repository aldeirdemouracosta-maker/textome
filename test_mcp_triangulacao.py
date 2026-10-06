"""
Testes da triangulação (ARI/NMI, tabela, figura, relatórios) e do servidor MCP
(protocolo real via stdio + ferramentas em processo com rainette simulado).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import mcp_server
import triangulacao
from triangulacao import adjusted_rand_index, compare, normalized_mutual_info
from test_validacao import TEXTOS, BridgeTemas

RAIZ = Path(__file__).resolve().parent


def test_metricas_de_particao():
    a = [1, 1, 2, 2, 3, 3]
    assert adjusted_rand_index(a, [7, 7, 8, 8, 9, 9]) == 1.0, "rótulos diferentes, mesma partição"
    assert abs(normalized_mutual_info(a, [7, 7, 8, 8, 9, 9]) - 1.0) < 1e-12
    assert adjusted_rand_index(a, [1, 2, 1, 2, 1, 2]) < 0.1
    # valor conhecido (exemplo clássico do scikit-learn): ARI([0,0,1,1],[0,0,1,2]) = 0,5714…
    assert abs(adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) - 0.5714285714) < 1e-9
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        pass
    else:
        import random
        rng = random.Random(3)
        for _ in range(50):
            x = [rng.randint(1, 4) for _ in range(30)]
            y = [rng.randint(0, 5) for _ in range(30)]
            assert abs(adjusted_rand_index(x, y) - adjusted_rand_score(x, y)) < 1e-9
            assert abs(normalized_mutual_info(x, y) - normalized_mutual_info_score(x, y)) < 1e-9
    print("OK test_metricas_de_particao")


def test_comparacao_ignora_sem_classe_e_outliers():
    chd = [1, 1, 1, 2, 2, 2, 0, 3]
    top = [5, 5, 6, 9, 9, 9, 5, -1]
    tri = compare(chd, top, topic_words={5: ["saude"], 9: ["escola"]})
    assert tri.n_segments == 6 and tri.outliers == 1
    assert tri.class_ids == [1, 2] and tri.topic_ids == [5, 6, 9]
    assert tri.best_topic[1] == {"topico": 5, "pct_classe": 66.7}
    assert tri.best_topic[2] == {"topico": 9, "pct_classe": 100.0}
    assert "ARI" in tri.summary() and "1 segmentos sem tópico" in tri.summary()
    print("OK test_comparacao_ignora_sem_classe_e_outliers")


def fake_bertopic(texts, n_topics=None, **kw):
    """Tópicos 'semânticos' simulados: concordam com os temas, com um erro."""
    topics = [10 if "ospita" in t else 20 if "Escola" in t else 30 for t in texts]
    topics[-1] = 10
    return topics, {10: ["hospital", "filas"], 20: ["escola", "professores"], 30: ["bairro", "violência"]}


def _run_with_topics(out: Path, tmp: Path) -> None:
    import cli

    corpus = tmp / "c.txt"
    corpus.write_text("\n".join(TEXTOS), encoding="utf-8")
    conf = tmp / "p.yaml"
    conf.write_text(f"cache_db: {tmp / 'c.db'}\ntranslate: false\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        ["run", str(corpus), "--config", str(conf), "--mock", "--topicos", "--saida", str(out)])
    original = triangulacao.run_bertopic
    triangulacao.run_bertopic = fake_bertopic
    try:
        with redirect_stdout(io.StringIO()):
            assert cli.cmd_run(args, bridge_factory=BridgeTemas) == 0
    finally:
        triangulacao.run_bertopic = original


def test_pipeline_com_triangulacao():
    from docx import Document as Docx

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "out"
        _run_with_topics(out, tmp)
        tri = json.loads((out / "triangulacao.json").read_text(encoding="utf-8"))
        assert 0.8 < tri["ari"] < 1.0 and tri["best_topic"]["2"]["topico"] == 20
        assert (out / "figuras" / "triangulacao.png").stat().st_size > 5000
        md = (out / "relatorio.md").read_text(encoding="utf-8")
        assert "## Triangulação CHD × BERTopic" in md and "índice de Rand ajustado" in md
        text = "\n".join(p.text for p in Docx(str(out / "relatorio.docx")).paragraphs)
        assert "Triangulação CHD × BERTopic" in text and "Triangulação: classes da CHD" in text
    print("OK test_pipeline_com_triangulacao")


def test_bertopic_ausente_vira_aviso():
    import cli

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "c.txt"
        corpus.write_text("\n".join(TEXTOS), encoding="utf-8")
        conf = tmp / "p.yaml"
        conf.write_text(f"cache_db: {tmp / 'c.db'}\ntranslate: false\nuse_llm: false\ntopics: true\n",
                        encoding="utf-8")
        original = triangulacao.run_bertopic

        def sem_bertopic(*a, **k):
            raise ImportError("Triangulação requer: pip install -r requirements-topicos.txt")

        triangulacao.run_bertopic = sem_bertopic
        try:
            args = cli.build_parser().parse_args(["run", str(corpus), "--config", str(conf),
                                                  "--saida", str(tmp / "o")])
            with redirect_stdout(io.StringIO()):
                assert cli.cmd_run(args, bridge_factory=BridgeTemas) == 0
        finally:
            triangulacao.run_bertopic = original
        md = (tmp / "o" / "relatorio.md").read_text(encoding="utf-8")
        assert "requirements-topicos.txt" in md, "falta do BERTopic aparece como aviso"
    print("OK test_bertopic_ausente_vira_aviso")


def test_caminhos_fora_da_pasta_sao_recusados():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TEXTOME_DADOS"] = tmp
        try:
            (Path(tmp) / "a.txt").write_text("x", encoding="utf-8")
            assert mcp_server.safe_path("a.txt").name == "a.txt"
            for ruim in ("../etc/passwd", "/etc/passwd", "sub/../../x"):
                try:
                    mcp_server.safe_path(ruim, must_exist=False)
                    raise AssertionError(f"deveria recusar {ruim}")
                except ValueError:
                    pass
        finally:
            del os.environ["TEXTOME_DADOS"]
    print("OK test_caminhos_fora_da_pasta_sao_recusados")


def test_ferramenta_analisar_em_processo():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        os.environ["TEXTOME_DADOS"] = str(tmp)
        (tmp / "entrevistas.txt").write_text("\n".join(TEXTOS), encoding="utf-8")
        mcp_server.BRIDGE_FACTORY = BridgeTemas
        cwd = os.getcwd()
        os.chdir(tmp)  # cache .cache/ fica no tmp
        try:
            with redirect_stdout(io.StringIO()):
                resumo = mcp_server.analisar_corpus("entrevistas.txt", k=3, traduzir=False, mock=True)
        finally:
            os.chdir(cwd)
            mcp_server.BRIDGE_FACTORY = None
            del os.environ["TEXTOME_DADOS"]
        assert resumo["pasta"] == "resultados/entrevistas"
        assert [c["classe"] for c in resumo["classes"]] == [1, 2, 3]
        assert "relatorio.docx" in resumo["arquivos"] and "figuras/afc.png" in resumo["arquivos"]
    print("OK test_ferramenta_analisar_em_processo")


async def _protocol_session(tmp: Path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=[str(RAIZ / "mcp_server.py")],
        env={**os.environ, "TEXTOME_DADOS": str(tmp), "PYTHONPATH": str(RAIZ)},
        cwd=str(tmp),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = sorted(t.name for t in (await s.list_tools()).tools)
            # ferramentas que devolvem lista: um item de conteúdo por elemento
            corpora = [c.text for c in (await s.call_tool("listar_corpora", {})).content]
            limpo = await s.call_tool("limpar_corpus", {"arquivo": "bruto.txt",
                                                        "termos_compostos": ["posto de saúde"]})
            trad = await s.call_tool("traduzir_corpus", {"arquivo": "en.txt", "mock": True})
            fora = await s.call_tool("limpar_corpus", {"arquivo": "../fora.txt"})
            lido = await s.call_tool("ler_resultado", {"pasta": "resultados/c"})
            segs = await s.call_tool("segmentos_da_classe", {"pasta": "resultados/c", "classe": 2,
                                                             "limite": 3})
            return tools, corpora, limpo, trad, fora, lido, segs


def test_protocolo_mcp_via_stdio():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "bruto.txt").write_text('O "posto de saúde" fecha cedo... disse-me ela.\n\nOutro texto.',
                                       encoding="utf-8")
        (tmp / "en.txt").write_text("The hospitals are full and the waiting lists are long.",
                                    encoding="utf-8")
        _run_with_topics(tmp / "resultados" / "c", tmp)

        tools, corpora, limpo, trad, fora, lido, segs = asyncio.run(_protocol_session(tmp))
        assert tools == ["analisar_corpus", "ler_resultado", "limpar_corpus", "listar_corpora",
                         "segmentos_da_classe", "traduzir_corpus"]
        assert "bruto.txt" in corpora and "en.txt" in corpora
        assert not limpo.isError
        assert (tmp / "bruto_limpo.txt").read_text(encoding="utf-8").startswith(
            "**** *doc_001\nO posto_de_saúde fecha cedo. disse me ela.")
        assert not trad.isError and json.loads(trad.content[0].text)["traduzidos"] == 1
        assert fora.isError and "fora da pasta" in fora.content[0].text
        resumo = json.loads(lido.content[0].text)
        assert resumo["triangulacao"]["ari"] > 0.8 and len(resumo["classes"]) == 3
        assert [c.text[:6] for c in segs.content] == ["Escola"] * 3
    print("OK test_protocolo_mcp_via_stdio")


if __name__ == "__main__":
    test_metricas_de_particao()
    test_comparacao_ignora_sem_classe_e_outliers()
    test_pipeline_com_triangulacao()
    test_bertopic_ausente_vira_aviso()
    test_caminhos_fora_da_pasta_sao_recusados()
    test_ferramenta_analisar_em_processo()
    test_protocolo_mcp_via_stdio()
    print("\n=== TRIANGULAÇÃO E MCP OK ===")
