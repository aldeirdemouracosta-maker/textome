"""
Testes da linha de comando e do pipeline (rainette simulado, LLM mock).
"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import yaml

import cli
from pipeline import Config


class FakeBridge:
    """Imita RainetteBridge: duas classes fixas + uma vazia."""

    def __init__(self):
        self.calls = []

    def run(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {
            1: {"forms": [("hospital", 12.3), ("fila", 8.0)], "segments": texts[:2]},
            2: {"forms": [("escola", 10.1)], "segments": texts[2:3]},
            3: {"forms": [], "segments": []},
        }

    def get_group_sizes(self):
        return {1: 2, 2: 1}


def _run(argv, bridge):
    args = cli.build_parser().parse_args(argv)
    return cli.cmd_run(args, bridge_factory=lambda: bridge)


def test_config_exemplo_e_valida():
    cfg = Config.from_dict(yaml.safe_load(cli.EXAMPLE_CONFIG))
    assert cfg == Config()
    try:
        Config.from_dict({"kk": 3})
        raise AssertionError("chave desconhecida deveria falhar")
    except ValueError as e:
        assert "kk" in str(e)
    print("OK test_config_exemplo_e_valida")


def test_run_completo_com_traducao():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "entrevistas.txt"
        corpus.write_text(
            "**** *suj_01 *pais_us\nThe hospitals are full and the waiting lists are long.\n"
            "**** *suj_02 *pais_br\nOs hospitais estão lotados e as filas são longas.\n"
            "**** *suj_03 *pais_es\nLas escuelas no tienen profesores suficientes para todos.\n",
            encoding="utf-8",
        )
        conf = tmp / "pesquisa.yaml"
        conf.write_text(f"k: 3\ncache_db: {tmp / 'c.db'}\n", encoding="utf-8")
        bridge = FakeBridge()
        with redirect_stdout(io.StringIO()):
            code = _run(["run", str(corpus), "--config", str(conf), "--mock",
                         "--saida", str(tmp / "out")], bridge)
        assert code == 0

        texts, kwargs = bridge.calls[0]
        assert kwargs["k"] == 3 and kwargs["language"] == "pt" and kwargs["seed"] == 42
        assert texts[0].startswith("[tradução simulada]") and texts[1].startswith("Os hospitais")

        out = tmp / "out"
        nomes = sorted(p.name for p in out.iterdir())
        assert nomes == ["classes.json", "config_usada.yaml", "corpus_pt.txt",
                         "formas.csv", "relatorio.md"], nomes

        dados = json.loads((out / "classes.json").read_text(encoding="utf-8"))
        assert [c["id"] for c in dados["classes"]] == [1, 2, 3]
        assert dados["classes"][0]["nome"] == "Saúde pública e acesso"
        assert dados["classes"][2]["interpretacao"] == {}, "classe vazia não vai ao LLM"
        assert dados["avisos"], "classe vazia gera aviso"

        corpus_pt = (out / "corpus_pt.txt").read_text(encoding="utf-8")
        assert "**** *suj_01 *pais_us *lang_en\n[tradução simulada]" in corpus_pt
        assert "**** *suj_02 *pais_br *lang_pt\nOs hospitais" in corpus_pt

        rel = (out / "relatorio.md").read_text(encoding="utf-8")
        assert "k = 3" in rel and "modo simulado" in rel and "| hospital | 12.3 |" in rel
        assert "Tradução automática para o português de 2 documento(s)" in rel

        usada = yaml.safe_load((out / "config_usada.yaml").read_text(encoding="utf-8"))
        assert usada["k"] == 3 and usada["mock"] is True

        csv_txt = (out / "formas.csv").read_text(encoding="utf-8-sig")
        assert csv_txt.splitlines()[0] == "classe;nome;forma;chi2"
    print("OK test_run_completo_com_traducao")


def test_lote_de_pasta_continua_apos_erro():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pasta = tmp / "corpora"
        pasta.mkdir()
        (pasta / "a.txt").write_text("Primeiro texto.\n\nSegundo texto.\n\nTerceiro.", encoding="utf-8")
        (pasta / "b.txt").write_text("", encoding="utf-8")  # vazio → falha
        (pasta / "c.txt").write_text("Outro texto.\n\nMais um.\n\nFim.", encoding="utf-8")
        db = tmp / "c.db"
        conf = tmp / "p.yaml"
        conf.write_text(f"cache_db: {db}\n", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            code = _run(["run", str(pasta), "--config", str(conf), "--mock", "--sem-traducao",
                         "--sem-llm", "--saida", str(tmp / "out")], FakeBridge())
        assert code == 1, "um arquivo falhou"
        assert (tmp / "out" / "a" / "relatorio.md").exists()
        assert (tmp / "out" / "c" / "relatorio.md").exists()
        assert not (tmp / "out" / "b").exists()
        assert not (tmp / "out" / "a" / "corpus_pt.txt").exists(), "sem tradução, sem corpus_pt"
    print("OK test_lote_de_pasta_continua_apos_erro")


def test_comando_traduzir():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        f = tmp / "en.txt"
        f.write_text("The schools need more teachers.\n\nThe hospitals are full.", encoding="utf-8")
        conf = tmp / "p.yaml"
        conf.write_text(f"cache_db: {tmp / 'c.db'}\n", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            code = cli.main(["traduzir", str(f), "--config", str(conf), "--mock",
                             "--saida", str(tmp / "tr")])
        assert code == 0
        out = (tmp / "tr" / "en_pt.txt").read_text(encoding="utf-8")
        assert out.startswith("**** *doc_001 *lang_en\n[tradução simulada] The schools")
    print("OK test_comando_traduzir")


def test_entrada_inexistente():
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()) as err:
        assert cli.main(["run", "nao_existe.txt"]) == 2
    assert "Não encontrado" in err.getvalue()
    print("OK test_entrada_inexistente")


if __name__ == "__main__":
    test_config_exemplo_e_valida()
    test_run_completo_com_traducao()
    test_lote_de_pasta_continua_apos_erro()
    test_comando_traduzir()
    test_entrada_inexistente()
    print("\n=== CLI OK ===")
