"""
Testes da validação humana (nomes, atribuição às cegas, kappa) e das
saídas estruturadas da IA.
"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import cli
from llm_interpreter import NAME_SCHEMA, ClassInterpreter, assignment_schema
from test_analises import BridgeComMatriz
from validacao import (
    ValidationSession, cohen_kappa, decide_name, interpret_kappa, sample_items,
)


def test_kappa_calculado_a_mao():
    # po = 0,75; pe = (2·1 + 2·3)/16 = 0,5 → κ = (0,75 − 0,5)/(1 − 0,5) = 0,5
    ag = cohen_kappa([1, 1, 2, 2], [1, 2, 2, 2])
    assert abs(ag.observed - 0.75) < 1e-9 and abs(ag.kappa - 0.5) < 1e-9
    assert ag.matrix == [[1, 1], [0, 2]] and ag.interpretation == "moderada"
    assert cohen_kappa([1, 2, 3], [1, 2, 3]).kappa == 1.0
    assert cohen_kappa([1, 1], [1, 1]).kappa is None, "sem variação: kappa indefinido"
    assert cohen_kappa([1, 2], [2, 1]).kappa < 0
    assert [interpret_kappa(k) for k in (-0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)] == [
        "pobre", "leve", "leve", "razoável", "moderada", "substancial", "quase perfeita",
        "quase perfeita"]
    print("OK test_kappa_calculado_a_mao")


def test_decisoes_de_nome():
    assert decide_name(1, "Saúde", "").decision == "aceito"
    assert decide_name(1, "Saúde", "Saúde").decision == "aceito"
    d = decide_name(1, "Saúde", "Acesso ao SUS")
    assert (d.decision, d.final_name) == ("editado", "Acesso ao SUS")
    d = decide_name(2, "Algo", "-")
    assert (d.decision, d.final_name) == ("rejeitado", "Classe 2")
    print("OK test_decisoes_de_nome")


def test_sorteio_estratificado_sem_vazamento():
    segs = [(f"c1 texto {i}", 1) for i in range(10)] + [(f"c2 texto {i}", 2) for i in range(3)] \
        + [("sem classe", 0), ("c1 texto 0", 1)]
    items = sample_items(segs, exclude=["c1 texto 1"], per_class=4, seed=7)
    by_class = {}
    for it in items:
        by_class.setdefault(it.chd_class, []).append(it.text)
    assert len(by_class[1]) == 4 and len(by_class[2]) == 3 and 0 not in by_class
    assert "c1 texto 1" not in by_class[1], "segmento mostrado à IA não entra no teste"
    assert len(set(by_class[1])) == 4, "sem duplicatas"
    assert [i.text for i in sample_items(segs, per_class=4, seed=7)] == \
        [i.text for i in sample_items(segs, per_class=4, seed=7)], "reprodutível"
    print("OK test_sorteio_estratificado_sem_vazamento")


def test_sessao_ida_e_volta_json():
    s = ValidationSession(names=[decide_name(1, "A", "B")], descriptions={1: "desc"})
    s.items = sample_items([("x y", 1), ("z w", 1)], per_class=2)
    s.items[0].human, s.items[1].human = 1, 0
    data = json.loads(s.to_json())
    assert data["resultados"]["aceitacao_nomes"] == {"aceito": 0, "editado": 1, "rejeitado": 0}
    s2 = ValidationSession.from_json(s.to_json())
    assert s2.final_names() == {1: "B"} and s2.descriptions == {1: "desc"}
    assert [i.human for i in s2.items] == [1, 0]
    print("OK test_sessao_ida_e_volta_json")


def test_saidas_estruturadas_e_atribuicao_mock():
    assert NAME_SCHEMA["required"] == ["nome", "descricao"]
    assert assignment_schema([3, 1])["properties"]["classe"]["enum"] == [1, 3, 0]

    calls = []

    class Spy:
        def chat(self, model, messages, options=None, **kw):
            calls.append(kw.get("format"))
            if isinstance(kw.get("format"), dict):
                raise RuntimeError("invalid format: schema not supported")  # Ollama antigo
            return {"message": {"content": '{"nome": "X", "descricao": "Y"}'}}

    li = ClassInterpreter(use_cache=False, mock=True)
    li.client = Spy()
    li.interpret_class(1, ["a"], ["b"])
    assert isinstance(calls[0], dict) and calls[1] == "json", "esquema com recuo para json"

    li = ClassInterpreter(use_cache=False, mock=True)
    opcoes = {1: "Saúde — hospitais filas atendimento", 2: "Educação — escolas professores"}
    assert li.atribuir_segmento("As escolas precisam de professores", opcoes) == 2
    assert li.atribuir_segmento("Os hospitais têm filas enormes", opcoes) == 1
    assert li.atribuir_segmento("zzz", opcoes) == 0
    print("OK test_saidas_estruturadas_e_atribuicao_mock")


TEXTOS = (
    ["Os hospitais têm filas enormes e falta atendimento médico."] * 1
    + [f"Hospital {i}: filas, leitos e atendimento médico demorado." for i in range(8)]
    + [f"Escola {i}: professores, alunos e materiais didáticos." for i in range(8)]
    + [f"Bairro {i}: violência, policiamento e segurança pública." for i in range(8)]
)


class BridgeTemas(BridgeComMatriz):
    """Classes coerentes por tema, com todos os segmentos disponíveis."""

    def run(self, texts, **kwargs):
        self.texts = texts
        self.groups = [1 if "ospita" in t else 2 if "Escola" in t else 3 for t in texts]
        return {cid: {"forms": [], "segments": [t for t, g in zip(texts, self.groups) if g == cid][:2]}
                for cid in (1, 2, 3)}

    def get_segments(self):
        return list(zip(self.texts, self.groups))


def test_validacao_completa_pela_linha_de_comando():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "c.txt"
        corpus.write_text("\n".join(TEXTOS), encoding="utf-8")
        conf = tmp / "p.yaml"
        conf.write_text(f"cache_db: {tmp / 'c.db'}\ntranslate: false\nuse_llm: false\n", encoding="utf-8")
        out = tmp / "out"
        args = cli.build_parser().parse_args(["run", str(corpus), "--config", str(conf), "--saida", str(out)])
        with redirect_stdout(io.StringIO()):
            assert cli.cmd_run(args, bridge_factory=BridgeTemas) == 0
        dados = json.loads((out / "classes.json").read_text(encoding="utf-8"))
        assert len(dados["todos_segmentos"]) == len(TEXTOS)

        # Pesquisador: renomeia as 3 classes e acerta todos os segmentos, exceto um.
        respostas = iter(["Saúde e hospitais", "Educação escolar", "-"])
        sessao_prev = __import__("validacao").session_from_outputs(out, per_class=3)
        gabarito = [it.chd_class for it in sessao_prev.items]
        gabarito[0] = 0  # um "não sei"
        respostas = iter(["Saúde e hospitais", "Educação escolar", "-", "9"] + [str(g) for g in gabarito])

        args = cli.build_parser().parse_args(["validar", str(out), "--por-classe", "3", "--ia", "--mock"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert cli.cmd_validar(args, ask=lambda _: next(respostas)) == 0
        log = buf.getvalue()
        assert "Digite um destes números" in log, "resposta inválida (9) é recusada"

        val = json.loads((out / "validacao.json").read_text(encoding="utf-8"))
        assert val["resultados"]["aceitacao_nomes"] == {"aceito": 0, "editado": 2, "rejeitado": 1}
        conc = val["resultados"]["concordancia"]
        assert set(conc) == {"pesquisador × CHD", "IA × CHD", "pesquisador × IA"}
        assert abs(conc["pesquisador × CHD"]["observed"] - 8 / 9) < 1e-9
        assert conc["pesquisador × CHD"]["interpretation"] in ("substancial", "quase perfeita")
        md = (out / "validacao.md").read_text(encoding="utf-8")
        assert "kappa de Cohen" in md and "| 3 | Classe 3 | Classe 3 | rejeitado |" in md
        assert (out / "validacao.docx").exists()
    print("OK test_validacao_completa_pela_linha_de_comando")


if __name__ == "__main__":
    test_kappa_calculado_a_mao()
    test_decisoes_de_nome()
    test_sorteio_estratificado_sem_vazamento()
    test_sessao_ida_e_volta_json()
    test_saidas_estruturadas_e_atribuicao_mock()
    test_validacao_completa_pela_linha_de_comando()
    print("\n=== VALIDAÇÃO OK ===")
