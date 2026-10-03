"""
Testes da tradução para português e das correções da revisão
(sem Ollama real e sem R).
"""
from __future__ import annotations

import tempfile
import types
from pathlib import Path

import llm_interpreter
from corpus import decode_bytes, parse_corpus, to_iramuteq
from llm_interpreter import LLMInterpreter, list_available_models
from translator import CorpusTranslator, detect_language, split_chunks


def test_texto_livre_vira_varios_documentos():
    linhas = "\n".join(f"Frase número {i} sobre saúde e escola." for i in range(20))
    assert len(parse_corpus(linhas)) == 20

    paragrafos = "Primeiro parágrafo\ncontinua aqui.\n\nSegundo parágrafo."
    docs = parse_corpus(paragrafos)
    assert [d.text for d in docs] == ["Primeiro parágrafo\ncontinua aqui.", "Segundo parágrafo."]
    print("OK test_texto_livre_vira_varios_documentos")


def test_formato_iramuteq_preserva_variaveis():
    raw = "**** *suj_01 *sexo_f\nTexto um.\n\n**** *suj_02 *sexo_m\nTexto dois.\n"
    docs = parse_corpus(raw)
    assert [(d.header, d.text) for d in docs] == [
        ("**** *suj_01 *sexo_f", "Texto um."),
        ("**** *suj_02 *sexo_m", "Texto dois."),
    ]
    out = to_iramuteq([d.text for d in docs], [d.header for d in docs], ["*lang_en", ""])
    assert out.startswith("**** *suj_01 *sexo_f *lang_en\nTexto um.")
    assert "**** *suj_02 *sexo_m\nTexto dois." in out
    assert to_iramuteq(["sem cabeçalho"]).startswith("**** *doc_001\n")
    print("OK test_formato_iramuteq_preserva_variaveis")


def test_decode_windows():
    assert decode_bytes("ação".encode("cp1252")) == "ação"
    assert decode_bytes("﻿ação".encode("utf-8")) == "ação"
    print("OK test_decode_windows")


def test_deteccao_de_idioma():
    casos = {
        "pt": "A saúde pública não tem recursos e os hospitais são precários.",
        "en": "The public health system is underfunded and the hospitals are in poor shape.",
        "es": "El sistema de salud pública no tiene recursos y los hospitales están mal.",
        "fr": "Le système de santé manque de moyens et les hôpitaux sont en mauvais état.",
        "de": "Das Gesundheitssystem ist unterfinanziert und die Krankenhäuser sind schlecht.",
        "it": "Il sistema sanitario non ha risorse e gli ospedali sono in cattive condizioni.",
        "outro": "Система здравоохранения недофинансирована и больницы в плохом состоянии.",
    }
    for esperado, texto in casos.items():
        assert detect_language(texto) == esperado, (esperado, detect_language(texto))
    assert detect_language("公共卫生系统资金不足，医院状况很差。") == "outro"
    print("OK test_deteccao_de_idioma")


def test_split_chunks_respeita_limite():
    chunks = split_chunks("Uma frase de teste. " * 500, max_chars=500)
    assert len(chunks) > 1 and all(len(c) <= 500 for c in chunks)
    print("OK test_split_chunks_respeita_limite")


def test_traducao_mock_com_cache():
    with tempfile.TemporaryDirectory() as tmp:
        tr = CorpusTranslator(model="m", db_path=Path(tmp) / "t.db", mock=True)
        textos = [
            "The hospitals are overcrowded and the waiting lists are long.",
            "Os hospitais estão lotados e as filas são longas.",
        ]
        r = tr.translate_corpus(textos)
        assert r[0].translated and r[0].source_lang == "en"
        assert r[0].text.startswith("[tradução simulada] The hospitals")
        assert not r[1].translated and r[1].text == textos[1], "pt não é traduzido"

        r2 = tr.translate(textos[0])
        assert r2.cached and r2.text == r[0].text
        assert not tr.translate(textos[0], force_refresh=True).cached

        fixo = tr.translate("Hola a todos", source_lang="es")
        assert fixo.source_lang == "es" and fixo.translated
    print("OK test_traducao_mock_com_cache")


def test_mock_nao_contamina_cache_do_modelo_real():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "c.db"
        mock = LLMInterpreter(model="qwen2.5:7b", db_path=db, mock=True)
        assert mock.interpret_class(1, ["saude"], ["seg"]).nome

        real = LLMInterpreter(model="qwen2.5:7b", db_path=db)
        assert not real.is_mock
        assert real.cache.get(
            real.cache.make_key(["saude"], ["seg"], real.model_key, real.temperature, False)
        ) is None, "resposta simulada não pode servir o modelo real"

        tr_mock = CorpusTranslator(model="qwen2.5:7b", db_path=db, mock=True)
        tr_mock.translate("The hospitals are full today.")
        tr_real = CorpusTranslator(model="qwen2.5:7b", db_path=db)
        assert tr_real._key("x", "en") != tr_mock._key("x", "en")
    print("OK test_mock_nao_contamina_cache_do_modelo_real")


def test_sem_expiracao_funciona():
    with tempfile.TemporaryDirectory() as tmp:
        li = LLMInterpreter(
            model="m", db_path=Path(tmp) / "t.db", mock=True,
            ttl_name_hours=None, ttl_deep_hours=None,
        )
        li.interpret_class(1, ["a"], ["s"], deep=False)
        li.interpret_class(1, ["a"], ["s"], deep=True)
        assert [r["expires_at"] for r in li.list_recent_cached()] == [None, None]

        li2 = LLMInterpreter(model="m", db_path=Path(tmp) / "u.db", mock=True)
        li2.interpret_class(1, ["a"], ["s"])
        assert li2.list_recent_cached()[0]["expires_at"] is not None, "padrão: 30 dias"
    print("OK test_sem_expiracao_funciona")


def test_lista_de_modelos_ollama_novo_e_antigo():
    class Novo:  # ollama>=0.4: objetos com campo "model"
        def __init__(self, model):
            self.model = model

    respostas = [
        types.SimpleNamespace(models=[Novo("qwen2.5:7b"), Novo("llama3.1:8b")]),
        {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.1:8b"}]},
    ]
    original = llm_interpreter.ollama
    try:
        for resp in respostas:
            llm_interpreter.ollama = types.SimpleNamespace(
                Client=lambda host=None, r=resp: types.SimpleNamespace(list=lambda: r)
            )
            assert list_available_models() == ["qwen2.5:7b", "llama3.1:8b"]
    finally:
        llm_interpreter.ollama = original
    print("OK test_lista_de_modelos_ollama_novo_e_antigo")


def test_remove_raciocinio_e_json_invalido():
    assert llm_interpreter.strip_thinking("<think>pensando…</think>\nResposta") == "Resposta"
    li = LLMInterpreter(use_cache=False, mock=True)
    assert li._extract_json('["lista"]')["nome"] == "Classe sem nome"
    assert li._extract_json('{"nome": 3, "descricao": null}') == {"nome": "3"}
    print("OK test_remove_raciocinio_e_json_invalido")


if __name__ == "__main__":
    test_texto_livre_vira_varios_documentos()
    test_formato_iramuteq_preserva_variaveis()
    test_decode_windows()
    test_deteccao_de_idioma()
    test_split_chunks_respeita_limite()
    test_traducao_mock_com_cache()
    test_mock_nao_contamina_cache_do_modelo_real()
    test_sem_expiracao_funciona()
    test_lista_de_modelos_ollama_novo_e_antigo()
    test_remove_raciocinio_e_json_invalido()
    print("\n=== TRADUÇÃO E CORREÇÕES OK ===")
