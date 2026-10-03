"""
Testes unitários do cache SQLite + TTL (sem depender do Ollama).
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from llm_interpreter import (
    ClassInterpretation,
    InterpretationCacheSQLite,
    LLMInterpreter,
)


def test_make_key_stable():
    k1 = InterpretationCacheSQLite.make_key(
        ["Saúde", "público"], ["seg1"], "qwen3:8b", 0.3, False
    )
    k2 = InterpretationCacheSQLite.make_key(
        ["saúde", "público"], ["seg1"], "qwen3:8b", 0.3, False
    )
    k3 = InterpretationCacheSQLite.make_key(
        ["saúde", "público"], ["seg1"], "qwen3:8b", 0.3, True
    )
    assert k1 == k2, "chave deve ser case-insensitive nas forms"
    assert k1 != k3, "deep=True deve mudar a chave"
    print("OK test_make_key_stable")


def test_set_get_clear():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        cache = InterpretationCacheSQLite(db, default_ttl_hours=24, auto_purge_on_init=False)
        key = InterpretationCacheSQLite.make_key(
            ["a", "b"], ["s1"], "m1", 0.3, False
        )
        interp = ClassInterpretation(
            class_id=1,
            nome="Teste",
            descricao="Desc",
            resumo="Resumo",
            interpretacao="",
            model_used="m1",
        )
        cache.set(key, interp, deep=False, temperature=0.3, ttl_hours=24)
        got = cache.get(key)
        assert got is not None
        assert got.nome == "Teste"
        assert got.cached is True
        assert got.expires_at is not None
        n = cache.clear()
        assert n == 1
        assert cache.get(key) is None
    print("OK test_set_get_clear")


def test_ttl_expiry():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        cache = InterpretationCacheSQLite(db, default_ttl_hours=1, auto_purge_on_init=False)
        key = InterpretationCacheSQLite.make_key(
            ["x"], ["y"], "m", 0.2, False
        )
        interp = ClassInterpretation(
            class_id=2,
            nome="Expira",
            descricao="",
            resumo="",
            interpretacao="",
            model_used="m",
        )
        # grava com TTL de -1h forçando expires_at no passado via SQL
        cache.set(key, interp, ttl_hours=1)
        # força expiração manual
        past = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
        with cache._connect() as conn:
            conn.execute(
                "UPDATE interpretations SET expires_at = ? WHERE cache_key = ?",
                (past, key),
            )
            conn.commit()

        got = cache.get(key)
        assert got is None, "entrada expirada deve retornar None e ser removida"
        stats = cache.stats()
        assert stats["total"] == 0
    print("OK test_ttl_expiry")


def test_purge_expired():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        cache = InterpretationCacheSQLite(db, auto_purge_on_init=False)
        key_ok = InterpretationCacheSQLite.make_key(["ok"], ["s"], "m", 0.3, False)
        key_exp = InterpretationCacheSQLite.make_key(["exp"], ["s"], "m", 0.3, False)

        interp = ClassInterpretation(
            class_id=1, nome="OK", descricao="", resumo="", interpretacao="", model_used="m"
        )
        cache.set(key_ok, interp, ttl_hours=100)
        cache.set(key_exp, interp, ttl_hours=1)

        past = (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds")
        with cache._connect() as conn:
            conn.execute(
                "UPDATE interpretations SET expires_at = ? WHERE cache_key = ?",
                (past, key_exp),
            )
            conn.commit()

        removed = cache.purge_expired()
        assert removed == 1
        assert cache.get(key_ok) is not None
        assert cache.get(key_exp) is None
        stats = cache.stats()
        assert stats["active"] == 1
        assert stats["expired"] == 0
    print("OK test_purge_expired")


def test_no_ttl_permanent():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        cache = InterpretationCacheSQLite(db, default_ttl_hours=None, auto_purge_on_init=False)
        key = InterpretationCacheSQLite.make_key(["p"], ["s"], "m", 0.3, False)
        interp = ClassInterpretation(
            class_id=9, nome="Perm", descricao="", resumo="", interpretacao="", model_used="m"
        )
        cache.set(key, interp, ttl_hours=None)
        got = cache.get(key)
        assert got is not None
        assert got.expires_at is None
        stats = cache.stats()
        assert stats["no_ttl"] == 1
    print("OK test_no_ttl_permanent")


def test_extract_json():
    interp = LLMInterpreter(use_cache=False)
    # sem ollama: só testamos _extract_json
    d1 = interp._extract_json('{"nome": "Classe A", "descricao": "Tema X"}')
    assert d1["nome"] == "Classe A"
    d2 = interp._extract_json('texto antes {"nome": "B", "descricao": "Y"} texto depois')
    assert d2["nome"] == "B"
    d3 = interp._extract_json("sem json nenhum")
    assert d3["nome"] == "Classe sem nome"
    print("OK test_extract_json")


def test_list_recent_and_stats():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        cache = InterpretationCacheSQLite(db, auto_purge_on_init=False)
        for i in range(3):
            key = InterpretationCacheSQLite.make_key([f"f{i}"], ["s"], "m", 0.3, False)
            interp = ClassInterpretation(
                class_id=i,
                nome=f"Nome{i}",
                descricao="",
                resumo="",
                interpretacao="",
                model_used="m",
            )
            cache.set(key, interp, ttl_hours=48)
        recent = cache.list_recent(10)
        assert len(recent) == 3
        stats = cache.stats()
        assert stats["total"] == 3
        assert stats["active"] == 3
    print("OK test_list_recent_and_stats")


def test_interpreter_cache_wrappers():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        # use_cache True mas sem chamar interpret_class (precisa ollama)
        li = LLMInterpreter(
            use_cache=True,
            db_path=db,
            auto_purge_on_init=True,
            default_ttl_hours=24,
        )
        assert li.cache_stats()["total"] == 0
        assert li.clear_cache() == 0
        assert li.purge_expired() == 0
        assert li.list_recent_cached() == []
    print("OK test_interpreter_cache_wrappers")


if __name__ == "__main__":
    test_make_key_stable()
    test_set_get_clear()
    test_ttl_expiry()
    test_purge_expired()
    test_no_ttl_permanent()
    test_extract_json()
    test_list_recent_and_stats()
    test_interpreter_cache_wrappers()
    print("\n=== TODOS OS TESTES PASSARAM ===")
