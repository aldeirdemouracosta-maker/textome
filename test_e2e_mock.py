"""
Teste ponta a ponta com mock do Ollama (sem rainette/R e sem Ollama real).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from llm_interpreter import (
    ClassInterpreter,
    enable_mock_ollama,
    InterpretationCacheSQLite,
)


def test_full_interpret_with_mock_and_cache():
    enable_mock_ollama()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "e2e.db"
        interp = ClassInterpreter(
            model="mock-model",
            temperature=0.3,
            use_cache=True,
            db_path=db,
            auto_purge_on_init=False,
            ttl_name_hours=24,
            ttl_deep_hours=12,
        )

        forms = [("saúde", 12.5), ("hospital", 9.1), ("fila", 7.0)]
        segments = [
            "os pacientes enfrentam longas filas de espera nos hospitais",
            "a qualidade do atendimento médico tem sido questionada",
        ]

        r1 = interp.interpretar(3, forms, segments, deep=False)
        assert r1["nome"]
        assert r1["resumo"]
        assert r1["cached"] is False or r1["cached"] is False

        r2 = interp.interpretar(3, forms, segments, deep=False)
        assert r2["cached"] is True, "segunda chamada deve vir do cache"
        assert r2["nome"] == r1["nome"]

        r3 = interp.interpretar(3, forms, segments, deep=False, force_refresh=True)
        assert r3["cached"] is False, "force_refresh deve regenerar"

        r4 = interp.interpretar(3, forms, segments, deep=True)
        assert r4["interpretacao"], "deep deve preencher interpretação"
        assert r4["cached"] is False

        r5 = interp.interpretar(3, forms, segments, deep=True)
        assert r5["cached"] is True

        cmp = interp.comparar(1, forms, segments, 2, forms, segments)
        assert isinstance(cmp, str) and len(cmp) > 20

        stats = interp.cache_stats()
        assert stats["active"] >= 2
        print("stats:", stats)
    print("OK test_full_interpret_with_mock_and_cache")


def test_enable_mock_before_client():
    enable_mock_ollama()
    interp = ClassInterpreter(use_cache=False, model="x")
    assert interp.client is not None
    out = interp._call("json nome teste")
    assert "nome" in out or "{" in out
    print("OK test_enable_mock_before_client")


if __name__ == "__main__":
    test_enable_mock_before_client()
    test_full_interpret_with_mock_and_cache()
    print("\n=== E2E MOCK OK ===")
