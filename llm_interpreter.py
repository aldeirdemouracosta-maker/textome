"""
Módulo de Interpretação de Classes com LLM Local (Ollama)
+ Cache SQLite com TTL e limpeza automática
Para Textome (análise textual + LLM local)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

# Endereço do Ollama: variável OLLAMA_HOST (ex.: no Docker) ou localhost.
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Sentinela: "usar o TTL padrão". None significa "sem expiração".
USE_DEFAULT_TTL: Any = object()


@dataclass
class ClassInterpretation:
    class_id: int
    nome: str
    descricao: str
    resumo: str
    interpretacao: str
    model_used: str
    cached: bool = False
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InterpretationCacheSQLite:
    """
    Cache SQLite com suporte a TTL (Time-To-Live).
    Entradas expiradas são ignoradas na leitura e podem ser limpas automaticamente.
    """

    def __init__(
        self,
        db_path: str | Path = ".cache/interpretations.db",
        default_ttl_hours: Optional[int] = 168,
        auto_purge_on_init: bool = True,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl_hours = default_ttl_hours
        self._init_db()

        if auto_purge_on_init:
            removed = self.purge_expired()
            if removed > 0:
                print(f"[cache] {removed} interpretações expiradas removidas automaticamente")

    @contextmanager
    def _connect(self):
        """Abre conexão, faz commit/rollback e sempre fecha (evita .db travado)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interpretations (
                    cache_key     TEXT PRIMARY KEY,
                    class_id      INTEGER,
                    nome          TEXT NOT NULL,
                    descricao     TEXT,
                    resumo        TEXT,
                    interpretacao TEXT,
                    model_used    TEXT,
                    raw_response  TEXT,
                    created_at    TEXT NOT NULL,
                    expires_at    TEXT,
                    deep          INTEGER NOT NULL DEFAULT 0,
                    temperature   REAL
                )
                """
            )
            try:
                conn.execute("SELECT expires_at FROM interpretations LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE interpretations ADD COLUMN expires_at TEXT")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON interpretations(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires_at ON interpretations(expires_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model ON interpretations(model_used)"
            )

    @staticmethod
    def make_key(
        forms: List[str],
        segments: List[str],
        model: str,
        temperature: float,
        deep: bool,
    ) -> str:
        payload = {
            "forms": [f.strip().lower() for f in forms],
            "segments": [s.strip() for s in segments],
            "model": model,
            "temperature": round(temperature, 3),
            "deep": deep,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_expired(self, expires_at: Optional[str]) -> bool:
        if not expires_at:
            return False
        try:
            return datetime.now() > datetime.fromisoformat(expires_at)
        except Exception:
            return True

    def get(self, key: str) -> Optional[ClassInterpretation]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM interpretations WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        if self._is_expired(row["expires_at"]):
            self.delete(key)
            return None

        return ClassInterpretation(
            class_id=row["class_id"],
            nome=row["nome"],
            descricao=row["descricao"] or "",
            resumo=row["resumo"] or "",
            interpretacao=row["interpretacao"] or "",
            model_used=row["model_used"],
            cached=True,
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            raw_response=row["raw_response"],
        )

    def set(
        self,
        key: str,
        interpretation: ClassInterpretation,
        deep: bool = False,
        temperature: float = 0.3,
        ttl_hours: Optional[int] = USE_DEFAULT_TTL,
    ) -> None:
        """ttl_hours: omitido → TTL padrão; None ou 0 → sem expiração."""
        now = datetime.now()
        created_at = now.isoformat(timespec="seconds")

        ttl = self.default_ttl_hours if ttl_hours is USE_DEFAULT_TTL else ttl_hours
        if ttl is not None and ttl > 0:
            expires_at = (now + timedelta(hours=ttl)).isoformat(timespec="seconds")
        else:
            expires_at = None

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO interpretations
                (cache_key, class_id, nome, descricao, resumo, interpretacao,
                 model_used, raw_response, created_at, expires_at, deep, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    interpretation.class_id,
                    interpretation.nome,
                    interpretation.descricao,
                    interpretation.resumo,
                    interpretation.interpretacao,
                    interpretation.model_used,
                    interpretation.raw_response,
                    created_at,
                    expires_at,
                    int(deep),
                    temperature,
                ),
            )

    def delete(self, key: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM interpretations WHERE cache_key = ?",
                (key,),
            )
            return cur.rowcount > 0

    def clear(self) -> int:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0]
            conn.execute("DELETE FROM interpretations")
        return count

    def purge_expired(self) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM interpretations
                WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                (now,),
            )
            return cur.rowcount

    def stats(self) -> Dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0]
            expired = conn.execute(
                """
                SELECT COUNT(*) FROM interpretations
                WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                (now,),
            ).fetchone()[0]
            no_ttl = conn.execute(
                """
                SELECT COUNT(*) FROM interpretations
                WHERE expires_at IS NULL
                """
            ).fetchone()[0]
            by_model = conn.execute(
                """
                SELECT model_used, COUNT(*) as qty
                FROM interpretations
                GROUP BY model_used
                """
            ).fetchall()

        return {
            "total": total,
            "expired": expired,
            "active": total - expired,
            "no_ttl": no_ttl,
            "by_model": {row["model_used"]: row["qty"] for row in by_model},
            "db_path": str(self.db_path.resolve()),
            "default_ttl_hours": self.default_ttl_hours,
        }

    def list_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT class_id, nome, model_used, created_at, expires_at, deep
                FROM interpretations
                WHERE expires_at IS NULL OR expires_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return [dict(row) for row in rows]


class LLMInterpreter:
    """
    Interpreta classes (método Reinert) com LLM local + cache SQLite com TTL.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        temperature: float = 0.3,
        host: str = DEFAULT_HOST,
        db_path: str | Path = ".cache/interpretations.db",
        use_cache: bool = True,
        default_ttl_hours: Optional[int] = 168,
        ttl_name_hours: Optional[int] = 720,
        ttl_deep_hours: Optional[int] = 168,
        auto_purge_on_init: bool = True,
        mock: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.host = host
        self.use_cache = use_cache
        self.default_ttl_hours = default_ttl_hours
        self.ttl_name_hours = ttl_name_hours
        self.ttl_deep_hours = ttl_deep_hours

        self.client = make_client(host, mock=mock)
        # Respostas simuladas nunca podem se misturar no cache com as do modelo real.
        self.is_mock = isinstance(self.client, MockOllamaClient)
        self.model_key = f"mock::{model}" if self.is_mock else model

        self.cache = (
            InterpretationCacheSQLite(
                db_path,
                default_ttl_hours=default_ttl_hours,
                auto_purge_on_init=auto_purge_on_init,
            )
            if use_cache
            else None
        )

    def _prompt_nome_descricao(self, forms: List[str], segments: List[str]) -> str:
        forms_txt = ", ".join(forms[:40])
        segs_txt = "\n\n---\n\n".join(segments[:5])
        return f"""Você é um especialista em análise de discurso e lexicométrica (método Reinert / IRaMuTeQ).

Abaixo estão as **formas características** de uma classe (ordenadas por relevância estatística) e alguns **segmentos de texto representativos** (UCE).

FORMAS CARACTERÍSTICAS:
{forms_txt}

SEGMENTOS REPRESENTATIVOS:
{segs_txt}

TAREFA:
1. Sugira um **nome curto e preciso** para esta classe (máximo 6 palavras).
2. Escreva uma **descrição** de 1 a 2 frases que capture o tema central.

Responda APENAS com um JSON válido no formato:
{{
  "nome": "Nome da Classe",
  "descricao": "Descrição clara e objetiva."
}}

Não adicione nenhum texto antes ou depois do JSON."""

    def _prompt_resumo(self, forms: List[str], segments: List[str], nome: str = "") -> str:
        forms_txt = ", ".join(forms[:30])
        segs_txt = "\n\n---\n\n".join(segments[:6])
        contexto = f"Nome sugerido da classe: {nome}\n\n" if nome else ""
        return f"""Você é um analista de discurso experiente.

{contexto}FORMAS CARACTERÍSTICAS:
{forms_txt}

SEGMENTOS REPRESENTATIVOS:
{segs_txt}

Escreva um **resumo interpretativo** da classe em 90 a 140 palavras.
Destaque o tema central, as nuances e o que essa classe revela sobre o corpus.
Use linguagem clara e acadêmica, em português.
Não invente informações que não estejam nos dados fornecidos."""

    def _prompt_interpretacao_profunda(
        self, forms: List[str], segments: List[str], nome: str = ""
    ) -> str:
        forms_txt = ", ".join(forms[:35])
        segs_txt = "\n\n---\n\n".join(segments[:7])
        return f"""Você é um pesquisador sênior em análise de discurso (abordagem lexicométrica / Reinert).

Classe: {nome or "sem nome"}

FORMAS CARACTERÍSTICAS:
{forms_txt}

SEGMENTOS REPRESENTATIVOS:
{segs_txt}

Faça uma **interpretação aprofundada** (150-220 palavras) respondendo:
- Qual é o núcleo temático desta classe?
- Quais sentidos e posicionamentos discursivos ela carrega?
- Que hipóteses de pesquisa ela sugere?

Seja rigoroso e baseie-se apenas nos dados apresentados."""

    def _call(self, prompt: str, json_mode: bool = False) -> str:
        if self.client is None:
            raise RuntimeError(
                "Biblioteca 'ollama' não está instalada ou o cliente não foi inicializado."
            )
        kwargs: Dict[str, Any] = {"format": "json"} if json_mode else {}
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": 1024,
            },
            **kwargs,
        )
        return strip_thinking(response["message"]["content"])

    def _extract_json(self, text: str) -> Dict[str, str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        fallback = {"nome": "Classe sem nome", "descricao": text[:200]}
        if not match:
            return fallback
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return fallback
        if not isinstance(data, dict):
            return fallback
        return {k: str(v) for k, v in data.items() if v is not None}

    def interpret_class(
        self,
        class_id: int,
        forms: List[str],
        segments: List[str],
        deep: bool = False,
        force_refresh: bool = False,
        ttl_hours: Optional[int] = USE_DEFAULT_TTL,
    ) -> ClassInterpretation:
        """ttl_hours: omitido → TTL do tipo (nome ou profunda); None → sem expiração."""
        key = InterpretationCacheSQLite.make_key(
            forms, segments, self.model_key, self.temperature, deep
        )
        if self.use_cache and self.cache and not force_refresh:
            cached = self.cache.get(key)
            if cached is not None:
                cached.class_id = class_id
                return cached

        raw_nome = self._call(self._prompt_nome_descricao(forms, segments), json_mode=True)
        parsed = self._extract_json(raw_nome)
        nome = parsed.get("nome", f"Classe {class_id}")
        descricao = parsed.get("descricao", "")

        resumo = self._call(self._prompt_resumo(forms, segments, nome))

        interpretacao = ""
        if deep:
            interpretacao = self._call(
                self._prompt_interpretacao_profunda(forms, segments, nome)
            )

        result = ClassInterpretation(
            class_id=class_id,
            nome=nome,
            descricao=descricao,
            resumo=resumo,
            interpretacao=interpretacao,
            model_used=self.model_key,
            cached=False,
            created_at=datetime.now().isoformat(timespec="seconds"),
            raw_response=raw_nome,
        )

        if self.use_cache and self.cache:
            if ttl_hours is not USE_DEFAULT_TTL:
                effective_ttl = ttl_hours
            elif deep:
                effective_ttl = self.ttl_deep_hours
            else:
                effective_ttl = self.ttl_name_hours

            self.cache.set(
                key,
                result,
                deep=deep,
                temperature=self.temperature,
                ttl_hours=effective_ttl,
            )

        return result

    def clear_cache(self) -> int:
        if self.cache:
            return self.cache.clear()
        return 0

    def purge_expired(self) -> int:
        if self.cache:
            return self.cache.purge_expired()
        return 0

    def cache_stats(self) -> Dict[str, Any]:
        if self.cache:
            return self.cache.stats()
        return {
            "total": 0,
            "expired": 0,
            "active": 0,
            "no_ttl": 0,
            "by_model": {},
            "db_path": None,
            "default_ttl_hours": None,
        }

    def list_recent_cached(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self.cache:
            return self.cache.list_recent(limit)
        return []


def strip_thinking(text: str) -> str:
    """Remove blocos <think>…</think> de modelos de raciocínio (ex.: qwen3)."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def make_client(host: str = DEFAULT_HOST, mock: bool = False):
    """Cliente Ollama real, ou o mock quando pedido explicitamente."""
    if mock:
        return MockOllamaClient()
    return ollama.Client(host=host) if ollama is not None else None


def _model_name(entry: Any) -> Optional[str]:
    # ollama>=0.4 usa objetos com o campo "model"; versões antigas, dicts com "name".
    for field in ("model", "name"):
        value = getattr(entry, field, None)
        if value is None and isinstance(entry, dict):
            value = entry.get(field)
        if value:
            return str(value)
    return None


def list_available_models(host: str = DEFAULT_HOST) -> List[str]:
    if ollama is None:
        return []
    try:
        client = ollama.Client(host=host)
        response = client.list()
        entries = getattr(response, "models", None)
        if entries is None and isinstance(response, dict):
            entries = response.get("models", [])
        return [name for name in map(_model_name, entries or []) if name]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Mock do Ollama (para testes sem serviço real)
# ---------------------------------------------------------------------------

class MockOllamaClient:
    """Simula respostas do Ollama para testes e demo offline."""

    def chat(
        self, model: str, messages: list, options: dict | None = None, **kwargs: Any
    ) -> dict:
        content = messages[-1]["content"] if messages else ""
        lower = content.lower()

        if "traduza o texto" in lower:
            match = re.search(r"<texto>\n?(.*?)\n?</texto>", content, re.DOTALL)
            original = match.group(1) if match else content
            text = f"[tradução simulada] {original}"
        elif "json" in lower and "nome" in lower:
            # Prompt de nome + descrição
            text = (
                '{"nome": "Saúde pública e acesso", '
                '"descricao": "Discursos sobre sistema de saúde, filas e qualidade do atendimento."}'
            )
        elif "resumo interpretativo" in lower or "90 a 140" in lower:
            text = (
                "Esta classe reúne enunciados centrados no acesso e na qualidade dos serviços "
                "de saúde. Predominam referências a filas, hospitais, profissionais e "
                "infraestrutura. O discurso aponta escassez de recursos e demanda reprimida, "
                "sugerindo um posicionamento crítico em relação à capacidade de resposta do "
                "sistema público. A presença recorrente de termos operacionais (atendimento, "
                "espera, leitos) indica uma abordagem concreta dos problemas, mais do que "
                "debates abstratos sobre políticas."
            )
        elif "interpretação aprofundada" in lower or "núcleo temático" in lower:
            text = (
                "O núcleo temático articula direito à saúde e experiência concreta do usuário. "
                "Os segmentos mobilizam um registro de denúncia e urgência, associando falhas "
                "operacionais a impactos sobre pacientes. Hipóteses de pesquisa incluem a "
                "relação entre território e tempo de espera, bem como a tensão entre discurso "
                "oficial de universalidade e práticas de exclusão. A classe pode contrastar com "
                "outras centradas em educação ou segurança, revelando eixos distintos de "
                "demanda por políticas públicas."
            )
        elif "compare" in lower or "compar" in lower:
            text = (
                "As duas classes diferem no objeto central: uma enfatiza saúde e acesso "
                "assistencial; a outra, quando contrastada, tende a deslocar o foco para "
                "outro domínio de políticas. A tensão discursiva aparece na priorização de "
                "recursos e na definição de urgências sociais."
            )
        else:
            text = "Resposta simulada do mock Ollama para fins de teste."

        return {"message": {"content": text}}


def enable_mock_ollama() -> None:
    """
    Substitui o módulo ollama pelo mock no processo inteiro (útil em testes).
    No app, prefira LLMInterpreter(mock=True), que não afeta outras instâncias.
    """
    global ollama

    class _FakeOllamaModule:
        Client = lambda host=DEFAULT_HOST: MockOllamaClient()

    ollama = _FakeOllamaModule  # type: ignore


# ---------------------------------------------------------------------------
# Compatibilidade com app.py legado (ClassInterpreter + render_class_card)
# ---------------------------------------------------------------------------

class ClassInterpreter(LLMInterpreter):
    """Alias compatível com o app.py original."""

    def interpretar(
        self,
        class_id: int,
        forms: List[str] | List[tuple],
        segments: List[str],
        deep: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, str]:
        form_labels = [
            f[0] if isinstance(f, (list, tuple)) else str(f) for f in forms
        ]
        result = self.interpret_class(
            class_id=class_id,
            forms=form_labels,
            segments=segments,
            deep=deep,
            force_refresh=force_refresh,
        )
        return {
            "nome": result.nome,
            "descricao": result.descricao,
            "resumo": result.resumo,
            "interpretacao": result.interpretacao,
            "cached": result.cached,
            "model_used": result.model_used,
        }

    def comparar(
        self,
        id_a: int,
        forms_a: List,
        segments_a: List[str],
        id_b: int,
        forms_b: List,
        segments_b: List[str],
    ) -> str:
        fa = [f[0] if isinstance(f, (list, tuple)) else str(f) for f in forms_a][:25]
        fb = [f[0] if isinstance(f, (list, tuple)) else str(f) for f in forms_b][:25]
        sa = "\n".join(segments_a[:4])
        sb = "\n".join(segments_b[:4])
        prompt = f"""Compare as Classes {id_a} e {id_b} (análise de discurso / Reinert).

Classe {id_a} — formas: {', '.join(fa)}
Segmentos:
{sa}

Classe {id_b} — formas: {', '.join(fb)}
Segmentos:
{sb}

Explique diferenças temáticas e possíveis tensões discursivas (80-150 palavras, em português)."""
        return self._call(prompt)


def render_class_card(
    class_id: int,
    forms: List,
    segments: List[str],
    nome: str = "",
    descricao: str = "",
    resumo: str = "",
) -> None:
    """Renderiza um card de classe no Streamlit (se disponível)."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(f"### Classe {class_id}: {nome or '(sem nome)'}")
    if descricao:
        st.markdown(f"*{descricao}*")
    if resumo:
        st.markdown("**Resumo**")
        st.write(resumo)

    form_labels = []
    for f in forms:
        if isinstance(f, (list, tuple)) and len(f) >= 2:
            form_labels.append(f"{f[0]} (χ²={f[1]:.1f})")
        elif isinstance(f, (list, tuple)):
            form_labels.append(str(f[0]))
        else:
            form_labels.append(str(f))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Formas características**")
        if form_labels:
            st.write(", ".join(form_labels[:30]))
        else:
            st.caption("Nenhuma forma disponível.")
    with c2:
        st.markdown("**Segmentos representativos**")
        if segments:
            for s in segments[:6]:
                st.markdown(f"> {s}")
        else:
            st.caption("Nenhum segmento disponível.")
