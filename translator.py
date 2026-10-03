"""
Detecção de idioma e tradução de corpus para português (pt-BR) com LLM local.
Traduções ficam em cache no mesmo SQLite das interpretações (sem expiração).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from llm_interpreter import DEFAULT_HOST, make_client, strip_thinking

LANGUAGE_NAMES: Dict[str, str] = {
    "pt": "português",
    "en": "inglês",
    "es": "espanhol",
    "fr": "francês",
    "de": "alemão",
    "it": "italiano",
    "outro": "outro idioma",
    "und": "indeterminado",
}

# Palavras funcionais frequentes e pouco ambíguas entre si.
_STOPWORDS: Dict[str, set] = {
    "pt": set("não é uma com os das dos ao também são pelo pela você isso muito "
              "foi tem nos então ou seu sua já há quando em um do da no na mas "
              "ainda onde pelos pelas aos às num numa essa esse está estão o e que de se por para como mais".split()),
    "en": set("the and of to is in that it for was with are this be on have not "
              "they by from you which an or at but has were their".split()),
    "es": set("el los las y del es una por con para pero más también muy sus fue "
              "hay este cuando ya en un al lo se está están su que de la como mejor puede no "
              "tienen tiene son sin entre nosotros ellos usted esto eso".split()),
    "fr": set("le les des et est une pas dans que pour qui sur avec ce il sont au "
              "du nous ne vous un la aux ces cette leur que de".split()),
    "de": set("der die und das ist nicht ein eine zu den mit von sich auch auf für "
              "im dem wir ich werden wird sind er sie wurde hat haben bin letztes".split()),
    "it": set("il di che è per non sono gli del della con anche più nel ma ci "
              "questo come degli un una le nella alla molto mia mio si ho ha hanno sul nelle "
              "dei delle miei".split()),
}

# Terminações/letras características (peso extra por palavra).
_HINTS: Dict[str, re.Pattern] = {
    "pt": re.compile(r"(ção|ções|ão|ões|ã|õ|nh|lh|ência$|ância$|dade$)"),
    "es": re.compile(r"(ción|ciones|ñ|¿|¡|encia$|dad$|ie|(?<!q)ue)"),
    "fr": re.compile(r"(eau|aux$|è|oi|ou$|ë)"),
    "de": re.compile(r"(ß|ä|ö|ü|sch|ung$|keit$|tz|cht)"),
    "it": re.compile(r"(zione|zioni|gli|zz|ò|ì|iamo$|cch|ggi|tt)"),
    "en": re.compile(r"(th|ing$|ly$)"),
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    """
    Retorna "pt", "en", "es", "fr", "de", "it", "outro" (ex.: escrita não latina
    ou idioma não listado) ou "und" (texto curto demais).
    Usa langdetect se estiver instalado; senão, contagem de palavras funcionais.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if letters and sum(ord(c) < 0x250 for c in letters) / len(letters) < 0.5:
        return "outro"

    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    if len(words) < 3:
        return "und"

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        code = detect(text)
        return code if code in _STOPWORDS else "outro"
    except Exception:
        pass

    scores = {
        lang: 2 * sum(w in sw for w in words) + sum(bool(_HINTS[lang].search(w)) for w in words)
        for lang, sw in _STOPWORDS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "outro" if len(words) >= 8 else "und"
    return best


def split_chunks(text: str, max_chars: int = 1800) -> List[str]:
    """Divide em blocos por parágrafo/frase para não estourar o contexto do modelo."""
    chunks: List[str] = []
    current = ""
    for piece in re.split(r"(?<=[.!?…])\s+|\n+", text.strip()):
        if not piece:
            continue
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


@dataclass
class TranslationResult:
    original: str
    text: str            # texto final (traduzido ou original, se já era pt)
    source_lang: str
    translated: bool
    cached: bool = False


class CorpusTranslator:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = DEFAULT_HOST,
        db_path: str | Path = ".cache/interpretations.db",
        use_cache: bool = True,
        mock: bool = False,
        target_lang: str = "pt",
    ):
        self.model = model
        self.client = make_client(host, mock=mock)
        self.model_key = f"mock::{model}" if mock else model
        self.target_lang = target_lang
        self.use_cache = use_cache
        self.db_path = Path(db_path)
        if use_cache:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS translations (
                        cache_key   TEXT PRIMARY KEY,
                        source_lang TEXT,
                        model_used  TEXT,
                        translation TEXT NOT NULL,
                        created_at  TEXT NOT NULL
                    )
                    """
                )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _key(self, text: str, source_lang: str) -> str:
        raw = f"{self.model_key}|{source_lang}|{self.target_lang}|{text.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _prompt(self, text: str, source_lang: str) -> str:
        origem = LANGUAGE_NAMES.get(source_lang, "idioma de origem")
        if source_lang in ("outro", "und"):
            origem = "idioma de origem (identifique-o)"
        return f"""Traduza o texto abaixo do {origem} para o português do Brasil.

Regras:
- Tradução fiel e completa: não resuma, não comente, não acrescente nada.
- Preserve nomes próprios, siglas, números e citações.
- Mantenha o registro do autor (formal, coloquial, oral).
- Responda APENAS com a tradução, sem as marcações de início e fim.

<texto>
{text}
</texto>"""

    def _call(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError(
                "Biblioteca 'ollama' não está instalada ou o cliente não foi inicializado."
            )
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 4096},
        )
        out = strip_thinking(response["message"]["content"])
        return re.sub(r"</?texto>", "", out).strip()

    def translate(
        self, text: str, source_lang: str = "auto", force_refresh: bool = False
    ) -> TranslationResult:
        lang = detect_language(text) if source_lang == "auto" else source_lang
        if lang == self.target_lang or not text.strip():
            return TranslationResult(text, text, lang, translated=False)

        key = self._key(text, lang)
        if self.use_cache and not force_refresh:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT translation FROM translations WHERE cache_key = ?", (key,)
                ).fetchone()
            if row:
                return TranslationResult(text, row[0], lang, translated=True, cached=True)

        parts = [self._call(self._prompt(chunk, lang)) for chunk in split_chunks(text)]
        translation = "\n".join(p for p in parts if p)

        if self.use_cache:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?, ?)",
                    (key, lang, self.model_key, translation,
                     datetime.now().isoformat(timespec="seconds")),
                )
        return TranslationResult(text, translation, lang, translated=True)

    def translate_corpus(
        self,
        texts: List[str],
        source_lang: str = "auto",
        force_refresh: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[TranslationResult]:
        results = []
        for i, text in enumerate(texts, 1):
            results.append(self.translate(text, source_lang, force_refresh))
            if progress:
                progress(i, len(texts))
        return results


def language_summary(results: List[TranslationResult]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for r in results:
        summary[r.source_lang] = summary.get(r.source_lang, 0) + 1
    return summary
