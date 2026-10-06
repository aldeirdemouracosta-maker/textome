"""
Leitura e exportação de corpus (texto livre ou formato IRaMuTeQ / Alceste).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_HEADER_RE = re.compile(r"^\s*\*{4}")

# Complementos às listas Snowball (quanteda), que não trazem formas muito frequentes,
# sobretudo na fala transcrita. Removidas da CHD e do BERTopic.
EXTRA_STOPWORDS = {
    "pt": ["é", "pra", "pro", "pras", "pros", "né", "tá", "tô", "aí", "daí", "nas", "nos",
           "às", "ao", "aos", "num", "numa", "dum", "duma", "vc", "q"],
}


@dataclass
class Document:
    text: str
    header: str = ""  # linha "**** *var_1 *var_2" original, se houver


def decode_bytes(content: bytes) -> str:
    """Decodifica arquivos vindos do Windows/Linux sem perder acentos."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")


def has_iramuteq_headers(raw: str) -> bool:
    return any(_HEADER_RE.match(line) for line in raw.splitlines())


def parse_corpus(raw: str) -> List[Document]:
    """
    Formato IRaMuTeQ: cada linha "****" abre um documento (variáveis preservadas).
    Texto livre: parágrafos separados por linha em branco; sem parágrafos,
    uma linha por documento quando há mais de 5 linhas; senão, um bloco único.
    """
    if has_iramuteq_headers(raw):
        docs: List[Document] = []
        header = ""
        current: List[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if _HEADER_RE.match(stripped):
                if current:
                    docs.append(Document("\n".join(current), header))
                header, current = stripped, []
            elif stripped:
                current.append(stripped)
        if current:
            docs.append(Document("\n".join(current), header))
        return docs

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) >= 2:
        return [Document(p) for p in paragraphs]
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 5:
        return [Document(ln) for ln in lines]
    return [Document(raw.strip())] if raw.strip() else []


def parse_iramuteq_format(raw: str) -> List[str]:
    """Compatibilidade: devolve só os textos."""
    return [d.text for d in parse_corpus(raw)]


def load_documents_from_upload(uploaded) -> List[Document]:
    return parse_corpus(decode_bytes(uploaded.read()))


def to_iramuteq(texts: List[str], headers: List[str] | None = None,
                extra_vars: List[str] | None = None) -> str:
    """
    Gera corpus no formato IRaMuTeQ. Documentos sem cabeçalho recebem
    "**** *doc_001". extra_vars acrescenta variáveis (ex.: "*lang_en").
    """
    headers = headers or [""] * len(texts)
    extra_vars = extra_vars or [""] * len(texts)
    blocks = []
    for i, (text, header, extra) in enumerate(zip(texts, headers, extra_vars), 1):
        head = header or f"**** *doc_{i:03d}"
        if extra:
            head = f"{head} {extra}"
        blocks.append(f"{head}\n{text.strip()}")
    return "\n\n".join(blocks) + "\n"
