"""
Transcrição de entrevistas (áudio/vídeo) com Whisper local (faster-whisper)
e montagem do corpus IRaMuTeQ com as variáveis de cada participante.

  pip install faster-whisper
  Na primeira execução o modelo é baixado da Hugging Face (tiny ~75 MB,
  small ~460 MB, medium ~1,5 GB, large-v3 ~3 GB). Depois funciona offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from corpus import Document
from limpeza import build_header, read_table

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac",
                    ".aac", ".wma", ".mp4", ".webm", ".mkv", ".mov"}


@dataclass
class Transcript:
    source: Path
    text: str
    language: str
    duration_s: float = 0.0


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> Transcript: ...


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        language: Optional[str] = None,   # None = detectar; "pt" força português
        device: str = "auto",
        compute_type: str = "int8",
        beam_size: int = 5,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "Transcrição requer o pacote faster-whisper: pip install faster-whisper"
            ) from e
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, path: Path) -> Transcript:
        segments, info = self.model.transcribe(
            str(path), language=self.language, beam_size=self.beam_size, vad_filter=True
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return Transcript(path, text, info.language, float(info.duration or 0.0))


def find_audio(paths: Sequence[str | Path]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"Não encontrado: {raw}")
    if not found:
        raise FileNotFoundError("Nenhum arquivo de áudio/vídeo encontrado.")
    return found


def load_variables(table_path: str | Path, file_column: str = "arquivo") -> Dict[str, Dict[str, str]]:
    """
    Planilha (CSV/XLSX) com uma linha por entrevista: coluna 'arquivo' com o nome
    do áudio (com ou sem extensão) e as demais colunas viram variáveis IRaMuTeQ.
    """
    rows = read_table(table_path)
    if rows and file_column not in rows[0]:
        raise ValueError(f"A planilha precisa de uma coluna '{file_column}'. Colunas: {', '.join(rows[0])}")
    table: Dict[str, Dict[str, str]] = {}
    for row in rows:
        name = Path(str(row.get(file_column, "")).strip()).stem.lower()
        if name:
            table[name] = {k: v for k, v in row.items() if k != file_column}
    return table


def transcribe_to_documents(
    audio_files: Sequence[Path],
    transcriber: Transcriber,
    variables: Optional[Dict[str, Dict[str, str]]] = None,
    progress: Optional[Callable[[int, int, Path], None]] = None,
) -> tuple[List[Document], List[Transcript], List[str]]:
    """Retorna (documentos, transcrições, avisos)."""
    variables = variables or {}
    docs: List[Document] = []
    transcripts: List[Transcript] = []
    warnings: List[str] = []
    for i, path in enumerate(audio_files, 1):
        if progress:
            progress(i, len(audio_files), path)
        tr = transcriber.transcribe(path)
        transcripts.append(tr)
        if not tr.text.strip():
            warnings.append(f"{path.name}: nenhuma fala reconhecida")
            continue
        own = variables.get(path.stem.lower())
        if variables and own is None:
            warnings.append(f"{path.name}: sem linha na planilha de variáveis")
        header_vars = {"ent": path.stem, **(own or {})}
        docs.append(Document(tr.text, build_header(header_vars)))
    return docs, transcripts, warnings
