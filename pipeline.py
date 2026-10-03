"""
Pipeline sem interface: corpus → tradução → CHD (rainette) → interpretação LLM
→ arquivos de saída (relatório, JSON, CSV, corpus em português, config usada).
Usado pela linha de comando (cli.py).
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from corpus import Document, decode_bytes, parse_corpus, to_iramuteq
from llm_interpreter import DEFAULT_HOST, ClassInterpreter
from translator import LANGUAGE_NAMES, CorpusTranslator, language_summary


@dataclass
class Config:
    # Classificação (rainette)
    k: int = 5
    segment_size: int = 40
    min_segment_size: int = 12
    min_split_members: int = 10
    min_docfreq: int = 3
    n_terms: int = 20
    seed: int = 42
    language: str = "pt"          # stopwords quando não há tradução
    # Tradução
    translate: bool = True
    source_lang: str = "auto"
    # LLM
    use_llm: bool = True
    model: str = "qwen2.5:7b"
    host: str = DEFAULT_HOST
    temperature: float = 0.3
    deep: bool = False
    mock: bool = False
    # Cache
    use_cache: bool = True
    force_refresh: bool = False
    cache_db: str = ".cache/interpretations.db"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Parâmetros desconhecidos na configuração: {', '.join(unknown)}")
        return cls(**data)

    def with_overrides(self, **overrides: Any) -> "Config":
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return Config(**data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False)


@dataclass
class PipelineResult:
    config: Config
    source_name: str
    documents: List[Document]
    texts: List[str]                       # textos analisados (traduzidos, se for o caso)
    translations: list = field(default_factory=list)
    classes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    sizes: Dict[int, int] = field(default_factory=dict)
    interpretations: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    warnings: List[str] = field(default_factory=list)


Log = Callable[[str], None]


def default_bridge_factory():
    from rainette_bridge import RainetteBridge

    return RainetteBridge()


def load_documents(path: str | Path) -> List[Document]:
    return parse_corpus(decode_bytes(Path(path).read_bytes()))


def run_pipeline(
    documents: List[Document],
    config: Config,
    source_name: str = "corpus",
    log: Log = print,
    bridge_factory: Callable[[], Any] = default_bridge_factory,
) -> PipelineResult:
    if not documents:
        raise ValueError("Corpus vazio.")
    result = PipelineResult(
        config=config,
        source_name=source_name,
        documents=documents,
        texts=[d.text for d in documents],
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    log(f"{len(documents)} documento(s) em {source_name}")

    if config.translate:
        translator = CorpusTranslator(
            model=config.model, host=config.host, db_path=config.cache_db,
            use_cache=config.use_cache, mock=config.mock,
        )
        result.translations = translator.translate_corpus(
            result.texts, source_lang=config.source_lang,
            force_refresh=config.force_refresh,
            progress=lambda i, n: log(f"  tradução {i}/{n}") if i == n or i % 10 == 0 else None,
        )
        result.texts = [r.text for r in result.translations]
        resumo = ", ".join(
            f"{LANGUAGE_NAMES.get(k, k)}: {v}"
            for k, v in language_summary(result.translations).items()
        )
        log(f"Idiomas: {resumo}")

    log(f"CHD (rainette), k={config.k}…")
    bridge = bridge_factory()
    result.classes = bridge.run(
        texts=result.texts,
        k=config.k,
        segment_size=config.segment_size,
        min_segment_size=config.min_segment_size,
        min_split_members=config.min_split_members,
        language="pt" if config.translate else config.language,
        min_docfreq=config.min_docfreq,
        n_terms=config.n_terms,
        seed=config.seed,
    )
    result.sizes = bridge.get_group_sizes()
    empty = [cid for cid, data in result.classes.items() if not data.get("segments")]
    if empty:
        result.warnings.append(
            f"Classes sem segmentos: {empty}. O rainette pode ter gerado menos de k classes."
        )

    if config.use_llm:
        interpreter = ClassInterpreter(
            model=config.model, host=config.host, temperature=config.temperature,
            db_path=config.cache_db, use_cache=config.use_cache, mock=config.mock,
        )
        for cid, data in result.classes.items():
            if not data.get("segments"):
                continue
            log(f"  interpretando classe {cid}…")
            result.interpretations[cid] = interpreter.interpretar(
                class_id=cid, forms=data.get("forms", []),
                segments=data.get("segments", []), deep=config.deep,
                force_refresh=config.force_refresh,
            )

    result.finished_at = datetime.now().isoformat(timespec="seconds")
    return result


# ---------------------------------------------------------------------------
# Saídas
# ---------------------------------------------------------------------------

def _class_name(result: PipelineResult, cid: int) -> str:
    return result.interpretations.get(cid, {}).get("nome") or f"Classe {cid}"


def build_report(result: PipelineResult) -> str:
    cfg = result.config
    total = sum(result.sizes.values()) or 1
    lines = [
        f"# Relatório Textome — {result.source_name}",
        "",
        f"Gerado em {result.finished_at}.",
        "",
        "## Método (para citar no trabalho)",
        "",
        f"- Corpus: {len(result.documents)} documento(s).",
    ]
    if cfg.translate and result.translations:
        n = sum(r.translated for r in result.translations)
        idiomas = ", ".join(
            f"{LANGUAGE_NAMES.get(k, k)} ({v})"
            for k, v in language_summary(result.translations).items()
        )
        lines.append(
            f"- Tradução automática para o português de {n} documento(s) com o modelo "
            f"local `{cfg.model}` (temperatura 0). Idiomas de origem: {idiomas}."
        )
    lines += [
        f"- Classificação Hierárquica Descendente (método Reinert) com o pacote R rainette: "
        f"k = {cfg.k}, segmentos de ~{cfg.segment_size} palavras, mínimo de "
        f"{cfg.min_segment_size} formas por segmento, frequência mínima {cfg.min_docfreq}, "
        f"semente {cfg.seed}.",
    ]
    if cfg.use_llm:
        lines.append(
            f"- Nomes e resumos das classes sugeridos pelo modelo `{cfg.model}` "
            f"(temperatura {cfg.temperature}) e revisados pelo pesquisador."
            + (" **Atenção: modo simulado (mock).**" if cfg.mock else "")
        )
    lines += [f"- Software: Textome, Python {platform.python_version()}.", ""]

    if result.warnings:
        lines += ["## Avisos", ""] + [f"- {w}" for w in result.warnings] + [""]

    lines += ["## Classes", ""]
    for cid, data in result.classes.items():
        n_seg = result.sizes.get(cid, 0)
        interp = result.interpretations.get(cid, {})
        lines += [
            f"### Classe {cid} — {_class_name(result, cid)}",
            "",
            f"{n_seg} segmento(s) ({100 * n_seg / total:.1f}% do corpus classificado).",
            "",
        ]
        if interp.get("descricao"):
            lines += [f"*{interp['descricao']}*", ""]
        if interp.get("resumo"):
            lines += [interp["resumo"], ""]
        if interp.get("interpretacao"):
            lines += ["**Interpretação aprofundada**", "", interp["interpretacao"], ""]
        forms = data.get("forms", [])
        if forms:
            lines += ["| Forma | χ² |", "|---|---:|"]
            lines += [f"| {f} | {chi2:.1f} |" for f, chi2 in forms]
            lines.append("")
        segs = data.get("segments", [])
        if segs:
            lines += ["**Segmentos**", ""] + [f"> {s}\n" for s in segs]
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(result: PipelineResult, out_dir: str | Path) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def write(name: str, content: str) -> None:
        path = out / name
        path.write_text(content, encoding="utf-8")
        written.append(path)

    write("relatorio.md", build_report(result))
    write("config_usada.yaml", result.config.to_yaml())

    payload = {
        "fonte": result.source_name,
        "inicio": result.started_at,
        "fim": result.finished_at,
        "avisos": result.warnings,
        "classes": [
            {
                "id": cid,
                "nome": _class_name(result, cid),
                "segmentos_total": result.sizes.get(cid, 0),
                "formas": [{"forma": f, "chi2": c} for f, c in data.get("forms", [])],
                "segmentos": data.get("segments", []),
                "interpretacao": result.interpretations.get(cid, {}),
            }
            for cid, data in result.classes.items()
        ],
    }
    write("classes.json", json.dumps(payload, ensure_ascii=False, indent=2))

    csv_path = out / "formas.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["classe", "nome", "forma", "chi2"])
        for cid, data in result.classes.items():
            for f, c in data.get("forms", []):
                writer.writerow([cid, _class_name(result, cid), f, f"{c:.3f}"])
    written.append(csv_path)

    if result.translations:
        write(
            "corpus_pt.txt",
            to_iramuteq(
                [r.text for r in result.translations],
                headers=[d.header for d in result.documents],
                extra_vars=[f"*lang_{r.source_lang}" for r in result.translations],
            ),
        )
    return written
