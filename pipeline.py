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
from limpeza import CleaningReport, clean_documents
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
    extra_stopwords: List[str] = field(default_factory=list)  # palavras a ignorar na CHD
    # Limpeza (regras do IRaMuTeQ)
    clean: bool = True
    compound_terms: List[str] = field(default_factory=list)  # ex.: "sistema único de saúde"
    # Análises complementares (AFC, similitude, nuvem, dendrograma) e relatório Word
    analyses: bool = True
    # Triangulação com BERTopic (opcional; requer requirements-topicos.txt)
    topics: bool = False
    topic_mode: str = "kmeans"     # "kmeans" = mesmo k da CHD; "auto" = HDBSCAN
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
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
        for key in ("compound_terms", "extra_stopwords"):
            if key in data and data[key] is None:
                data = {**data, key: []}
        if "compound_terms" not in data:
            data = {**data, "compound_terms": []}
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
    texts: List[str]                       # textos analisados (traduzidos e limpos)
    final_documents: List[Document] = field(default_factory=list)  # corpus como analisado
    cleaning: Optional[CleaningReport] = None
    translations: list = field(default_factory=list)
    classes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    sizes: Dict[int, int] = field(default_factory=dict)
    interpretations: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    warnings: List[str] = field(default_factory=list)
    matrix: Any = None                     # analises.CorpusMatrix
    all_segments: List[Any] = field(default_factory=list)  # (texto, classe) de todos os segmentos
    triangulation: Any = None              # triangulacao.Triangulation
    dendrogram_png: Optional[bytes] = None
    analysis: Any = None                   # analises.AnalysisOutput (preenchido em write_outputs)


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

    headers = [d.header for d in documents]
    if result.translations:
        headers = [
            f"{h or f'**** *doc_{i:03d}'} *lang_{r.source_lang}"
            for i, (h, r) in enumerate(zip(headers, result.translations), 1)
        ]
    final = [Document(t, h) for t, h in zip(result.texts, headers)]
    if config.clean:
        final, result.cleaning = clean_documents(final, config.compound_terms)
        n = sum(result.cleaning.changes.values())
        log(f"Limpeza: {n} alteração(ões), {len(result.cleaning.header_problems)} problema(s) de cabeçalho")
        result.warnings += result.cleaning.header_problems
        if result.cleaning.empty_documents:
            result.warnings.append(
                f"Documentos vazios após a limpeza (ignorados): {result.cleaning.empty_documents}"
            )
    result.final_documents = final
    result.texts = [d.text for d in final]
    if not result.texts:
        raise ValueError("Nenhum texto restou após a limpeza.")

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
        **({"extra_stopwords": config.extra_stopwords} if config.extra_stopwords else {}),
    )
    result.sizes = bridge.get_group_sizes()
    if hasattr(bridge, "get_segments"):
        try:
            result.all_segments = list(bridge.get_segments())
        except Exception as e:
            result.warnings.append(f"Segmentos para validação indisponíveis: {e}")
    if config.analyses and hasattr(bridge, "get_matrix"):
        try:
            result.matrix = bridge.get_matrix()
        except Exception as e:  # análises são complementares: não derrubam a CHD
            result.warnings.append(f"Matriz para AFC/similitude indisponível: {e}")
        if hasattr(bridge, "save_dendrogram"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                png = Path(tmp) / "dendrograma.png"
                err = bridge.save_dendrogram(str(png))
                if err is None and png.exists():
                    result.dendrogram_png = png.read_bytes()
                else:
                    result.warnings.append(f"Dendrograma não gerado: {err}")
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

    if config.topics:
        if not result.all_segments:
            result.warnings.append("Triangulação não feita: segmentos da CHD indisponíveis.")
        else:
            import triangulacao

            log("Triangulação com BERTopic…")
            try:
                texts = [t for t, _ in result.all_segments]
                n_classes = len([c for c in result.sizes if c > 0])
                topics, words = triangulacao.run_bertopic(
                    texts, n_topics=n_classes if config.topic_mode == "kmeans" else None,
                    seed=config.seed, embedding_model=config.embedding_model,
                )
                result.triangulation = triangulacao.compare(
                    [c for _, c in result.all_segments], topics, topic_words=words,
                    method="BERTopic (KMeans, k igual à CHD)" if config.topic_mode == "kmeans"
                    else "BERTopic (HDBSCAN)",
                )
                log(f"  {result.triangulation.summary()}")
            except Exception as e:
                result.warnings.append(f"Triangulação com BERTopic falhou: {type(e).__name__}: {e}")

    result.finished_at = datetime.now().isoformat(timespec="seconds")
    return result


# ---------------------------------------------------------------------------
# Saídas
# ---------------------------------------------------------------------------

def _class_name(result: PipelineResult, cid: int) -> str:
    return result.interpretations.get(cid, {}).get("nome") or f"Classe {cid}"


def method_items(result: PipelineResult) -> List[str]:
    """Itens da seção de método (markdown leve: `código` e **negrito**)."""
    cfg = result.config
    items = [f"Corpus: {len(result.documents)} documento(s)."]
    if cfg.translate and result.translations:
        n = sum(r.translated for r in result.translations)
        idiomas = ", ".join(
            f"{LANGUAGE_NAMES.get(k, k)} ({v})"
            for k, v in language_summary(result.translations).items()
        )
        items.append(
            f"Tradução automática para o português de {n} documento(s) com o modelo "
            f"local `{cfg.model}` (temperatura 0). Idiomas de origem: {idiomas}."
        )
    if cfg.clean and result.cleaning is not None:
        detalhes = ", ".join(f"{r} ({n})" for r, n in result.cleaning.changes.most_common())
        items.append(
            "Corpus preparado segundo as regras do IRaMuTeQ (remoção de aspas e "
            "caracteres especiais, hífens e expressões compostas unidas por \"_\")"
            + (f": {detalhes}." if detalhes else "; nenhuma alteração necessária.")
        )
        if cfg.compound_terms:
            items.append(f"Expressões compostas definidas: {', '.join(cfg.compound_terms)}.")
    items.append(
        f"Classificação Hierárquica Descendente (método Reinert) com o pacote R rainette: "
        f"k = {cfg.k}, segmentos de ~{cfg.segment_size} palavras, mínimo de "
        f"{cfg.min_segment_size} formas por segmento, frequência mínima {cfg.min_docfreq}, "
        f"semente {cfg.seed}."
    )
    if result.analysis is not None:
        items.append(
            "Análises complementares: análise fatorial de correspondência (AFC) da tabela "
            "classes × formas, análise de similitude (árvore máxima do grafo de coocorrência "
            "das 50 formas mais frequentes) e nuvem de palavras."
        )
    if result.triangulation is not None:
        items.append(
            f"Triangulação com {result.triangulation.method}, a partir de embeddings semânticos "
            f"(`{cfg.embedding_model}`), comparada às classes da CHD pelo índice de Rand ajustado "
            "(ARI) e pela informação mútua normalizada (NMI)."
        )
    if cfg.use_llm:
        items.append(
            f"Nomes e resumos das classes sugeridos pelo modelo `{cfg.model}` "
            f"(temperatura {cfg.temperature}) e revisados pelo pesquisador."
            + (" **Atenção: modo simulado (mock).**" if cfg.mock else "")
        )
    items.append(f"Software: Textome, Python {platform.python_version()}.")
    return items


def build_report(result: PipelineResult) -> str:
    total = sum(result.sizes.values()) or 1
    lines = [
        f"# Relatório Textome — {result.source_name}",
        "",
        f"Gerado em {result.finished_at}.",
        "",
        "## Método (para citar no trabalho)",
        "",
    ] + [f"- {item}" for item in method_items(result)] + [""]

    if result.warnings:
        lines += ["## Avisos", ""] + [f"- {w}" for w in result.warnings] + [""]

    if result.analysis is not None:
        from analises import FIGURE_TITLES

        st = result.analysis.stats
        lines += [
            "## Estatísticas textuais",
            "",
            f"- Segmentos de texto: {st.segments}; classificados: {st.classified} "
            f"({st.classified_pct:.1f}%)",
            f"- Ocorrências: {st.occurrences}; formas distintas: {st.forms}; "
            f"hápax: {st.hapax} ({st.hapax_pct_forms:.1f}% das formas)",
            "",
            "## Figuras",
            "",
        ]
        for key, path in result.analysis.figures.items():
            lines += [f"![{FIGURE_TITLES.get(key, key)}](figuras/{path.name})", ""]

    if result.triangulation is not None:
        tri = result.triangulation
        lines += ["## Triangulação CHD × BERTopic", "", tri.summary(), "",
                  "| Classe | Tópico predominante | % da classe | Palavras do tópico |", "|---|---|---:|---|"]
        for cid in tri.class_ids:
            b = tri.best_topic[cid]
            lines.append(f"| {cid} — {_class_name(result, cid)} | T{b['topico']} | {b['pct_classe']:.0f}% | "
                         f"{', '.join(tri.topic_words.get(b['topico'], []))} |")
        lines.append("")

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

    if result.matrix is not None and result.config.analyses:
        from analises import run_all

        names = {cid: _class_name(result, cid) for cid in result.classes}
        names = {cid: n for cid, n in names.items() if n != f"Classe {cid}"}
        try:
            result.analysis = run_all(
                result.matrix, out / "figuras", result.sizes, names,
                class_forms={cid: d.get("forms", []) for cid, d in result.classes.items()},
                dendrogram_png=result.dendrogram_png, seed=result.config.seed,
            )
            result.warnings += result.analysis.warnings
            written += list(result.analysis.figures.values())
        except Exception as e:
            result.warnings.append(f"Análises complementares falharam: {e}")

    if result.triangulation is not None:
        from triangulacao import plot_crosstab

        write("triangulacao.json", json.dumps(result.triangulation.to_dict(), ensure_ascii=False, indent=2))
        try:
            (out / "figuras").mkdir(exist_ok=True)
            fig = plot_crosstab(result.triangulation,
                                {cid: _class_name(result, cid) for cid in result.classes},
                                out / "figuras" / "triangulacao.png")
            written.append(fig)
            if result.analysis is not None:
                result.analysis.figures["triangulacao"] = fig
        except Exception as e:
            result.warnings.append(f"Figura da triangulação não gerada: {e}")

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
        "todos_segmentos": [{"texto": t, "classe": int(c)} for t, c in result.all_segments],
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

    try:
        from relatorio_docx import build_docx

        written.append(build_docx(result, out / "relatorio.docx"))
    except ImportError:
        result.warnings.append("relatorio.docx não gerado: instale python-docx")

    # Corpus exatamente como foi analisado (traduzido e limpo), pronto para o IRaMuTeQ.
    write(
        "corpus_final.txt",
        to_iramuteq([d.text for d in result.final_documents],
                    headers=[d.header for d in result.final_documents]),
    )
    return written
