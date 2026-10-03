"""
Textome — protótipo
Interface Streamlit moderna com Classificação Hierárquica Descendente (rainette)
e interpretação automática via LLM local (Ollama).
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Textome",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from corpus import Document, load_documents_from_upload, to_iramuteq
from limpeza import clean_documents
from translator import LANGUAGE_NAMES

# ---------------------------------------------------------------------------
# Sidebar — parâmetros
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Parâmetros")

    st.markdown("### Corpus")
    uploaded = st.file_uploader(
        "Arquivo de texto (.txt)",
        type=["txt"],
        help="Formato livre ou formato IRaMuTeQ (****). Qualquer idioma.",
    )

    demo = st.checkbox("Usar corpus de demonstração", value=not bool(uploaded))

    st.markdown("### Classificação (rainette)")
    k = st.slider("Número de classes (k)", min_value=2, max_value=12, value=5)
    segment_size = st.slider("Tamanho preferido do segmento (palavras)", 20, 80, 40)
    min_segment_size = st.slider("Mín. formas por segmento", 5, 30, 12)
    min_docfreq = st.slider("Frequência mínima do termo (docfreq)", 2, 20, 3)

    st.markdown("### Idioma e tradução")
    translate_pt = st.checkbox(
        "Traduzir textos para português",
        value=True,
        help="Detecta o idioma de cada texto e traduz com o LLM local o que não "
             "estiver em português. A análise é feita sobre o texto traduzido.",
    )
    source_lang = st.selectbox(
        "Idioma de origem",
        options=["auto", "en", "es", "fr", "de", "it", "outro"],
        format_func=lambda c: "Detectar automaticamente" if c == "auto" else LANGUAGE_NAMES[c],
        disabled=not translate_pt,
    )
    if translate_pt:
        language = "pt"
    else:
        language = st.selectbox(
            "Idioma do corpus (stopwords)",
            options=["pt", "en", "fr", "es", "de", "it"],
            format_func=lambda c: LANGUAGE_NAMES[c],
            index=0,
        )

    st.markdown("### Limpeza do corpus")
    clean_corpus = st.checkbox(
        "Limpar segundo as regras do IRaMuTeQ",
        value=True,
        help="Remove aspas, emojis, URLs e caracteres especiais; troca hífens por _ "
             "(guarda_chuva) e separa pronomes (disse me); corrige cabeçalhos ****.",
    )
    compound_raw = st.text_area(
        "Expressões compostas (uma por linha)",
        placeholder="sistema único de saúde\nbem estar",
        help="Viram uma palavra só: sistema_único_de_saúde.",
        disabled=not clean_corpus,
    )
    compound_terms = [t.strip() for t in compound_raw.splitlines() if t.strip()]

    st.markdown("### LLM local (Ollama)")
    model = st.text_input("Modelo Ollama", value="qwen2.5:7b")
    use_llm = st.checkbox("Interpretar classes com LLM", value=True)
    use_mock = st.checkbox(
        "Modo mock (sem Ollama)",
        value=False,
        help="Usa respostas simuladas para testar o fluxo sem ollama serve.",
    )
    temperature = st.slider("Temperatura do LLM", 0.0, 1.0, 0.3, 0.05)
    deep_interp = st.checkbox("Interpretação aprofundada", value=False)
    use_cache = st.checkbox("Usar cache SQLite", value=True)
    force_refresh = st.checkbox("Forçar nova geração (ignorar cache)", value=False)

    st.markdown("### Cache")
    if st.button("🧹 Limpar expiradas"):
        from llm_interpreter import LLMInterpreter
        n = LLMInterpreter(use_cache=True, auto_purge_on_init=False).purge_expired()
        st.success(f"{n} entradas expiradas removidas")
    if st.button("🗑️ Limpar todo o cache"):
        from llm_interpreter import LLMInterpreter
        n = LLMInterpreter(use_cache=True, auto_purge_on_init=False).clear_cache()
        st.success(f"{n} entradas removidas")

    run_btn = st.button("▶ Executar análise", type="primary", use_container_width=True)
    translate_btn = st.button(
        "🌐 Só traduzir o corpus",
        use_container_width=True,
        disabled=not translate_pt,
        help="Traduz e oferece o corpus em português (formato IRaMuTeQ) para download, sem rodar a CHD.",
    )


# ---------------------------------------------------------------------------
# Corpus de demonstração
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    "A saúde pública enfrenta sérios problemas de financiamento e falta de profissionais qualificados nos hospitais.",
    "Os hospitais da rede pública sofrem com a escassez de leitos, medicamentos e equipamentos básicos.",
    "O atendimento no SUS é frequentemente demorado e as filas de espera aumentam a cada ano.",
    "É urgente investir em infraestrutura hospitalar e na capacitação contínua dos médicos e enfermeiros.",
    "A educação de qualidade depende de professores bem formados, valorizados e com condições dignas de trabalho.",
    "As escolas públicas precisam de melhores instalações, materiais didáticos atualizados e merenda adequada.",
    "O ensino fundamental ainda registra altas taxas de evasão e repetência em várias regiões do país.",
    "A valorização do magistério e a formação continuada são essenciais para melhorar a educação nacional.",
    "A violência urbana cresce nas periferias e exige políticas de segurança integradas com assistência social.",
    "O policiamento ostensivo sozinho não resolve; é preciso prevenir com educação e oportunidade de emprego.",
    "As famílias mais pobres sofrem com a falta de acesso a serviços básicos e a exclusão digital.",
    "Programas de transferência de renda ajudam, mas precisam ser acompanhados de capacitação profissional.",
    "O meio ambiente está sob pressão: desmatamento, poluição dos rios e mudança climática afetam todos.",
    "A preservação das florestas e o uso sustentável dos recursos naturais são temas centrais da agenda pública.",
    "A juventude enfrenta desemprego e falta de perspectiva, o que alimenta ciclos de pobreza e violência.",
    "Investir em cultura, esporte e lazer nas periferias é uma forma de inclusão e prevenção social.",
] * 3


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title("📊 Textome")
st.caption(
    "Classificação Hierárquica Descendente (método Reinert) + interpretação automática com LLM local"
)

with st.expander("Sobre este protótipo", expanded=False):
    st.markdown(
        """
        Este aplicativo reimplementa a lógica central do IRaMuTeQ usando:

        - **rainette** (R) — Classificação Hierárquica Descendente
        - **quanteda** — pré-processamento textual
        - **Ollama** — interpretação semântica das classes (nomes e resumos)
        - **Streamlit** — interface moderna

        Fluxo: upload do corpus → segmentação → CHD → formas características →
        interpretação por LLM → visualização.
        """
    )


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def load_corpus() -> list[Document]:
    if demo:
        docs = [Document(t) for t in DEMO_TEXTS]
        st.info(f"Corpus de demonstração carregado ({len(docs)} textos).")
        return docs
    if uploaded is None:
        st.warning("Envie um arquivo ou ative o corpus de demonstração.")
        st.stop()
    try:
        uploaded.seek(0)
        docs = load_documents_from_upload(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()
    if not docs:
        st.error("O arquivo está vazio.")
        st.stop()
    st.success(f"Arquivo carregado: {len(docs)} documento(s).")
    return docs


def translate_docs(docs: list[Document], progress) -> list[str]:
    """Traduz para português o que não estiver em pt; guarda o resultado na sessão."""
    from translator import CorpusTranslator, language_summary

    translator = CorpusTranslator(model=model, use_cache=use_cache, mock=use_mock)

    def on_progress(i: int, n: int) -> None:
        progress.progress(int(10 * i / n), text=f"Traduzindo texto {i}/{n}…")

    try:
        results = translator.translate_corpus(
            [d.text for d in docs],
            source_lang=source_lang,
            force_refresh=force_refresh,
            progress=on_progress,
        )
    except Exception as e:
        progress.empty()
        st.error(f"Falha na tradução (o Ollama está rodando com o modelo {model}?): {e}")
        st.stop()

    st.session_state["translation"] = {
        "results": results,
        "summary": language_summary(results),
        "headers": [d.header for d in docs],
    }
    return [r.text for r in results]


def render_translation_panel() -> None:
    data = st.session_state.get("translation")
    if not data:
        return
    results = data["results"]
    n_translated = sum(r.translated for r in results)
    resumo = ", ".join(
        f"{LANGUAGE_NAMES.get(lang, lang)}: {qty}" for lang, qty in data["summary"].items()
    )
    st.subheader("🌐 Tradução para português")
    st.caption(f"{n_translated} de {len(results)} texto(s) traduzido(s) · Idiomas detectados — {resumo}")

    corpus_pt = to_iramuteq(
        [r.text for r in results],
        headers=data["headers"],
        extra_vars=[f"*lang_{r.source_lang}" for r in results],
    )
    st.download_button(
        "⬇️ Baixar corpus em português (formato IRaMuTeQ)",
        data=corpus_pt.encode("utf-8"),
        file_name="corpus_pt.txt",
        mime="text/plain",
    )
    with st.expander("Ver original × tradução", expanded=False):
        for i, r in enumerate(results, 1):
            if not r.translated:
                continue
            st.markdown(f"**Texto {i}** · {LANGUAGE_NAMES.get(r.source_lang, r.source_lang)}"
                        + (" · cache" if r.cached else ""))
            c1, c2 = st.columns(2)
            c1.caption("Original")
            c1.write(r.original)
            c2.caption("Português")
            c2.write(r.text)
    st.divider()



def render_final_corpus_panel() -> None:
    corpus_final = st.session_state.get("final_corpus")
    if not corpus_final:
        return
    with st.expander("🧹 Corpus final (como foi analisado)", expanded=False):
        cleaning = st.session_state.get("cleaning")
        if cleaning:
            st.markdown("**Limpeza aplicada:**\n" + "\n".join(f"- {c}" for c in cleaning))
        elif cleaning is not None:
            st.caption("Nenhuma alteração de limpeza foi necessária.")
        st.download_button(
            "⬇️ Baixar corpus final (formato IRaMuTeQ)",
            data=corpus_final.encode("utf-8"),
            file_name="corpus_final.txt",
            mime="text/plain",
        )
        st.code(corpus_final[:3000] + ("\n…" if len(corpus_final) > 3000 else ""), language=None)


if translate_btn:
    docs = load_corpus()
    progress = st.progress(0, text="Traduzindo…")
    translate_docs(docs, progress)
    progress.empty()

if run_btn:
    docs = load_corpus()
    progress = st.progress(0, text="Preparando análise…")
    status = st.empty()

    # --- Tradução ---
    if translate_pt:
        texts = translate_docs(docs, progress)
        langs = [r.source_lang for r in st.session_state["translation"]["results"]]
        headers = [
            f"{d.header or f'**** *doc_{i:03d}'} *lang_{lang}"
            for i, (d, lang) in enumerate(zip(docs, langs), 1)
        ]
    else:
        texts = [d.text for d in docs]
        headers = [d.header for d in docs]
        st.session_state.pop("translation", None)

    # --- Limpeza (regras do IRaMuTeQ) ---
    final_docs = [Document(t, h) for t, h in zip(texts, headers)]
    if clean_corpus:
        final_docs, report = clean_documents(final_docs, compound_terms)
        st.session_state["cleaning"] = report.summary_lines()
        if not final_docs:
            st.error("Nenhum texto restou após a limpeza.")
            st.stop()
    else:
        st.session_state.pop("cleaning", None)
    texts = [d.text for d in final_docs]
    st.session_state["final_corpus"] = to_iramuteq(texts, [d.header for d in final_docs])

    # --- Classificação ---

    try:
        status.info("Importando rainette_bridge…")
        from rainette_bridge import RainetteBridge

        progress.progress(15, text="Executando CHD (rainette)…")
        status.info("Rodando Classificação Hierárquica Descendente. Isso pode levar alguns minutos…")

        bridge = RainetteBridge()
        classes = bridge.run(
            texts=texts,
            k=k,
            segment_size=segment_size,
            min_segment_size=min_segment_size,
            language=language,
            min_docfreq=min_docfreq,
            n_terms=20,
        )
        sizes = bridge.get_group_sizes()

        progress.progress(55, text="CHD concluída. Extraindo resultados…")
        st.session_state["classes"] = classes
        st.session_state["sizes"] = sizes
        st.session_state["k"] = k

    except ImportError as e:
        progress.empty()
        status.empty()
        st.error("Dependência ausente")
        st.code(str(e))
        st.markdown(
            """
            **Checklist de instalação:**
            1. R instalado (`sudo apt install r-base r-base-dev`)
            2. No R: `install.packages(c("quanteda", "rainette"))`
            3. Python: `pip install -r requirements-r.txt` (ou use o Docker)
            """
        )
        render_translation_panel()
        render_final_corpus_panel()
        st.stop()
    except Exception as e:
        progress.empty()
        status.empty()
        st.error("Falha na classificação")
        st.exception(e)
        render_translation_panel()
        render_final_corpus_panel()
        st.stop()

    # --- Interpretação LLM ---
    interpretations = {}
    if use_llm:
        try:
            progress.progress(65, text="Interpretando classes com LLM…")
            from llm_interpreter import ClassInterpreter

            if use_mock:
                status.info("Modo mock ativo (sem Ollama real).")
            else:
                status.info(f"Chamando Ollama ({model})…")

            interpreter = ClassInterpreter(
                model=model,
                temperature=temperature,
                use_cache=use_cache,
                auto_purge_on_init=True,
                mock=use_mock,
            )
            n = len(classes)
            cache_hits = 0
            for i, (cid, data) in enumerate(classes.items(), 1):
                progress.progress(
                    65 + int(30 * i / max(n, 1)),
                    text=f"Interpretando classe {i}/{n}…",
                )
                forms_raw = data.get("forms", [])
                segments = data.get("segments", [])
                result = interpreter.interpretar(
                    class_id=cid,
                    forms=forms_raw,
                    segments=segments,
                    deep=deep_interp,
                    force_refresh=force_refresh,
                )
                if result.get("cached"):
                    cache_hits += 1
                interpretations[cid] = result
            st.session_state["interpretations"] = interpretations
            if cache_hits:
                status.info(f"Cache: {cache_hits}/{n} classe(s) reutilizadas.")
        except Exception as e:
            st.warning(f"LLM indisponível ou falhou: {e}")
            st.session_state["interpretations"] = {}
            interpretations = {}
    else:
        st.session_state["interpretations"] = {}

    progress.progress(100, text="Concluído")
    status.success("Análise finalizada.")
    progress.empty()


# ---------------------------------------------------------------------------
# Exibição dos resultados (session_state)
# ---------------------------------------------------------------------------

render_translation_panel()
render_final_corpus_panel()

classes = st.session_state.get("classes")
interpretations = st.session_state.get("interpretations", {})
sizes = st.session_state.get("sizes", {})

if not classes:
    st.markdown(
        """
        ### Como começar
        1. (Opcional) Envie um arquivo `.txt` na barra lateral — formato livre ou IRaMuTeQ (`****`),
           em qualquer idioma (textos estrangeiros são traduzidos para o português).
        2. Ajuste `k` e demais parâmetros.
        3. Ative o LLM se o Ollama estiver rodando com o modelo desejado.
        4. Clique em **Executar análise**.

        Sem arquivo, use o **corpus de demonstração**.
        """
    )
    st.stop()

st.header("Resultados da classificação")

# Métricas rápidas
cols = st.columns(min(len(classes), 6))
for i, (cid, data) in enumerate(classes.items()):
    with cols[i % len(cols)]:
        label = interpretations.get(cid, {}).get("nome", f"Classe {cid}")
        n_seg = sizes.get(cid, len(data.get("segments", [])))
        st.metric(label=label[:40], value=f"{n_seg} seg.")

st.divider()

# Cards por classe
from llm_interpreter import render_class_card

tabs = st.tabs([interpretations.get(cid, {}).get("nome", f"Classe {cid}") for cid in classes])

for tab, (cid, data) in zip(tabs, classes.items()):
    with tab:
        interp = interpretations.get(cid, {})
        render_class_card(
            class_id=cid,
            forms=data.get("forms", []),
            segments=data.get("segments", []),
            nome=interp.get("nome", f"Classe {cid}"),
            descricao=interp.get("descricao", ""),
            resumo=interp.get("resumo", ""),
        )

# Comparação opcional
if use_llm and len(classes) >= 2 and interpretations:
    st.divider()
    st.subheader("Comparar duas classes")
    ids = list(classes.keys())
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        id_a = st.selectbox("Classe A", ids, index=0, key="cmp_a")
    with c2:
        id_b = st.selectbox("Classe B", ids, index=min(1, len(ids) - 1), key="cmp_b")
    with c3:
        do_cmp = st.button("Comparar")

    if do_cmp and id_a != id_b:
        try:
            from llm_interpreter import ClassInterpreter
            interpreter = ClassInterpreter(
                model=model,
                temperature=temperature,
                use_cache=use_cache,
                mock=use_mock,
            )
            texto = interpreter.comparar(
                id_a,
                classes[id_a]["forms"],
                classes[id_a]["segments"],
                id_b,
                classes[id_b]["forms"],
                classes[id_b]["segments"],
            )
            st.write(texto)
        except Exception as e:
            st.error(f"Erro na comparação: {e}")

st.divider()
st.caption("Textome — protótipo experimental · rainette + Ollama + Streamlit")
