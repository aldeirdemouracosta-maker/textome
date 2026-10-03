"""
Textome — protótipo
Interface Streamlit moderna com Classificação Hierárquica Descendente (rainette)
e interpretação automática via LLM local (Ollama).
"""

from __future__ import annotations

import io
from typing import List

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

# ---------------------------------------------------------------------------
# Utilitários de leitura de corpus
# ---------------------------------------------------------------------------

def parse_iramuteq_format(raw: str) -> List[str]:
    """
    Lê texto no formato IRaMuTeQ / Alceste.
    Separadores típicos: **** ou **** *var_1 *var_2
    """
    lines = raw.splitlines()
    documents: List[str] = []
    current: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("****"):
            if current:
                documents.append("\n".join(current).strip())
                current = []
            continue
        if stripped:
            current.append(stripped)

    if current:
        documents.append("\n".join(current).strip())

    # Se não havia separadores, trata o arquivo inteiro como um único texto
    # e depois será segmentado pelo rainette
    if not documents and raw.strip():
        documents = [raw.strip()]

    return [d for d in documents if d]


def load_texts_from_upload(uploaded) -> List[str]:
    content = uploaded.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1", errors="replace")

    name = (uploaded.name or "").lower()
    if name.endswith(".txt") or "iramuteq" in name or "****" in text[:2000]:
        docs = parse_iramuteq_format(text)
        if len(docs) >= 1:
            return docs

    # Fallback: uma linha = um documento, ou bloco único
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 5:
        return lines
    return [text.strip()] if text.strip() else []


# ---------------------------------------------------------------------------
# Sidebar — parâmetros
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Parâmetros")

    st.markdown("### Corpus")
    uploaded = st.file_uploader(
        "Arquivo de texto (.txt)",
        type=["txt"],
        help="Formato livre ou formato IRaMuTeQ (****).",
    )

    demo = st.checkbox("Usar corpus de demonstração", value=not bool(uploaded))

    st.markdown("### Classificação (rainette)")
    k = st.slider("Número de classes (k)", min_value=2, max_value=12, value=5)
    segment_size = st.slider("Tamanho preferido do segmento (palavras)", 20, 80, 40)
    min_segment_size = st.slider("Mín. formas por segmento", 5, 30, 12)
    min_docfreq = st.slider("Frequência mínima do termo (docfreq)", 2, 20, 3)
    language = st.selectbox(
        "Idioma (stopwords)",
        options=["pt", "en", "fr", "es", "de", "it"],
        index=0,
    )

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

if run_btn:
    # --- Carregar textos ---
    if demo:
        texts = DEMO_TEXTS
        st.info(f"Corpus de demonstração carregado ({len(texts)} textos).")
    elif uploaded is not None:
        try:
            texts = load_texts_from_upload(uploaded)
            st.success(f"Arquivo carregado: {len(texts)} documento(s)/bloco(s).")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            st.stop()
    else:
        st.warning("Envie um arquivo ou ative o corpus de demonstração.")
        st.stop()

    if len(texts) < 2:
        st.error("É necessário pelo menos 2 textos/documentos para a classificação.")
        st.stop()

    # --- Classificação ---
    progress = st.progress(0, text="Preparando análise…")
    status = st.empty()

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
            3. Python: `pip install rpy2`
            """
        )
        st.stop()
    except Exception as e:
        progress.empty()
        status.empty()
        st.error("Falha na classificação")
        st.exception(e)
        st.stop()

    # --- Interpretação LLM ---
    interpretations = {}
    if use_llm:
        try:
            progress.progress(65, text="Interpretando classes com LLM…")
            from llm_interpreter import ClassInterpreter, enable_mock_ollama

            if use_mock:
                enable_mock_ollama()
                status.info("Modo mock ativo (sem Ollama real).")
            else:
                status.info(f"Chamando Ollama ({model})…")

            interpreter = ClassInterpreter(
                model=model,
                temperature=temperature,
                use_cache=use_cache,
                auto_purge_on_init=True,
            )
            n = len(classes)
            cache_hits = 0
            for i, (cid, data) in enumerate(classes.items(), 1):
                progress.progress(
                    65 + int(30 * i / max(n, 1)),
                    text=f"Interpretando classe {cid}/{k}…",
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

classes = st.session_state.get("classes")
interpretations = st.session_state.get("interpretations", {})
sizes = st.session_state.get("sizes", {})

if not classes:
    st.markdown(
        """
        ### Como começar
        1. (Opcional) Envie um arquivo `.txt` na barra lateral — formato livre ou IRaMuTeQ (`****`).
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
            from llm_interpreter import ClassInterpreter, enable_mock_ollama
            if use_mock:
                enable_mock_ollama()
            interpreter = ClassInterpreter(
                model=model,
                temperature=temperature,
                use_cache=use_cache,
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
