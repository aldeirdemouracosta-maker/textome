"""
Demo Streamlit – Interpretação de Classes com LLM Local + Cache SQLite/TTL
Textome / Textome
"""

import streamlit as st
from llm_interpreter import LLMInterpreter, list_available_models

st.set_page_config(
    page_title="Textome – Interpretação LLM",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Interpretação de Classes com LLM Local")
st.caption("Módulo para Textome / Textome • Ollama + Cache SQLite com TTL")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Configuração")

    available = list_available_models()
    if available:
        model = st.selectbox("Modelo Ollama", available, index=0)
    else:
        model = st.text_input("Modelo Ollama", value="qwen3:8b")
        st.warning("Ollama não detectado em localhost:11434")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    deep = st.checkbox("Interpretação aprofundada", value=False)

    st.divider()
    st.subheader("Cache (SQLite + TTL)")

    use_cache = st.checkbox("Usar cache", value=True)
    force_refresh = st.checkbox("Forçar nova geração", value=False)
    auto_purge = st.checkbox("Limpar expiradas ao iniciar", value=True)

    ttl_name_opt = st.selectbox(
        "TTL – Nome + descrição",
        options=[
            ("30 dias", 720),
            ("7 dias", 168),
            ("24 horas", 24),
            ("Sem expiração", None),
        ],
        format_func=lambda x: x[0],
        index=0,
    )
    ttl_deep_opt = st.selectbox(
        "TTL – Interpretação profunda",
        options=[
            ("7 dias", 168),
            ("3 dias", 72),
            ("24 horas", 24),
            ("Sem expiração", None),
        ],
        format_func=lambda x: x[0],
        index=0,
    )

    stats_interpreter = LLMInterpreter(
        model=model,
        temperature=temperature,
        use_cache=True,
        default_ttl_hours=168,
        ttl_name_hours=ttl_name_opt[1],
        ttl_deep_hours=ttl_deep_opt[1],
        auto_purge_on_init=auto_purge,
        db_path=".cache/interpretations.db",
    )
    stats = stats_interpreter.cache_stats()

    st.caption(f"Ativas: **{stats['active']}** | Expiradas: **{stats['expired']}**")
    st.caption(f"Sem TTL: **{stats['no_ttl']}** | Total: **{stats['total']}**")
    if stats.get("by_model"):
        for mname, qty in stats["by_model"].items():
            st.caption(f"• {mname}: {qty}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Limpar expiradas", use_container_width=True):
            removed = stats_interpreter.purge_expired()
            st.success(f"{removed} removidas")
            st.rerun()
    with col_b:
        if st.button("🗑️ Limpar tudo", use_container_width=True):
            removed = stats_interpreter.clear_cache()
            st.success(f"{removed} removidas")
            st.rerun()

    with st.expander("Últimas interpretações em cache"):
        recent = stats_interpreter.list_recent_cached(10)
        if recent:
            for item in recent:
                exp = item.get("expires_at") or "sem expiração"
                st.markdown(
                    f"**Classe {item['class_id']}** – {item['nome']}  \n"
                    f"`{item['model_used']}` • {item['created_at']}  \n"
                    f"Expira: `{exp}`"
                )
        else:
            st.caption("Nenhum item em cache ainda.")

# ------------------------------------------------------------------
# Área principal – dados da classe
# ------------------------------------------------------------------
st.subheader("Dados da Classe")

col1, col2 = st.columns(2)

with col1:
    class_id = st.number_input("ID da Classe", min_value=1, value=3)
    forms_input = st.text_area(
        "Formas características (vírgula ou linha)",
        value="saúde, público, sistema, atendimento, hospital, médico, paciente, fila, espera, qualidade",
        height=120,
    )

with col2:
    segments_input = st.text_area(
        "Segmentos representativos (UCE) – um por linha",
        value=(
            "o sistema de saúde público não consegue atender a demanda\n"
            "os pacientes enfrentam longas filas de espera nos hospitais\n"
            "a qualidade do atendimento médico tem sido questionada\n"
            "falta de profissionais e infraestrutura precária\n"
            "o acesso à saúde continua sendo um problema grave"
        ),
        height=180,
    )

forms = [f.strip() for f in forms_input.replace("\n", ",").split(",") if f.strip()]
segments = [s.strip() for s in segments_input.split("\n") if s.strip()]

if st.button("🚀 Interpretar Classe", type="primary", use_container_width=True):
    if not forms or not segments:
        st.error("Informe pelo menos algumas formas e segmentos.")
    else:
        with st.spinner(f"Interpretando com **{model}**..."):
            try:
                interpreter = LLMInterpreter(
                    model=model,
                    temperature=temperature,
                    use_cache=use_cache,
                    default_ttl_hours=168,
                    ttl_name_hours=ttl_name_opt[1],
                    ttl_deep_hours=ttl_deep_opt[1],
                    auto_purge_on_init=False,  # já feito na sidebar
                    db_path=".cache/interpretations.db",
                )
                result = interpreter.interpret_class(
                    class_id=class_id,
                    forms=forms,
                    segments=segments,
                    deep=deep,
                    force_refresh=force_refresh,
                )

                if result.cached:
                    st.info("⚡ Resultado carregado do cache (LLM não foi chamado)")
                else:
                    st.success("✨ Nova interpretação gerada e salva no cache")

                st.markdown(f"### 🏷️ {result.nome}")
                st.markdown(f"**Descrição:** {result.descricao}")

                st.markdown("#### Resumo")
                st.write(result.resumo)

                if result.interpretacao:
                    st.markdown("#### Interpretação Aprofundada")
                    st.write(result.interpretacao)

                with st.expander("Dados brutos (JSON)"):
                    st.json(result.to_dict())

            except Exception as e:
                st.error(f"Erro ao chamar o modelo: {e}")
                st.info(
                    "Verifique se o Ollama está rodando (`ollama serve`) "
                    "e se o modelo está baixado (`ollama pull qwen3:8b`)."
                )
