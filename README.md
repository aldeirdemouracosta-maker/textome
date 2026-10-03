# Textome (ex-IRaMuTeQ 2.0)

Interface moderna para **Classificação Hierárquica Descendante** (método Reinert) com interpretação automática por **LLM local** (Ollama) e **cache SQLite com TTL**.

## O que faz

1. Lê corpus em texto livre ou formato IRaMuTeQ (`****`)
2. Segmenta e classifica com **rainette** (R) via `rainette_bridge.py`
3. Extrai formas características (χ²) e segmentos por classe
4. Nomeia e resume cada classe com Ollama (`llm_interpreter.py`)
5. Cacheia interpretações em SQLite (TTL configurável, limpeza automática)
6. Interface **Streamlit** (`app.py`)

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| `app.py` | App completo (corpus → CHD → LLM) |
| `app_demo.py` | Demo só do módulo de interpretação |
| `llm_interpreter.py` | LLM + cache SQLite + TTL + mock |
| `rainette_bridge.py` | Ponte Python ↔ rainette (R) |
| `test_cache.py` | Testes unitários do cache |
| `test_e2e_mock.py` | Teste ponta a ponta com mock Ollama |

## Pré-requisitos

### R
```bash
sudo apt install r-base r-base-dev
```
No R:
```r
install.packages(c("quanteda", "rainette"))
```

### Ollama (opcional se usar modo mock)
```bash
ollama pull qwen2.5:7b
# ou: ollama pull llama3.1:8b
```

### Python
```bash
cd Textome
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# para CHD completo:
pip install rpy2
```

## Executar

```bash
# App completo
streamlit run app.py

# Só interpretação (sem R)
streamlit run app_demo.py
```

Na sidebar do `app.py`:
- **Modo mock** — testa o fluxo sem Ollama
- **Usar cache SQLite** / **Forçar nova geração**
- **Limpar expiradas** / **Limpar todo o cache**

## Testes (sem Ollama e sem R)

```bash
python test_cache.py
python test_e2e_mock.py
```

## Cache

- Banco: `.cache/interpretations.db`
- Chave: hash de forms + segments + model + temperature + deep
- TTL padrão: 30 dias (nome/resumo) e 7 dias (interpretação profunda)
- Auto-purge de expirados na inicialização do interpretador
