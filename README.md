# Textome (ex-IRaMuTeQ 2.0)

Interface moderna para **Classificação Hierárquica Descendante** (método Reinert) com interpretação automática por **LLM local** (Ollama) e **cache SQLite com TTL**.

## O que faz

1. Lê corpus em texto livre ou formato IRaMuTeQ (`****`), em **qualquer idioma**
2. Detecta o idioma de cada texto e **traduz para português** com o LLM local (`translator.py`)
3. Segmenta e classifica com **rainette** (R) via `rainette_bridge.py`
4. Extrai formas características (χ²) e segmentos por classe
5. Nomeia e resume cada classe com Ollama (`llm_interpreter.py`)
6. Cacheia interpretações e traduções em SQLite (TTL configurável, limpeza automática)
7. Interface **Streamlit** (`app.py`)

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| `app.py` | App completo (corpus → CHD → LLM) |
| `app_demo.py` | Demo só do módulo de interpretação |
| `llm_interpreter.py` | LLM + cache SQLite + TTL + mock |
| `rainette_bridge.py` | Ponte Python ↔ rainette (R) |
| `corpus.py` | Leitura de corpus (livre / IRaMuTeQ) e exportação no formato IRaMuTeQ |
| `translator.py` | Detecção de idioma + tradução para pt-BR com cache |
| `test_traducao_e_correcoes.py` | Testes da tradução e das correções |
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
- **Traduzir textos para português** — detecta o idioma (ou use o idioma de origem fixo) e traduz o que não estiver em português; a CHD roda sobre o texto traduzido
- **🌐 Só traduzir o corpus** — traduz e oferece `corpus_pt.txt` (formato IRaMuTeQ, com a variável `*lang_xx` em cada texto) para baixar, sem rodar a CHD
- **Modo mock** — testa o fluxo sem Ollama (as respostas simuladas ficam separadas no cache)
- **Usar cache SQLite** / **Forçar nova geração**
- **Limpar expiradas** / **Limpar todo o cache**

## Testes (sem Ollama e sem R)

```bash
python test_cache.py
python test_e2e_mock.py
python test_traducao_e_correcoes.py
```

## Tradução

- Idiomas detectados automaticamente: inglês, espanhol, francês, alemão e italiano; outros idiomas (inclusive escrita não latina) aparecem como "outro idioma" e também são traduzidos.
- Textos já em português não são enviados ao modelo.
- Textos longos são divididos em blocos de ~1800 caracteres.
- Traduções ficam no cache SQLite (tabela `translations`) e não expiram.
- Tradução é feita localmente (Ollama), sem enviar dados de pesquisa para a internet.
- **Atenção metodológica:** a CHD passa a analisar o vocabulário da tradução. Registre no método o modelo usado e revise uma amostra das traduções.

## Cache

- Banco: `.cache/interpretations.db`
- Chave: hash de forms + segments + model + temperature + deep
- TTL padrão: 30 dias (nome/resumo) e 7 dias (interpretação profunda)
- Auto-purge de expirados na inicialização do interpretador
