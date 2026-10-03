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
| `cli.py` | Linha de comando: `run`, `traduzir`, `config-exemplo` |
| `pipeline.py` | Pipeline sem interface + relatório/JSON/CSV |
| `analises.py` | Estatísticas, AFC, similitude, nuvem e figuras |
| `relatorio_docx.py` | Relatório Word |
| `validacao.py` | Revisão de nomes, atribuição às cegas, kappa de Cohen |
| `limpeza.py` | Limpeza IRaMuTeQ, validação de cabeçalhos, planilha → corpus |
| `transcricao.py` | Áudio → texto com Whisper local (faster-whisper) |
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
pip install -r requirements-r.txt
# para transcrever áudio:
pip install -r requirements-audio.txt
```

## Executar com Docker (recomendado)

Tudo pronto — R, rainette, Python e Ollama — sem instalar nada além do Docker:

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b   # só na primeira vez
# abra http://localhost:8501
```

Linha de comando dentro do Docker (coloque os arquivos na pasta `dados/`):

```bash
docker compose run --rm textome python cli.py run dados/corpus.txt --saida dados/resultados
```

## Preparar o corpus

### Limpeza automática (regras do IRaMuTeQ)

Aplicada por padrão na análise (`clean: true`) e disponível como comando:

```bash
python cli.py limpar corpus.txt --termos termos.txt --saida limpos/
```

| Antes | Depois |
|---|---|
| `"não"`, `“sim”` | `não`, `sim` (aspas removidas) |
| `guarda-chuva`, `bem-te-vi` | `guarda_chuva`, `bem_te_vi` |
| `disse-me`, `fazê-lo` | `disse me`, `fazê lo` (pronome separado) |
| `Sistema Único de Saúde` (em `termos.txt`) | `sistema_único_de_saúde` |
| `50%`, `&`, `...` | `50 por_cento`, `e`, `.` |
| `*`, `$`, `#`, `@`, emojis, URLs | removidos |
| `**** *Suj_01 *região_Sul` | `**** *Suj_01 *regiao_sul` (cabeçalho corrigido) |

Linhas temáticas `-*tema` são preservadas. O relatório lista tudo que foi alterado.
`termos.txt` tem uma expressão composta por linha (ou use `compound_terms` no YAML).

### Planilha de respostas abertas → corpus

```bash
python cli.py importar respostas.csv --texto resposta --variaveis sexo idade escolaridade --saida corpus.txt
```

Aceita CSV (separador `,` `;` ou tab, UTF-8 ou Windows) e XLSX. Cada linha vira um texto
com cabeçalho `**** *sexo_f *idade_30 *escolaridade_superior`; respostas vazias são ignoradas.

### Áudio de entrevistas → corpus (Whisper local)

```bash
pip install -r requirements-audio.txt
python cli.py transcrever audios/ --variaveis participantes.csv --saida corpus_entrevistas.txt
```

- `participantes.csv` tem a coluna `arquivo` (nome do áudio, com ou sem extensão) e uma coluna por variável.
- Formatos: mp3, wav, m4a, ogg, opus, flac, mp4, webm etc.
- `--modelo-whisper`: `tiny`, `base`, `small` (padrão), `medium`, `large-v3` — maiores são mais precisos e mais lentos.
- `--idioma pt` força português; o padrão detecta o idioma. Áudio em outro idioma é traduzido depois pelo `run`/`traduzir`.
- O modelo é baixado uma vez na primeira execução; depois tudo roda offline.
- **Revise a transcrição** antes da análise. O Whisper não separa entrevistador e entrevistado: remova as perguntas do entrevistador se elas não devem entrar no corpus.

## Análises complementares (estilo IRaMuTeQ)

Geradas automaticamente após a CHD (`analyses: true` no YAML):

| Análise | O que mostra |
|---|---|
| Estatísticas textuais | segmentos, % classificado, ocorrências, formas, hápax |
| Dendrograma | árvore da CHD com as formas de maior χ² por classe (gerado pelo rainette) |
| Distribuição das classes | % de segmentos por classe |
| AFC | formas características no plano fatorial 1×2 (exige ≥ 3 classes) |
| Similitude | árvore máxima de coocorrência das 50 formas mais frequentes; cor = classe em que a forma é mais característica, tamanho = frequência |
| Nuvem de palavras | formas mais frequentes do corpus |

Cada classe tem sempre a mesma cor em todos os gráficos (paleta validada para daltonismo,
com marcadores diferentes como segunda pista). Rótulos que não cabem sem sobreposição
são omitidos e a figura informa quantos — a lista completa está em `classes.json`.

## Validação humana e concordância (kappa)

A IA **sugere**; o pesquisador **decide** — e o processo fica registrado:

```bash
python cli.py validar resultados/ --ia        # ou a seção "✅ Validação" no app
```

1. **Revisão dos nomes:** para cada classe, aceite (Enter), edite (digite) ou rejeite (`-`)
   o nome sugerido. Registra-se a taxa de aceitação.
2. **Atribuição às cegas:** 5 segmentos por classe são sorteados — *fora* dos que foram
   mostrados à IA — e você indica a classe que melhor descreve cada um, sem ver a classe
   da CHD. Com `--ia`, o modelo faz o mesmo teste (temperatura 0).
3. **Concordância:** kappa de Cohen para pesquisador × CHD, IA × CHD e pesquisador × IA,
   com interpretação de Landis e Koch (1977) e matrizes de confusão.

Saídas: `validacao.docx` / `validacao.md` (com um parágrafo pronto para a seção de método)
e `validacao.json` (todas as respostas, para auditoria).

As respostas da IA usam **saída estruturada** (JSON Schema no Ollama ≥ 0.5; em versões
antigas, recua para `format="json"`).

## Linha de comando (sem clicar, em lote)

```bash
python cli.py config-exemplo > pesquisa.yaml        # gera a configuração comentada
python cli.py run corpus.txt --config pesquisa.yaml --saida resultados/
python cli.py run pasta_com_txts/ --k 6              # um subdiretório de resultado por arquivo
python cli.py traduzir entrevistas_en.txt --saida traduzidos/   # só tradução
```

Opções úteis: `--modelo`, `--k`, `--idioma-origem`, `--sem-traducao`, `--sem-limpeza`, `--termos`, `--sem-llm`,
`--profunda`, `--forcar` (ignora o cache), `--mock` (teste sem Ollama).
A linha de comando sobrepõe o YAML. Em lote, um arquivo com erro não interrompe os demais.

Arquivos gerados em cada pasta de resultado:

| Arquivo | Conteúdo |
|---------|----------|
| `relatorio.docx` | **Relatório Word** pronto para o capítulo de resultados: método, estatísticas textuais, figuras numeradas com legenda e fonte, uma seção por classe (para PDF: "Salvar como PDF" no Word/LibreOffice) |
| `relatorio.md` | O mesmo conteúdo em Markdown |
| `figuras/` | `dendrograma.png` (rainette), `classes.png`, `afc.png`, `similitude.png`, `nuvem.png` (200 dpi) |
| `classes.json` | Todos os resultados, para outras análises |
| `formas.csv` | Formas por classe (separado por `;`, abre direto no Excel) |
| `corpus_final.txt` | Corpus **exatamente como foi analisado** (traduzido e limpo), no formato IRaMuTeQ, com `*lang_xx` |
| `config_usada.yaml` | Parâmetros exatos usados (reprodutibilidade) |

O endereço do Ollama vem da variável `OLLAMA_HOST` (padrão `http://localhost:11434`).

## Executar a interface sem Docker

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
python test_cli.py
python test_limpeza_transcricao.py
python test_analises.py
python test_validacao.py
```

Os testes rodam automaticamente no GitHub Actions a cada envio.

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
