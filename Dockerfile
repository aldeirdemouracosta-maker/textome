# Textome: R (rainette + quanteda) + Python (Streamlit, rpy2, Ollama client)
# Construir:  docker build -t textome .
# Usar:       docker compose up   (app em http://localhost:8501, com Ollama)
FROM rocker/r-ver:4.5

# Pacotes R (binários do repositório configurado na imagem rocker)
RUN install2.r --error --skipinstalled --ncpus -1 quanteda rainette \
    && rm -rf /tmp/downloaded_packages

# Python + dependências de compilação do rpy2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 python3-venv python3-dev build-essential \
       libpcre2-dev libbz2-dev liblzma-dev zlib1g-dev libicu-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    R_HOME=/usr/local/lib/R \
    LD_LIBRARY_PATH=/usr/local/lib/R/lib \
    OLLAMA_HOST=http://ollama:11434
RUN python3 -m venv $VIRTUAL_ENV

WORKDIR /app
COPY requirements.txt requirements-r.txt requirements-audio.txt requirements-mcp.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-r.txt -r requirements-audio.txt \
    -r requirements-mcp.txt

COPY . .

# Verifica na construção que R + rainette + rpy2 estão funcionando
RUN python -c "from rainette_bridge import RainetteBridge; RainetteBridge(); print('rainette OK')" \
    && python -c "import faster_whisper; print('whisper OK')"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
