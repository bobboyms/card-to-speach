# ============================
# Stage 1: Builder
# ============================
FROM python:3.11-slim AS builder

WORKDIR /app

# Dependências de sistema para compilar pacotes (praat-parselmouth, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    python3-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python do projeto (a partir do pyproject.toml)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Se você tiver arquivos extras para build (por ex. src/), pode copiá-los aqui
# e rodar testes/build, se quiser:
# COPY . .
# RUN pytest ...


# ============================
# Stage 2: Runtime
# ============================
FROM python:3.11-slim

WORKDIR /app

# Dependências em tempo de execução:
# - ffmpeg: processamento de áudio/vídeo
# - build-essential/g++: necessário para TorchInductor (compilação JIT)
# - espeak-ng / festival / mbrola: backends para phonemizer
# - libasound2-dev / libsndfile1-dev: áudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    g++ \
    espeak-ng \
    libasound2-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar libs Python já instaladas do builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar código da aplicação
COPY . .

# Criar pastas usadas pela app e dar permissão
RUN mkdir -p temp_files gravacoes && \
    chmod 777 temp_files gravacoes

# Usuário não-root
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Porta da API
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
