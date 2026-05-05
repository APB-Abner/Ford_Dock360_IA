ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/python:3.11-bookworm
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    jq \
    sudo \
    bash \
    build-essential \
    libpq-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN if ! command -v node >/dev/null 2>&1; then \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs; \
    fi \
    && npm install -g @openai/codex \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements.txt /tmp/requirements-dev.txt

USER vscode
WORKDIR /workspace
