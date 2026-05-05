#!/usr/bin/env bash
set -euo pipefail

mkdir -p notebooks src/pipeline data/raw data/processed models reports tests

echo "Container criado."
echo "User: $(whoami)"
echo "PWD: $(pwd)"

command -v python && python --version || true
command -v python3 && python3 --version || true
command -v node && node --version || true
command -v npm && npm --version || true
command -v codex && codex --version || true
