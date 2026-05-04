#!/usr/bin/env bash
set -euo pipefail

echo "== Identidade =="
whoami
id
pwd

echo ""
echo "== Ferramentas =="
python --version || true
pip --version || true
node --version || true
npm --version || true
git --version || true
jq --version || true
codex --version || true

echo ""
echo "== Mounts suspeitos =="
mount | grep -E "docker.sock|/Users|/home|/.ssh|/.aws|/var/run" || true

echo ""
echo "== Variáveis sensíveis =="
env | grep -E "AWS|DATABASE|SECRET|TOKEN|OPENAI|ANTHROPIC|GITHUB" || true

echo ""
echo "== Workspace =="
ls -la /workspace

echo ""
echo "Checagem finalizada."