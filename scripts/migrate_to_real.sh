#!/usr/bin/env bash
# Fluxo pos-venda com dados reais Ford
# Executar de /workspace: bash scripts/migrate_to_real.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "Raiz do projeto: $ROOT"
echo ""

# 1. Config ja vive em src/pipeline/config.py
echo ""
echo "=== 1. Configuracao pos-venda ==="
python - <<'PYEOF'
from src.pipeline.config import DATA_CORTE, JANELA_CHURN_MESES, TARGET_CHURN
print(f"  DATA_CORTE={DATA_CORTE}")
print(f"  JANELA_CHURN_MESES={JANELA_CHURN_MESES}")
print(f"  TARGET_CHURN={TARGET_CHURN}")
PYEOF

# 2. Garantir __init__.py
echo ""
echo "=== 2. Criando __init__.py ==="
touch src/__init__.py
touch src/pipeline/__init__.py
echo "  __init__.py garantidos"

# 3. AGENTS.md deve refletir o fluxo pos-venda do repositorio
echo ""
echo "=== 3. AGENTS.md ==="
echo "  Mantido no repositorio; nao sobrescrever automaticamente."

# 4. Garantir estrutura de saida
echo ""
echo "=== 4. Verificando estrutura de pastas ==="
mkdir -p data/processed models reports
echo "  Pastas garantidas"

echo ""
echo "=== Migracao concluida ==="
echo ""
echo "PROXIMO PASSO MANUAL:"
echo "  1. Copie o arquivo Excel para data/raw/:"
echo "     cp ~/Downloads/vin_share_Desafio_02.xlsx data/raw/"
echo ""
echo "  2. Rode os 3 scripts em sequencia:"
echo "     export PYTHONPATH=/workspace:\$PYTHONPATH"
echo "     python -m src.pipeline.feature_engineering_real"
echo "     python -m src.pipeline.clustering_real"
echo "     python -m src.pipeline.train_churn_real"
echo ""
echo "  3. Modelos vao aparecer em models/"
echo "  4. Graficos em reports/"
