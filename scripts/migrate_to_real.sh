#!/usr/bin/env bash
# Fluxo pos-venda com dados reais Ford
# Executar de /workspace: bash scripts/migrate_to_real.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "Raiz do projeto: $ROOT"
echo ""

# 1. Apagar scripts antigos do caminho A
echo "=== 1. Removendo scripts do Caminho A (sintetico) ==="
rm -f src/pipeline/clustering.py
rm -f src/pipeline/train_classifier.py
rm -f src/pipeline/train_churn.py
rm -f src/pipeline/generate_data.py
echo "  Scripts antigos removidos"

# 2. Apagar dados sinteticos antigos
echo ""
echo "=== 2. Removendo CSVs sinteticos antigos ==="
rm -f data/raw/ford_clientes_historico_completo.csv
rm -f data/raw/ford_clientes_operacional_compra.csv
rm -f data/processed/cluster_labels.csv
echo "  CSVs sinteticos removidos"

# 3. Config ja vive em src/pipeline/config.py
echo ""
echo "=== 3. Configuracao pos-venda ==="
python - <<'PYEOF'
from src.pipeline.config import DATA_CORTE, JANELA_CHURN_MESES, TARGET_CHURN
print(f"  DATA_CORTE={DATA_CORTE}")
print(f"  JANELA_CHURN_MESES={JANELA_CHURN_MESES}")
print(f"  TARGET_CHURN={TARGET_CHURN}")
PYEOF

# 4. Garantir __init__.py
echo ""
echo "=== 4. Criando __init__.py ==="
touch src/__init__.py
touch src/pipeline/__init__.py
echo "  __init__.py garantidos"

# 5. AGENTS.md deve refletir o fluxo pos-venda do repositorio
echo ""
echo "=== 5. AGENTS.md ==="
echo "  Mantido no repositorio; nao sobrescrever automaticamente."

# 6. Atualizar config da API para apontar para o novo modelo
echo ""
echo "=== 6. Verificando estrutura de pastas ==="
mkdir -p data/processed models reports
echo "  Pastas garantidas"

echo ""
echo "=== Migracao concluida ==="
echo ""
echo "PROXIMO PASSO MANUAL:"
echo "  1. Copie o arquivo Excel para data/raw/:"
echo "     cp ~/Downloads/vin_share_Desafio_02.xlsx data/raw/"
echo ""
echo "  2. Rode os 4 scripts em sequencia:"
echo "     export PYTHONPATH=/workspace:\$PYTHONPATH"
echo "     python -m src.pipeline.feature_engineering_real"
echo "     python -m src.pipeline.clustering_real"
echo "     python -m src.pipeline.train_churn_real"
echo ""
echo "     # Opcional/experimental:"
echo "     python -m src.pipeline.train_classifier_real"
echo ""
echo "  3. Modelos vao aparecer em models/"
echo "  4. Graficos em reports/"
