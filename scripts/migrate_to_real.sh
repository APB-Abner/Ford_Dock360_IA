#!/usr/bin/env bash
# Migracao completa para Caminho B (dataset real Ford)
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

# 3. Atualizar config.py
echo ""
echo "=== 3. Atualizando config.py ==="
cat > src/pipeline/config.py << 'PYEOF'
"""Constantes do pipeline Ford VinGuard - Caminho B (dados reais)."""

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLUSTERS = 4
N_ESTIMATORS = 200

# Janela de churn em meses (padrao industria automotiva)
JANELA_CHURN_MESES = 18

# Features comportamentais derivadas das ordens de servico (NAO usar no classificador)
LEAKAGE_BEHAVIORAL = [
    "qtde_revisoes",
    "meses_desde_ultimo_servico",
    "meses_relacionamento",
    "n_dealers_usados",
    "km_max",
    "pct_agenda",
    "intervalo_medio_revisoes_dias",
    "dias_ate_primeira_revisao",
    "primeiro_servico",
    "ultimo_servico",
    "churn",
]

# Features disponiveis no momento da compra (uso permitido em classificacao)
PURCHASE_FEATURES_NUMERIC = ["ano_modelo", "dias_ate_entrega", "idade_veiculo_meses"]
PURCHASE_FEATURES_CATEGORICAL = ["modelo"]

# Mantido por compatibilidade — apontando para o novo conjunto
LEAKAGE_COLUMNS = LEAKAGE_BEHAVIORAL
PYEOF
echo "  config.py atualizado"

# 4. Garantir __init__.py
echo ""
echo "=== 4. Criando __init__.py ==="
touch src/__init__.py
touch src/pipeline/__init__.py
echo "  __init__.py garantidos"

# 5. Atualizar AGENTS.md
echo ""
echo "=== 5. Atualizando AGENTS.md ==="
cat > AGENTS.md << 'MDEOF'
# Instrucoes para o Agente — Ford VinGuard (Caminho B: dados reais)

## Filosofia geral

- Solucao mais simples e direta. Funcoes em vez de classes.
- Sem Clean Architecture, DDD, Repository Pattern, ou camadas de abstracao.
- Sem logging framework, telemetria ou observabilidade alem do basico.

## Estilo Python

- Funcoes simples, scripts executaveis com if __name__ == '__main__'
- Type hints apenas onde ajudam
- Sem docstrings longas — comentarios curtos
- print() basta para scripts academicos

## Machine Learning — regras criticas

- Todo pre-processamento DENTRO de sklearn.Pipeline
- random_state=42 obrigatorio em tudo que aceita
- stratify=y no train_test_split de classificacao
- F1 Macro (perfil) ou AUC-ROC (churn) como metrica principal — nunca accuracy isolada
- class_weight='balanced' obrigatorio em classificadores

## Dataset (Caminho B — dados reais Ford)

Fonte unica: data/raw/vin_share_Desafio_02.xlsx (aba 'vin_share')
- 602.788 ordens de servico reais Ford Brasil 2020-2026
- 175.554 VINs unicos (anonimizados via hash)
- Granularidade: 1 linha = 1 evento de manutencao

## Data Leakage — regra absoluta

As seguintes features sao DERIVADAS das ordens de servico e representam
comportamento POSTERIOR a compra. JAMAIS podem aparecer no X dos classificadores:

  qtde_revisoes, meses_desde_ultimo_servico, meses_relacionamento,
  n_dealers_usados, km_max, pct_agenda, intervalo_medio_revisoes_dias,
  dias_ate_primeira_revisao, primeiro_servico, ultimo_servico, churn

Features permitidas (momento da compra):
  modelo, ano_modelo, dias_ate_entrega, idade_veiculo_meses

## Targets

- churn (binario): 1 se VIN sem servico ha mais de 18 meses, senao 0
- perfil_cluster: derivado por K-Means nas features comportamentais (gerado por clustering_real.py)

## Pipeline (ordem de execucao)

1. python -m src.pipeline.feature_engineering_real  # agrega VIN-level
2. python -m src.pipeline.clustering_real           # gera perfis
3. python -m src.pipeline.train_classifier_real     # classificador de perfil
4. python -m src.pipeline.train_churn_real          # classificador de churn

## Estrutura

- src/pipeline/feature_engineering_real.py : agregacao por VIN
- src/pipeline/clustering_real.py          : K-Means + nomenclatura de perfis
- src/pipeline/train_classifier_real.py    : LogReg + DT + RF para perfil
- src/pipeline/train_churn_real.py         : RF calibrado para churn
- data/raw/vin_share_Desafio_02.xlsx       : dataset real Ford
- data/processed/vins_agregados.csv         : 1 linha por VIN com features+target
- data/processed/cluster_labels.csv         : VIN_Hash + perfil_cluster
- models/perfil_rf_classifier.joblib        : modelo de perfil
- models/churn_rf_calibrated.joblib         : modelo de churn

## Graficos

- matplotlib.use('Agg') na primeira linha de scripts que geram graficos
- Salvar com plt.savefig(..., dpi=150, bbox_inches='tight') antes de plt.close()
- Sempre os.makedirs('reports', exist_ok=True) antes
MDEOF
echo "  AGENTS.md atualizado"

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
echo "     python -m src.pipeline.train_classifier_real"
echo "     python -m src.pipeline.train_churn_real"
echo ""
echo "  3. Modelos vao aparecer em models/"
echo "  4. Graficos em reports/"
