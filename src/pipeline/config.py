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
