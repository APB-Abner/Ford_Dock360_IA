"""Constantes do pipeline Ford VinGuard - pos-venda com dados reais."""

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLUSTERS = 4
N_ESTIMATORS = 200

# Snapshot temporal: features ate DATA_CORTE, target depois da DATA_CORTE.
DATA_CORTE = "2024-10-31"
JANELA_CHURN_MESES = 18

# Features validas para churn pos-venda: calculadas somente ate a data de corte.
SNAPSHOT_FEATURES_NUMERIC = [
    "ano_modelo",
    "qtde_revisoes_ate_corte",
    "meses_desde_ultimo_servico_ate_corte",
    "meses_relacionamento_ate_corte",
    "n_dealers_usados_ate_corte",
    "km_max_ate_corte",
    "pct_agenda_ate_corte",
    "intervalo_medio_revisoes_dias_ate_corte",
    "dias_ate_primeira_revisao",
    "idade_veiculo_meses_ate_corte",
]

SNAPSHOT_FEATURES_CATEGORICAL = ["modelo"]

CLUSTER_FEATURES_POS_VENDA = [
    "qtde_revisoes_ate_corte",
    "meses_desde_ultimo_servico_ate_corte",
    "meses_relacionamento_ate_corte",
    "n_dealers_usados_ate_corte",
    "km_max_ate_corte",
    "pct_agenda_ate_corte",
    "intervalo_medio_revisoes_dias_ate_corte",
]

TARGET_CHURN = "churn_futuro_18m"

# Padroes proibidos no X: indicam target, futuro ou informacao pos-corte.
LEAKAGE_PATTERNS = ["_pos_corte", "futuro", "target", "churn", "voltou"]

LEAKAGE_COLUMNS = [
    TARGET_CHURN,
    "voltou_pos_corte",
    "primeiro_servico_pos_corte",
    "ultimo_servico_pos_corte",
    "qtd_servicos_pos_corte",
]
