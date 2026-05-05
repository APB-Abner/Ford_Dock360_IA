import os
from importlib.machinery import SourceFileLoader

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from src.pipeline.preprocessor import build_preprocessor
except ModuleNotFoundError:
    build_preprocessor = SourceFileLoader(
        "preprocessor",
        os.path.join(os.path.dirname(__file__), "preprocessor.py"),
    ).load_module().build_preprocessor


RANDOM_STATE = 42

LEAKAGE_COLS = [
    "fez_primeira_revisao_rede",
    "meses_ate_primeira_revisao",
    "perdeu_primeira_revisao",
    "voltou_tarde_revoltado",
    "trouxe_oleo_externo",
    "pede_desconto_revisao",
    "sensibilidade_desconto_pos",
    "qtde_revisoes_24m",
    "share_revisoes_rede_24m",
    "gasto_manutencao_rede_24m",
    "satisfacao_marca_24m",
    "churn_rede_24m",
]


def check_leakage(x):
    found = [col for col in LEAKAGE_COLS if col in x.columns]
    if found:
        raise ValueError(f"Colunas com data leakage em X: {found}")


def _split_columns(x):
    binary_cols = []
    categorical_cols = []
    numeric_cols = []

    for col in x.columns:
        if col == "modelo_veiculo":
            continue

        unique_count = x[col].dropna().nunique()
        if unique_count <= 2:
            binary_cols.append(col)
        elif x[col].dtype == "object" or str(x[col].dtype) == "category":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return numeric_cols, categorical_cols, binary_cols


def train_all_models(
    input_path="data/raw/ford_clientes_operacional_compra.csv",
    target_col="perfil_cluster",
    output_path="reports/model_comparison",
    labels_path="data/processed/cluster_labels.csv",
):
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(input_path)

    # Merge com labels do clustering — target vem da Base 1 via K-Means
    labels = pd.read_csv(labels_path)[["cliente_id", "perfil_cluster"]]
    df = df.merge(labels, left_on="id_cliente", right_on="cliente_id", how="inner")

    if target_col not in df.columns:
        raise ValueError(f"Coluna target ausente: {target_col}")

    drop_cols = ["id_cliente", "cliente_id", "modelo_veiculo", target_col]
    drop_cols += [col for col in LEAKAGE_COLS if col != target_col]

    y = df[target_col]
    x = df.drop(columns=[col for col in drop_cols if col in df.columns])
    check_leakage(x)

    numeric_cols, categorical_cols, binary_cols = _split_columns(x)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)

    models = {
        "LogReg": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in models.items():
        print(f"Treinando {name} com StratifiedKFold n_splits=5...")
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        scores = cross_val_score(pipeline, x, y, cv=cv, scoring="f1_macro", n_jobs=1)
        rows.append(
            {
                "model": name,
                "f1_macro_mean": scores.mean(),
                "f1_macro_std": scores.std(),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    comparison.to_csv(output_path, index=False)
    print(comparison.to_string(index=False))
    print(f"Tabela comparativa salva em {output_path}")

    return comparison


if __name__ == "__main__":
    train_all_models()
