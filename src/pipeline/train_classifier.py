import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipeline.config import LEAKAGE_COLUMNS, LEAKAGE_COLUMNS as LEAKAGE_COLS
from src.pipeline.preprocessor import build_preprocessor


RANDOM_STATE = 42
MAX_F1_MACRO_WITHOUT_LEAKAGE = 0.92
MAX_AUC_WITHOUT_LEAKAGE = 0.95


def check_leakage(x):
    found = [col for col in LEAKAGE_COLUMNS if col in x.columns]
    if found:
        raise ValueError(f"Colunas com data leakage em X: {found}")


def audit_features_used(feature_names):
    found = [col for col in LEAKAGE_COLUMNS if col in feature_names]
    if found:
        raise ValueError(f"Auditoria pos-treino encontrou leakage em features: {found}")
    print("Auditoria pos-treino: nenhuma coluna proibida em X.")


def assert_metrics_not_suspicious(f1_macro=None, auc=None):
    if f1_macro is not None and f1_macro > MAX_F1_MACRO_WITHOUT_LEAKAGE:
        raise AssertionError(
            f"F1 Macro suspeito ({f1_macro:.4f}) acima de {MAX_F1_MACRO_WITHOUT_LEAKAGE:.2f}; possivel leakage."
        )
    if auc is not None and auc > MAX_AUC_WITHOUT_LEAKAGE:
        raise AssertionError(
            f"AUC-ROC suspeito ({auc:.4f}) acima de {MAX_AUC_WITHOUT_LEAKAGE:.2f}; possivel leakage."
        )


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
    drop_cols += [col for col in LEAKAGE_COLUMNS if col != target_col]

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
        assert_metrics_not_suspicious(f1_macro=scores.mean())
        rows.append(
            {
                "model": name,
                "f1_macro_mean": scores.mean(),
                "f1_macro_std": scores.std(),
            }
        )

    audit_features_used(x.columns)
    comparison = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
    comparison.to_csv(output_path, index=False)
    print(comparison.to_string(index=False))
    print(f"Tabela comparativa salva em {output_path}")

    return comparison


if __name__ == "__main__":
    train_all_models()


def train_and_save_best(
    input_path="data/raw/ford_clientes_operacional_compra.csv",
    labels_path="data/processed/cluster_labels.csv",
    model_path="models/perfil_rf_classifier.joblib",
):
    import hashlib
    from pathlib import Path
    import joblib

    comparison = train_all_models(input_path=input_path, labels_path=labels_path)

    df = pd.read_csv(input_path)
    labels = pd.read_csv(labels_path)[["cliente_id", "perfil_cluster"]]
    df = df.merge(labels, left_on="id_cliente", right_on="cliente_id", how="inner")

    drop_cols = ["id_cliente", "cliente_id", "modelo_veiculo", "perfil_cluster"]
    drop_cols += [col for col in LEAKAGE_COLUMNS if col != "perfil_cluster"]
    y = df["perfil_cluster"]
    x = df.drop(columns=[col for col in drop_cols if col in df.columns])
    check_leakage(x)

    numeric_cols, categorical_cols, binary_cols = _split_columns(x)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=2,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf)])
    pipeline.fit(x, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)

    digest = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    Path(model_path).with_suffix(".sha256").write_text(digest.hexdigest() + "\n")
    print(f"Modelo de perfil salvo em {model_path}")
    return pipeline


if __name__ == "__main__":
    train_and_save_best()
