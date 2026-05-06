import matplotlib
matplotlib.use("Agg")

import hashlib
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.config import LEAKAGE_COLUMNS


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200
MAX_F1_MACRO_WITHOUT_LEAKAGE = 0.92
MAX_AUC_WITHOUT_LEAKAGE = 0.98


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


def _load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    return pd.read_csv(path)


def _normalize_cliente_id(df):
    if "cliente_id" in df.columns:
        return df
    if "id_cliente" in df.columns:
        return df.rename(columns={"id_cliente": "cliente_id"})
    raise ValueError("Base sem coluna cliente_id ou id_cliente")


def _split_columns(x):
    numeric_cols = []
    categorical_cols = []

    for col in x.columns:
        if x[col].dtype == "object" or str(x[col].dtype) == "category":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def _metrics_at_threshold(y_test, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return {
        f"precision_t{int(threshold * 100)}": precision_score(y_test, y_pred, zero_division=0),
        f"recall_t{int(threshold * 100)}": recall_score(y_test, y_pred, zero_division=0),
        f"f1_macro_t{int(threshold * 100)}": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def train_churn_model(
    features_path="data/raw/ford_clientes_operacional_compra.csv",
    target_path="data/raw/ford_clientes_historico_completo.csv",
    report_path="reports/precision_recall_churn",
    model_path="models/churn_rf_calibrated.joblib",
):
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    features = _normalize_cliente_id(_load_csv(features_path))
    targets = _normalize_cliente_id(_load_csv(target_path))

    if "churn_rede_24m" not in targets.columns:
        raise ValueError("Coluna target ausente: churn_rede_24m")

    targets = targets[["cliente_id", "churn_rede_24m"]]
    df = features.merge(targets, on="cliente_id", how="inner")
    if df.empty:
        raise ValueError("Join por cliente_id nao retornou linhas")

    y = df["churn_rede_24m"].astype(int)
    x = df.drop(columns=["cliente_id", "churn_rede_24m"])
    check_leakage(x)

    numeric_cols, categorical_cols = _split_columns(x)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )

    model = CalibratedClassifierCV(
        estimator=pipeline,
        cv=5,
        method="isotonic",
    )

    # Limitacao: split aleatorio — Base 1 sem coluna temporal. Split por safra seria necessario em producao.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"AUC-ROC teste: {auc:.4f}")
    print(f"PR-AUC teste: {pr_auc:.4f}")
    print(f"F1-Macro teste: {f1_macro:.4f}")
    assert_metrics_not_suspicious(f1_macro=f1_macro, auc=auc)
    audit_features_used(x.columns)
    if auc < 0.70:
        print("AUC-ROC abaixo da meta >= 0.70 para os dados atuais.")

    pr_curve_path = f"{report_path}.png"
    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title("Precision-Recall Churn")
    plt.savefig(pr_curve_path, dpi=150, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    cm_path = "reports/confusion_matrix_churn.png"
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Confusion Matrix Churn")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump(model, model_path, compress=3)
    digest = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    Path(model_path).with_suffix(".sha256").write_text(digest.hexdigest() + "\n")
    print(f"Precision-Recall curve salva em {pr_curve_path}")
    print(f"Modelo salvo em {model_path}")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_macro", f1_macro)

        for threshold in [0.3, 0.5, 0.7]:
            for name, value in _metrics_at_threshold(y_test, y_score, threshold).items():
                mlflow.log_metric(name, value)

        mlflow.log_artifact(pr_curve_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(model_path)

    return model, auc


if __name__ == "__main__":
    train_churn_model()
