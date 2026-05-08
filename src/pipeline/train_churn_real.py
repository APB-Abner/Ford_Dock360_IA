"""
Classificador de Churn — Caminho B (dataset real Ford)

Substitui train_churn.py original. Treina classificador binario para prever
churn (18 meses sem servico) usando APENAS features disponiveis no momento da venda.

Features (mesmas do classificador de perfil — sem leakage):
  - modelo, ano_modelo, dias_ate_entrega, idade_veiculo_meses

Modelo:
  RandomForestClassifier + CalibratedClassifierCV (isotonic, cv=5)

Saidas:
  models/churn_rf_calibrated.joblib
  reports/precision_recall_churn.png
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import PrecisionRecallDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.config import LEAKAGE_BEHAVIORAL, N_ESTIMATORS, RANDOM_STATE, TEST_SIZE


VINS_PATH = "data/processed/vins_agregados.csv"
MODEL_PATH = "models/churn_rf_calibrated.joblib"
REPORT_PATH = "reports/precision_recall_churn.png"

PURCHASE_FEATURES_NUMERIC = ["ano_modelo", "dias_ate_entrega", "idade_veiculo_meses"]
PURCHASE_FEATURES_CATEGORICAL = ["modelo"]


def check_leakage(x):
    found = [col for col in LEAKAGE_BEHAVIORAL if col in x.columns]
    if found:
        raise ValueError(f"Colunas comportamentais (leakage) em X: {found}")


def _build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, PURCHASE_FEATURES_NUMERIC),
        ("categorical", categorical_pipeline, PURCHASE_FEATURES_CATEGORICAL),
    ], remainder="drop")


def _load_data():
    if not os.path.exists(VINS_PATH):
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {VINS_PATH}\n"
            f"Rode antes: python -m src.pipeline.feature_engineering_real"
        )

    df = pd.read_csv(VINS_PATH)

    feature_cols = PURCHASE_FEATURES_NUMERIC + PURCHASE_FEATURES_CATEGORICAL
    x = df[feature_cols].copy()
    y = df["churn"].astype(int)

    check_leakage(x)

    print(f"Dataset: {len(x):,} VINs")
    print(f"Distribuicao de churn:\n{y.value_counts(normalize=True).round(3)}")
    return x, y


def assert_metrics_not_suspicious(auc):
    if auc > 0.98:
        raise ValueError(
            f"AUC-ROC = {auc:.4f} acima de 0.95 — suspeita de leakage. "
            f"Auditar features do X de treino."
        )


def train_churn_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    x, y = _load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = _build_preprocessor()
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf)])

    print("\n=== Treinando RandomForest com calibracao isotonica (cv=5) ===")
    print("Esse passo treina o modelo 5 vezes e pode demorar alguns minutos...")

    model = CalibratedClassifierCV(estimator=pipeline, cv=5, method="isotonic")
    model.fit(x_train, y_train)

    # Avaliacao
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_score)
    print(f"\n=== Avaliacao no teste ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['no_churn', 'churn'])}")

    # Anti-leakage check
    assert_metrics_not_suspicious(auc)

    # Curva precision-recall
    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title(f"Precision-Recall — Churn (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(REPORT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSalvo: {REPORT_PATH}")

    # Salvar modelo
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Salvo: {MODEL_PATH}")

    return model, auc


if __name__ == "__main__":
    train_churn_model()
