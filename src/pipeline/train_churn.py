import matplotlib
matplotlib.use("Agg")

import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200
MAX_F1_MACRO_WITHOUT_LEAKAGE = 0.92
MAX_AUC_WITHOUT_LEAKAGE = 0.95

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


def audit_features_used(feature_names):
    found = [col for col in LEAKAGE_COLS if col in feature_names]
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

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)

    print(f"AUC-ROC teste: {auc:.4f}")
    assert_metrics_not_suspicious(auc=auc)
    audit_features_used(x.columns)
    if auc < 0.70:
        print("AUC-ROC abaixo da meta >= 0.70 para os dados atuais.")

    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title("Precision-Recall Churn")
    plt.savefig(report_path, format="png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump(model, model_path, compress=3)
    print(f"Precision-Recall curve salva em {report_path}")
    print(f"Modelo salvo em {model_path}")

    return model, auc


if __name__ == "__main__":
    train_churn_model()
