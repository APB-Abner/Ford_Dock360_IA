"""
Registro MLflow dos experimentos principais do Ford VinGuard.

Este script nao treina modelos. Ele registra no MLflow os artefatos ja gerados
pelo pipeline real, para evidencia academica de experimentacao e auditoria.
"""

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.pipeline.config import (
    CLUSTER_FEATURES_POS_VENDA,
    DATA_CORTE,
    JANELA_CHURN_MESES,
    SNAPSHOT_FEATURES_CATEGORICAL,
    SNAPSHOT_FEATURES_NUMERIC,
    TARGET_CHURN,
)


TRACKING_URI = "file:./mlruns"

CHURN_EXPERIMENT = "ford_churn_pos_venda"
CLUSTERING_EXPERIMENT = "ford_segmentacao_pos_venda"
CLASSIFIER_EXPERIMENT = "ford_segmento_classifier_experimental"

SNAPSHOTS_PATH = Path("data/processed/snapshots_pos_venda.csv")
CHURN_DATASET_PATH = Path("data/processed/dataset_churn_pos_venda.csv")
SEGMENTOS_PATH = Path("data/processed/segmentos_pos_venda.csv")

CHURN_MODEL_PATH = Path("models/churn_pos_venda_rf_calibrated.joblib")
KMEANS_MODEL_PATH = Path("models/kmeans_segmentador_pos_venda.joblib")
SEGMENTO_MODEL_PATH = Path("models/segmento_pos_venda_classifier_experimental.joblib")

CHURN_REPORTS = [
    Path("reports/precision_recall_churn_pos_venda.png"),
    Path("reports/confusion_matrix_churn_pos_venda.png"),
    Path("reports/feature_importance_churn_pos_venda.csv"),
]
CLUSTERING_REPORTS = [
    Path("reports/elbow_silhouette_pos_venda.png"),
    Path("reports/clusters_pca_pos_venda.png"),
]
CLASSIFIER_REPORTS = [
    Path("reports/model_comparison_segmento_experimental.csv"),
    Path("reports/decision_tree_segmento_experimental.png"),
    Path("reports/feature_importance_segmento_experimental.png"),
    Path("reports/confusion_matrix_segmento_experimental.png"),
    Path("reports/feature_importance.csv"),
]


def _setup_mlflow():
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)


def _log_existing_artifacts(paths):
    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path))
        else:
            mlflow.set_tag(f"missing_artifact_{path.name}", str(path))


def _log_model_artifact(path):
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path="models")
        sidecar = path.with_suffix(".sha256")
        if sidecar.exists():
            mlflow.log_artifact(str(sidecar), artifact_path="models")
    else:
        mlflow.set_tag("missing_model", str(path))


def _log_dataset_shape(path, prefix):
    if not path.exists():
        mlflow.set_tag(f"{prefix}_missing", str(path))
        return None

    df = pd.read_csv(path)
    mlflow.log_param(f"{prefix}_path", str(path))
    mlflow.log_metric(f"{prefix}_rows", len(df))
    mlflow.log_metric(f"{prefix}_columns", len(df.columns))
    return df


def registrar_churn():
    mlflow.set_experiment(CHURN_EXPERIMENT)
    with mlflow.start_run(run_name="churn_pos_venda_rf_calibrated"):
        mlflow.set_tags({
            "pipeline_step": "train_churn_real",
            "target": TARGET_CHURN,
            "data_corte": DATA_CORTE,
            "janela_churn_meses": str(JANELA_CHURN_MESES),
            "tracking_type": "artifact_registry",
        })
        mlflow.log_param("model_path", str(CHURN_MODEL_PATH))
        mlflow.log_param("features_numeric", ",".join(SNAPSHOT_FEATURES_NUMERIC))
        mlflow.log_param("features_categorical", ",".join(SNAPSHOT_FEATURES_CATEGORICAL))

        df = _log_dataset_shape(CHURN_DATASET_PATH, "churn_dataset")
        if df is not None and TARGET_CHURN in df.columns:
            y = df[TARGET_CHURN].astype(int)
            mlflow.log_metric("target_churn_rate", float(y.mean()))

            feature_cols = SNAPSHOT_FEATURES_NUMERIC + SNAPSHOT_FEATURES_CATEGORICAL
            if CHURN_MODEL_PATH.exists() and all(col in df.columns for col in feature_cols):
                model = joblib.load(CHURN_MODEL_PATH)
                scores = model.predict_proba(df[feature_cols])[:, 1]
                preds = (scores >= 0.5).astype(int)
                mlflow.log_metric("auc_full_dataset_diagnostic", roc_auc_score(y, scores))
                mlflow.log_metric("f1_full_dataset_diagnostic", f1_score(y, preds, average="macro"))

        _log_existing_artifacts(CHURN_REPORTS)
        _log_model_artifact(CHURN_MODEL_PATH)
        print(f"Registrado MLflow: {CHURN_EXPERIMENT}")


def registrar_clustering():
    mlflow.set_experiment(CLUSTERING_EXPERIMENT)
    with mlflow.start_run(run_name="kmeans_segmentador_pos_venda"):
        mlflow.set_tags({
            "pipeline_step": "clustering_real",
            "data_corte": DATA_CORTE,
            "tracking_type": "artifact_registry",
        })
        mlflow.log_param("model_path", str(KMEANS_MODEL_PATH))
        mlflow.log_param("features", ",".join(CLUSTER_FEATURES_POS_VENDA))

        _log_dataset_shape(SNAPSHOTS_PATH, "snapshots")
        segmentos = _log_dataset_shape(SEGMENTOS_PATH, "segmentos")
        if segmentos is not None and "segmento_pos_venda" in segmentos.columns:
            counts = segmentos["segmento_pos_venda"].value_counts(normalize=True)
            for segmento, pct in counts.items():
                mlflow.log_metric(f"pct_segmento_{segmento}", float(pct))

        _log_existing_artifacts(CLUSTERING_REPORTS)
        _log_model_artifact(KMEANS_MODEL_PATH)
        print(f"Registrado MLflow: {CLUSTERING_EXPERIMENT}")


def registrar_classificador_segmento():
    mlflow.set_experiment(CLASSIFIER_EXPERIMENT)
    with mlflow.start_run(run_name="segmento_pos_venda_classifier_experimental"):
        mlflow.set_tags({
            "pipeline_step": "train_classifier_real",
            "target": "segmento_pos_venda",
            "tracking_type": "artifact_registry",
            "observacao": "modelo experimental; K-Means segue como segmentador principal",
        })
        mlflow.log_param("model_path", str(SEGMENTO_MODEL_PATH))
        mlflow.log_param("features_numeric", "ano_modelo,idade_veiculo_meses_ate_corte")
        mlflow.log_param("features_categorical", "modelo")

        _log_dataset_shape(SNAPSHOTS_PATH, "snapshots")
        _log_dataset_shape(SEGMENTOS_PATH, "segmentos")

        comparison_path = Path("reports/model_comparison_segmento_experimental.csv")
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            for _, row in comparison.iterrows():
                model_name = str(row["model"]).lower()
                mlflow.log_metric(f"{model_name}_f1_macro_cv_mean", float(row["f1_macro_cv_mean"]))
                mlflow.log_metric(f"{model_name}_f1_macro_cv_std", float(row["f1_macro_cv_std"]))
                mlflow.log_metric(f"{model_name}_f1_macro_test", float(row["f1_macro_test"]))

        _log_existing_artifacts(CLASSIFIER_REPORTS)
        _log_model_artifact(SEGMENTO_MODEL_PATH)
        print(f"Registrado MLflow: {CLASSIFIER_EXPERIMENT}")


def main():
    _setup_mlflow()
    registrar_clustering()
    registrar_churn()
    registrar_classificador_segmento()
    print("MLflow tracking concluido. Use: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
