"""MLflow Tracking para experimentos pos-venda com snapshots por VIN."""

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.pipeline.config import (
    CLUSTER_FEATURES_POS_VENDA,
    RANDOM_STATE,
    SNAPSHOT_FEATURES_CATEGORICAL,
    SNAPSHOT_FEATURES_NUMERIC,
    TARGET_CHURN,
    TEST_SIZE,
)
from src.pipeline.clustering_real import _make_pipeline
from src.pipeline.train_classifier_real import EXPERIMENT_FEATURES_CATEGORICAL, EXPERIMENT_FEATURES_NUMERIC
from src.pipeline.train_classifier_real import _build_preprocessor as build_preprocessor_perfil
from src.pipeline.train_churn_real import _build_preprocessor as build_preprocessor_churn
from src.pipeline.train_churn_real import check_temporal_leakage

VINS_PATH = "data/processed/dataset_churn_pos_venda.csv"
SEGMENTOS_PATH = "data/processed/segmentos_pos_venda.csv"
TRACKING_URI = "file:./mlruns"


def setup_mlflow(tracking_uri=TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def _load_data():
    df = pd.read_csv(VINS_PATH)
    return df


def register_kmeans_experiment(experiment_name="ford_segmentacao_kmeans"):
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    df = _load_data()
    x = df[CLUSTER_FEATURES_POS_VENDA]

    candidate_rows = []
    for k in range(2, 9):
        pipeline = _make_pipeline(k)
        labels = pipeline.fit_predict(x)
        x_scaled = pipeline[:-1].transform(x)
        sample_size = min(10000, len(x))
        candidate_rows.append({
            "k": k,
            "silhouette_score": silhouette_score(
                x_scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE
            ),
            "inertia": pipeline.named_steps["kmeans"].inertia_,
        })
    best = max(candidate_rows, key=lambda row: row["silhouette_score"])

    for row in candidate_rows:
        with mlflow.start_run(run_name=f"kmeans_k_{row['k']}"):
            mlflow.log_param("k", row["k"])
            mlflow.log_metric("silhouette_score", row["silhouette_score"])
            mlflow.log_metric("inertia", row["inertia"])

    with mlflow.start_run(run_name="kmeans_selected_k"):
        mlflow.log_param("selected_k", best["k"])
        mlflow.log_metric("selected_silhouette_score", best["silhouette_score"])
        mlflow.log_metric("selected_inertia", best["inertia"])

    return pd.DataFrame(candidate_rows)


def register_perfil_classifier(
    experiment_name="ford_segmento_classifier_experimental",
    model_name="ford_segmento_classifier_experimental",
):
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    df = _load_data().merge(pd.read_csv(SEGMENTOS_PATH), on="VIN_Hash", how="inner")
    if "segmento_pos_venda" not in df.columns:
        raise ValueError("Rode o clustering antes para gerar segmento_pos_venda")

    feature_cols = EXPERIMENT_FEATURES_NUMERIC + EXPERIMENT_FEATURES_CATEGORICAL
    x = df[feature_cols]
    y = df["segmento_pos_venda"]
    check_temporal_leakage(x, feature_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "LogReg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

    preprocessor = build_preprocessor_perfil()
    best_f1 = -1
    best_run = None

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        f1 = f1_score(y_test, y_pred, average="macro")

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model", name)
            mlflow.log_metric("f1_macro", f1)
            if f1 > best_f1:
                best_f1 = f1
                best_run = run
                mlflow.sklearn.log_model(pipeline, "model")

    if best_run:
        model_uri = f"runs:/{best_run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(model_name, mv.version, "Production", archive_existing_versions=True)

    return best_f1


def register_churn_classifier(experiment_name="ford_classificacao_churn", model_name="ford_churn_classifier"):
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    df = _load_data()
    feature_cols = SNAPSHOT_FEATURES_NUMERIC + SNAPSHOT_FEATURES_CATEGORICAL
    x = df[feature_cols]
    y = df[TARGET_CHURN]
    check_temporal_leakage(x, feature_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor_churn()
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", rf)])
    model = CalibratedClassifierCV(estimator=pipeline, cv=5, method="isotonic")

    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)

    with mlflow.start_run(run_name="RandomForest_Calibrated") as run:
        mlflow.log_param("model", "RandomForest_Calibrated")
        mlflow.log_metric("auc_roc", auc)
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(model_name, mv.version, "Production", archive_existing_versions=True)

    return auc


if __name__ == "__main__":
    register_kmeans_experiment()
    register_perfil_classifier()
    register_churn_classifier()
