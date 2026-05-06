import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipeline.config import LEAKAGE_COLUMNS
from src.pipeline.clustering import (
    MIN_ACCEPTANCE_ARI,
    MIN_ACCEPTANCE_SILHOUETTE,
    _evaluate_k_candidates,
)
from src.pipeline.preprocessor import build_preprocessor
from src.pipeline.train_classifier import (
    assert_metrics_not_suspicious,
    audit_features_used,
    check_leakage,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
TRACKING_URI = "file:./mlruns"

SEGMENTATION_FEATURE_COLS = [col for col in LEAKAGE_COLUMNS if col != "churn_rede_24m"]


def setup_mlflow(tracking_uri=TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


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


def _load_perfil_data(input_path, target_col):
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Coluna target ausente: {target_col}")

    drop_cols = ["id_cliente", "cliente_id", "modelo_veiculo", target_col]
    drop_cols += [col for col in LEAKAGE_COLUMNS if col != target_col]

    y = df[target_col]
    x = df.drop(columns=[col for col in drop_cols if col in df.columns])
    check_leakage(x)
    return x, y


def _load_churn_data(features_path, target_path):
    features = pd.read_csv(features_path)
    targets = pd.read_csv(target_path)

    if "id_cliente" in features.columns:
        features = features.rename(columns={"id_cliente": "cliente_id"})
    if "id_cliente" in targets.columns:
        targets = targets.rename(columns={"id_cliente": "cliente_id"})
    if "cliente_id" not in features.columns or "cliente_id" not in targets.columns:
        raise ValueError("Base sem coluna cliente_id ou id_cliente")
    if "churn_rede_24m" not in targets.columns:
        raise ValueError("Coluna target ausente: churn_rede_24m")

    df = features.merge(targets[["cliente_id", "churn_rede_24m"]], on="cliente_id", how="inner")
    if df.empty:
        raise ValueError("Join por cliente_id nao retornou linhas")

    y = df["churn_rede_24m"].astype(int)
    x = df.drop(columns=["cliente_id", "churn_rede_24m"])
    check_leakage(x)
    return x, y


def _make_classifier_pipeline(x, model):
    numeric_cols, categorical_cols, binary_cols = _split_columns(x)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _promote_latest_version(client, model_name, stage="Production"):
    latest = client.get_latest_versions(model_name)
    if not latest:
        raise ValueError(f"Modelo registrado sem versoes: {model_name}")

    version = max(latest, key=lambda item: int(item.version)).version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    return version


def register_kmeans_experiment(
    input_path="data/raw/ford_clientes_historico_completo.csv",
    experiment_name="ford_segmentacao_kmeans",
):
    setup_mlflow()
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(input_path)
    missing = [col for col in SEGMENTATION_FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para clustering: {missing}")

    x = df[SEGMENTATION_FEATURE_COLS]
    candidate_rows, best = _evaluate_k_candidates(x, len(df))

    rows = []
    for row in candidate_rows:
        with mlflow.start_run(run_name=f"kmeans_k_{row['k']}"):
            mlflow.log_param("k", row["k"])
            mlflow.log_metric("silhouette_score", row["silhouette_score"])
            mlflow.log_metric("inertia", row["inertia"])

        rows.append(
            {
                "k": row["k"],
                "silhouette_score": row["silhouette_score"],
                "inertia": row["inertia"],
            }
        )

    ari = None
    if "perfil_latente" in df.columns:
        ari = adjusted_rand_score(df["perfil_latente"], best["labels"])
        print(f"ARI vs perfil_latente: {ari:.4f}", flush=True)
        if ari <= MIN_ACCEPTANCE_ARI or best["silhouette_score"] <= MIN_ACCEPTANCE_SILHOUETTE:
            print(
                "ALERTA: criterio minimo de aceite nao atendido "
                f"(ARI={ari:.4f}, silhouette={best['silhouette_score']:.4f}); abortando clustering",
                flush=True,
            )
            raise ValueError("ARI/Silhouette insuficientes para aceitar labels de clustering")

    with mlflow.start_run(run_name="kmeans_selected_k"):
        mlflow.log_param("selected_k", best["k"])
        mlflow.log_metric("selected_silhouette_score", best["silhouette_score"])
        mlflow.log_metric("selected_inertia", best["inertia"])
        if ari is not None:
            mlflow.log_metric("ari_perfil_latente", ari)

    return pd.DataFrame(rows)


def register_perfil_classifier(
    input_path="data/raw/ford_clientes_historico_completo.csv",
    target_col="perfil_latente",
    experiment_name="ford_classificacao_perfil",
    model_name="ford_perfil_classifier",
):
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    x, y = _load_perfil_data(input_path, target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

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
        # Hiperparametros reduzidos intencionalmente para velocidade de tracking.
        # Modelo de producao usa n_estimators=200 (ver train_classifier.py)
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    best = None

    for run_name, estimator in models.items():
        pipeline = _make_classifier_pipeline(x_train, estimator)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        f1_macro = f1_score(y_test, predictions, average="macro")
        assert_metrics_not_suspicious(f1_macro=f1_macro)
        audit_features_used(x_train.columns)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("model", run_name)
            mlflow.log_metric("f1_macro_test", f1_macro)

        rows.append({"model": run_name, "f1_macro_test": f1_macro})
        if best is None or f1_macro > best["f1_macro_test"]:
            best = {
                "name": run_name,
                "pipeline": pipeline,
                "f1_macro_test": f1_macro,
                "run_id": run.info.run_id,
            }

        print(f"{run_name} f1_macro_test={f1_macro:.4f}", flush=True)

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.sklearn.log_model(
            best["pipeline"],
            artifact_path="model",
            registered_model_name=model_name,
        )

    time.sleep(1)
    version = _promote_latest_version(client, model_name)
    print(f"{model_name} versao {version} promovido para Production")
    return pd.DataFrame(rows), version


def register_churn_classifier(
    features_path="data/raw/ford_clientes_operacional_compra.csv",
    target_path="data/raw/ford_clientes_historico_completo.csv",
    experiment_name="ford_classificacao_churn",
    model_name="ford_churn_classifier",
):
    client = setup_mlflow()
    mlflow.set_experiment(experiment_name)

    x, y = _load_churn_data(features_path, target_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = _make_classifier_pipeline(x_train, rf)
    model = CalibratedClassifierCV(estimator=pipeline, cv=3, method="isotonic")

    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    assert_metrics_not_suspicious(auc=auc)
    audit_features_used(x_train.columns)

    with mlflow.start_run(run_name="RandomForest_calibrated"):
        mlflow.log_param("model", "RandomForest_calibrated")
        mlflow.log_metric("auc_roc_test", auc)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )

    time.sleep(1)
    version = _promote_latest_version(client, model_name)
    print(f"RandomForest_calibrated auc_roc_test={auc:.4f}")
    print(f"{model_name} versao {version} promovido para Production")
    return auc, version


def register_all_experiments():
    kmeans = register_kmeans_experiment()
    perfil, perfil_version = register_perfil_classifier()
    churn_auc, churn_version = register_churn_classifier()
    return {
        "kmeans": kmeans,
        "perfil_version": perfil_version,
        "perfil_runs": perfil,
        "churn_auc": churn_auc,
        "churn_version": churn_version,
    }


if __name__ == "__main__":
    register_all_experiments()
