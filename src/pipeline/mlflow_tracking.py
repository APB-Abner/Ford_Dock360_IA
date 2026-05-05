import os
import time
from importlib.machinery import SourceFileLoader

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from src.pipeline.preprocessor import build_preprocessor
    from src.pipeline.train_classifier import check_leakage
except ModuleNotFoundError:
    pipeline_dir = os.path.dirname(__file__)
    build_preprocessor = SourceFileLoader(
        "preprocessor",
        os.path.join(pipeline_dir, "preprocessor.py"),
    ).load_module().build_preprocessor
    check_leakage = SourceFileLoader(
        "train_classifier",
        os.path.join(pipeline_dir, "train_classifier.py"),
    ).load_module().check_leakage


RANDOM_STATE = 42
TEST_SIZE = 0.20
TRACKING_URI = "file:./mlruns"

SEGMENTATION_FEATURE_COLS = [
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
]


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


def _make_kmeans_pipeline(k):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, random_state=RANDOM_STATE)),
        ]
    )


def _load_perfil_data(input_path, target_col):
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Coluna target ausente: {target_col}")

    drop_cols = ["id_cliente", "cliente_id", "modelo_veiculo", target_col]
    drop_cols += [
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
    rows = []

    for k in range(2, 9):
        pipeline = _make_kmeans_pipeline(k)
        labels = pipeline.fit_predict(x)
        x_scaled = pipeline[:-1].transform(x)
        silhouette = silhouette_score(
            x_scaled,
            labels,
            sample_size=min(10000, len(df)),
            random_state=RANDOM_STATE,
        )
        inertia = pipeline.named_steps["kmeans"].inertia_

        with mlflow.start_run(run_name=f"kmeans_k_{k}"):
            mlflow.log_param("k", k)
            mlflow.log_metric("silhouette_score", silhouette)
            mlflow.log_metric("inertia", inertia)

        rows.append({"k": k, "silhouette_score": silhouette, "inertia": inertia})
        print(f"k={k} silhouette_score={silhouette:.4f} inertia={inertia:.2f}", flush=True)

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
