import matplotlib
matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import LEAKAGE_COLUMNS


RANDOM_STATE = 42
K_CANDIDATES = [3, 4, 5, 6]
MIN_SILHOUETTE = 0.25
MIN_ACCEPTANCE_SILHOUETTE = 0.27
MIN_ACCEPTANCE_ARI = 0.25

FEATURE_COLS = [col for col in LEAKAGE_COLUMNS if col != "churn_rede_24m"]

PERFIS = ["fiel", "economico", "abandono", "esquecido"]


def _make_pipeline(n_clusters):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20, max_iter=500)),
        ]
    )


def _evaluate_k_candidates(x, row_count):
    rows = []
    best = None

    for k in K_CANDIDATES:
        pipeline = _make_pipeline(k)
        labels = pipeline.fit_predict(x)
        x_scaled = pipeline[:-1].transform(x)
        silhouette = silhouette_score(
            x_scaled,
            labels,
            sample_size=min(10000, row_count),
            random_state=RANDOM_STATE,
        )
        inertia = pipeline.named_steps["kmeans"].inertia_

        row = {
            "k": k,
            "silhouette_score": silhouette,
            "inertia": inertia,
            "pipeline": pipeline,
            "labels": labels,
        }
        rows.append(row)
        print(f"k={k} silhouette_score={silhouette:.4f} inertia={inertia:.2f}", flush=True)

        if best is None or silhouette > best["silhouette_score"]:
            best = row

    if best["silhouette_score"] <= MIN_SILHOUETTE:
        print(
            "ALERTA: silhouette maximo abaixo do minimo "
            f"({best['silhouette_score']:.4f} <= {MIN_SILHOUETTE:.2f}); abortando clustering",
            flush=True,
        )
        raise ValueError("Silhouette insuficiente para gerar labels de clustering")

    print(
        "k escolhido pelo maior silhouette: "
        f"{best['k']} (silhouette={best['silhouette_score']:.4f})",
        flush=True,
    )
    return rows, best


def _log_selected_k_mlflow(best, ari=None):
    active_run = mlflow.active_run()

    def log_metrics():
        mlflow.log_param("selected_k", best["k"])
        mlflow.log_metric("selected_silhouette_score", best["silhouette_score"])
        mlflow.log_metric("selected_inertia", best["inertia"])
        if ari is not None:
            mlflow.log_metric("ari_perfil_latente", ari)

    if active_run:
        log_metrics()
    else:
        mlflow.set_experiment("ford_segmentacao_kmeans")
        with mlflow.start_run(run_name="kmeans_selected_k"):
            log_metrics()


def _map_clusters_to_profiles(df, labels):
    if "perfil_latente" not in df.columns:
        return {cluster: PERFIS[i % len(PERFIS)] for i, cluster in enumerate(sorted(set(labels)))}

    tab = pd.crosstab(pd.Series(labels, name="cluster"), df["perfil_latente"])
    mapping = {}
    used = set()

    for cluster in tab.max(axis=1).sort_values(ascending=False).index:
        ranked_perfis = [perfil for perfil in tab.loc[cluster].sort_values(ascending=False).index if perfil in PERFIS]
        for perfil in tab.loc[cluster].sort_values(ascending=False).index:
            if perfil in PERFIS and perfil not in used:
                mapping[cluster] = perfil
                used.add(perfil)
                break
        if cluster not in mapping and ranked_perfis:
            mapping[cluster] = ranked_perfis[0]

    remaining = [perfil for perfil in PERFIS if perfil not in used]
    for cluster in sorted(set(labels)):
        if cluster not in mapping:
            mapping[cluster] = remaining.pop(0) if remaining else PERFIS[0]

    return mapping


def plot_pca_clusters(df, labels, pipeline, output_path="reports/clusters_pca.png"):
    os.makedirs("reports", exist_ok=True)

    x_scaled = pipeline[:-1].transform(df[FEATURE_COLS])
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(x_scaled)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        components[:, 0],
        components[:, 1],
        c=labels,
        s=4,
        alpha=0.35,
        cmap="tab10",
        rasterized=True,
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Clusters K-Means em PCA 2D")
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    plt.close()


def run_clustering(input_path="data/raw/ford_clientes_historico_completo.csv"):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(input_path)
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para clustering: {missing}")

    x = df[FEATURE_COLS]

    candidate_rows, best = _evaluate_k_candidates(x, len(df))
    ks = [row["k"] for row in candidate_rows]
    inertias = [row["inertia"] for row in candidate_rows]
    silhouettes = [row["silhouette_score"] for row in candidate_rows]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow")

    plt.subplot(1, 2, 2)
    plt.plot(list(ks), silhouettes, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("Silhouette")
    plt.savefig("reports/elbow_silhouette.png", format="png", dpi=150, bbox_inches="tight")
    plt.close()

    pipeline = best["pipeline"]
    labels = best["labels"]
    selected_silhouette = best["silhouette_score"]

    ari = None
    if "perfil_latente" in df.columns:
        ari = adjusted_rand_score(df["perfil_latente"], labels)
        print(f"ARI vs perfil_latente: {ari:.4f}")
        if ari <= MIN_ACCEPTANCE_ARI or selected_silhouette <= MIN_ACCEPTANCE_SILHOUETTE:
            print(
                "ALERTA: criterio minimo de aceite nao atendido "
                f"(ARI={ari:.4f}, silhouette={selected_silhouette:.4f}); abortando clustering",
                flush=True,
            )
            raise ValueError("ARI/Silhouette insuficientes para aceitar labels de clustering")
    else:
        print("ARI vs perfil_latente: coluna perfil_latente ausente")

    _log_selected_k_mlflow(best, ari)

    mapping = _map_clusters_to_profiles(df, labels)
    labels_df = pd.DataFrame(
        {
            "cliente_id": df["id_cliente"] if "id_cliente" in df.columns else df["cliente_id"],
            "perfil_cluster": pd.Series(labels).map(mapping),
        }
    )
    labels_df.to_csv("data/processed/cluster_labels.csv", index=False)

    plot_pca_clusters(df, labels, pipeline)
    print("Labels salvos em data/processed/cluster_labels")
    print("Elbow/Silhouette salvo em reports/elbow_silhouette")
    print("PCA 2D salvo em reports/clusters_pca")

    return labels_df, pipeline


if __name__ == "__main__":
    run_clustering()
