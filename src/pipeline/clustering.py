import matplotlib; matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

FEATURE_COLS = [
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

PERFIS = ["fiel", "economico", "abandono", "esquecido"]


def _make_pipeline(n_clusters):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)),
        ]
    )


def _map_clusters_to_profiles(df, labels):
    if "perfil_latente" not in df.columns:
        return {cluster: PERFIS[i] for i, cluster in enumerate(sorted(set(labels)))}

    tab = pd.crosstab(pd.Series(labels, name="cluster"), df["perfil_latente"])
    mapping = {}
    used = set()

    for cluster in tab.max(axis=1).sort_values(ascending=False).index:
        for perfil in tab.loc[cluster].sort_values(ascending=False).index:
            if perfil in PERFIS and perfil not in used:
                mapping[cluster] = perfil
                used.add(perfil)
                break

    remaining = [perfil for perfil in PERFIS if perfil not in used]
    for cluster in sorted(set(labels)):
        if cluster not in mapping:
            mapping[cluster] = remaining.pop(0)

    return mapping


def plot_pca_clusters(df, labels, pipeline, output_path="reports/clusters_pca"):
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

    inertias = []
    silhouettes = []
    ks = range(2, 9)
    for k in ks:
        model = _make_pipeline(k)
        k_labels = model.fit_predict(x)
        x_scaled = model[:-1].transform(x)
        inertias.append(model.named_steps["kmeans"].inertia_)
        silhouettes.append(
            silhouette_score(
                x_scaled,
                k_labels,
                sample_size=min(10000, len(df)),
                random_state=RANDOM_STATE,
            )
        )

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
    plt.savefig("reports/elbow_silhouette", format="png", dpi=150, bbox_inches="tight")
    plt.close()

    pipeline = _make_pipeline(4)
    labels = pipeline.fit_predict(x)

    if "perfil_latente" in df.columns:
        ari = adjusted_rand_score(df["perfil_latente"], labels)
        print(f"ARI vs perfil_latente: {ari:.4f}")
    else:
        print("ARI vs perfil_latente: coluna perfil_latente ausente")

    mapping = _map_clusters_to_profiles(df, labels)
    labels_df = pd.DataFrame(
        {
            "cliente_id": df["id_cliente"] if "id_cliente" in df.columns else df["cliente_id"],
            "perfil_cluster": pd.Series(labels).map(mapping),
        }
    )
    labels_df.to_csv("data/processed/cluster_labels", index=False)

    plot_pca_clusters(df, labels, pipeline)
    print("Labels salvos em data/processed/cluster_labels")
    print("Elbow/Silhouette salvo em reports/elbow_silhouette")
    print("PCA 2D salvo em reports/clusters_pca")

    return labels_df, pipeline


if __name__ == "__main__":
    run_clustering()
