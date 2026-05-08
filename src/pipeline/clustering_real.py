"""
Clustering K-Means — Caminho B (dataset real Ford)

Substitui clustering.py original. Agrupa VINs em 4 perfis comportamentais
usando features agregadas de servicos reais.

Features de clustering (todas comportamentais — sem leakage para classificacao):
  - qtde_revisoes
  - meses_desde_ultimo_servico
  - meses_relacionamento
  - n_dealers_usados
  - km_max
  - pct_agenda
  - intervalo_medio_revisoes_dias

Saidas:
  data/processed/cluster_labels.csv — cliente_id (VIN_Hash) + perfil_cluster
  reports/elbow_silhouette.png
  reports/clusters_pca.png
"""

import matplotlib
matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import RANDOM_STATE


INPUT_PATH = "data/processed/vins_agregados.csv"
OUTPUT_LABELS = "data/processed/cluster_labels.csv"

CLUSTER_FEATURES = [
    "qtde_revisoes",
    "meses_desde_ultimo_servico",
    "meses_relacionamento",
    "n_dealers_usados",
    "km_max",
    "pct_agenda",
    "intervalo_medio_revisoes_dias",
]

PERFIS = ["fiel", "economico", "abandono", "esquecido"]


def _make_pipeline(n_clusters):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)),
    ])


def _interpretar_clusters(df, labels):
    """
    Mapeia cluster -> nome de negocio com base nos centroides.
    Heuristica baseada nas features comportamentais:
      - fiel: muitas revisoes, baixo tempo desde ultimo servico, alta agenda
      - abandono: poucas revisoes, alto tempo desde ultimo servico
      - economico: muitos dealers diferentes (shopping around)
      - esquecido: longos intervalos entre revisoes
    """
    df_c = df[CLUSTER_FEATURES].copy()
    df_c["cluster"] = labels
    centroides = df_c.groupby("cluster").mean()

    print("\n=== Centroides dos clusters (medias) ===")
    print(centroides.round(2))

    mapping = {}
    used = set()

    # 1. Abandono: maior meses_desde_ultimo_servico
    cluster_abandono = centroides["meses_desde_ultimo_servico"].idxmax()
    mapping[cluster_abandono] = "abandono"
    used.add("abandono")

    # 2. Fiel: maior qtde_revisoes (entre os restantes)
    restantes = centroides.drop(cluster_abandono)
    cluster_fiel = restantes["qtde_revisoes"].idxmax()
    mapping[cluster_fiel] = "fiel"
    used.add("fiel")

    # 3. Economico: mais dealers diferentes (entre os restantes)
    restantes = restantes.drop(cluster_fiel)
    cluster_economico = restantes["n_dealers_usados"].idxmax()
    mapping[cluster_economico] = "economico"
    used.add("economico")

    # 4. Esquecido: o que sobrou
    cluster_esquecido = restantes.drop(cluster_economico).index[0]
    mapping[cluster_esquecido] = "esquecido"

    print(f"\n=== Mapeamento cluster -> perfil ===")
    for cluster, perfil in sorted(mapping.items()):
        print(f"  Cluster {cluster}: {perfil}")

    return mapping


def _plot_elbow_silhouette(x):
    inertias = []
    silhouettes = []
    ks = range(2, 9)

    for k in ks:
        pipeline = _make_pipeline(k)
        labels_k = pipeline.fit_predict(x)
        x_scaled = pipeline[:-1].transform(x)
        inertias.append(pipeline.named_steps["kmeans"].inertia_)
        # Sample para silhouette nao demorar
        sample_size = min(10000, len(x))
        sil = silhouette_score(x_scaled, labels_k, sample_size=sample_size, random_state=RANDOM_STATE)
        silhouettes.append(sil)
        print(f"  k={k}: inertia={inertias[-1]:.0f}, silhouette={sil:.4f}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(list(ks), inertias, marker="o", color="#003478")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(list(ks), silhouettes, marker="o", color="#003478")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette por k")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Salvo: reports/elbow_silhouette.png")


def _plot_pca_clusters(x, labels, pipeline, mapping):
    x_scaled = pipeline[:-1].transform(x)

    # Sample para visualizacao
    sample_size = min(50000, len(x_scaled))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(x_scaled), sample_size, replace=False)
    x_sample = x_scaled[sample_idx]
    labels_sample = labels[sample_idx]

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(x_sample)

    cores = {"fiel": "#003478", "economico": "#FF7F0E", "abandono": "#D62728", "esquecido": "#6FA8C9"}

    plt.figure(figsize=(10, 7))
    for cluster_id, perfil in mapping.items():
        idx = labels_sample == cluster_id
        plt.scatter(
            components[idx, 0],
            components[idx, 1],
            c=cores.get(perfil, "gray"),
            label=perfil,
            alpha=0.4,
            s=8,
        )
    plt.legend(title="Perfil")
    plt.title("Clusters de Comportamento (PCA 2D)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variancia)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variancia)")
    plt.tight_layout()
    plt.savefig("reports/clusters_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Salvo: reports/clusters_pca.png")


def run_clustering(input_path=INPUT_PATH, n_clusters=4):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {input_path}\n"
            f"Rode antes: python -m src.pipeline.feature_engineering_real"
        )

    df = pd.read_csv(input_path)
    print(f"Carregado: {df.shape[0]:,} VINs")

    missing = [col for col in CLUSTER_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas de clustering ausentes: {missing}")

    x = df[CLUSTER_FEATURES]

    print("\n=== Selecao de k via Elbow + Silhouette ===")
    _plot_elbow_silhouette(x)

    print(f"\n=== Treinando K-Means com k={n_clusters} ===")
    pipeline = _make_pipeline(n_clusters)
    labels = pipeline.fit_predict(x)

    mapping = _interpretar_clusters(df, labels)

    df_labels = pd.DataFrame({
        "cliente_id": df["VIN_Hash"],
        "cluster_raw": labels,
        "perfil_cluster": pd.Series(labels).map(mapping),
    })
    df_labels.to_csv(OUTPUT_LABELS, index=False)
    print(f"\nSalvo: {OUTPUT_LABELS}")

    print(f"\n=== Distribuicao final ===")
    print(df_labels["perfil_cluster"].value_counts(normalize=True).round(3))

    _plot_pca_clusters(x, labels, pipeline, mapping)

    return df_labels, pipeline


if __name__ == "__main__":
    run_clustering()
