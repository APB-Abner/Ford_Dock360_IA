import matplotlib
matplotlib.use("Agg")

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.pipeline.config import LEAKAGE_COLUMNS

RANDOM_STATE = 42
TEST_SIZE = 0.20

SENSITIVE_FEATURES = ["genero", "renda_mensal", "score_credito"]

# Acceptable max gap in F1-macro between subgroups
MAX_F1_GAP_GENDER = 0.05
MAX_F1_GAP_INCOME = 0.10
MAX_F1_GAP_SCORE = 0.10


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


def _get_test_data(
    features_path="data/raw/ford_clientes_operacional_compra.csv",
    target_path="data/raw/ford_clientes_historico_completo.csv",
):
    features = _normalize_cliente_id(_load_csv(features_path))
    targets = _normalize_cliente_id(_load_csv(target_path))

    targets = targets[["cliente_id", "churn_rede_24m"]]
    df = features.merge(targets, on="cliente_id", how="inner")

    y = df["churn_rede_24m"].astype(int)
    x = df.drop(columns=["cliente_id", "churn_rede_24m"])

    # Replicate the exact split used during training
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return x_test.reset_index(drop=True), y_test.reset_index(drop=True)


def _f1_by_categorical(model, x_test, y_test, col):
    results = []
    for group in sorted(x_test[col].dropna().unique()):
        mask = x_test[col] == group
        if mask.sum() < 30:
            continue
        y_pred = model.predict(x_test[mask])
        f1 = f1_score(y_test[mask], y_pred, average="macro", zero_division=0)
        results.append({"subgrupo": str(group), "n": int(mask.sum()), "f1_macro": round(f1, 4), "feature": col})
    return results


def _f1_by_numeric_quartile(model, x_test, y_test, col):
    quartile_labels = ["Q1 (baixo)", "Q2 (medio-baixo)", "Q3 (medio-alto)", "Q4 (alto)"]
    bins = x_test[col].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    groups = pd.cut(x_test[col], bins=bins, labels=quartile_labels[: len(np.unique(bins)) - 1], duplicates="drop")
    results = []
    for group in groups.cat.categories:
        mask = groups == group
        if mask.sum() < 30:
            continue
        y_pred = model.predict(x_test[mask])
        f1 = f1_score(y_test[mask], y_pred, average="macro", zero_division=0)
        results.append({"subgrupo": str(group), "n": int(mask.sum()), "f1_macro": round(f1, 4), "feature": col})
    return results


def _plot_subgroup_f1(results_df, title, output_path, max_gap):
    mean_f1 = results_df["f1_macro"].mean()
    colors = ["#d62728" if abs(f - mean_f1) > max_gap / 2 else "#2ca02c" for f in results_df["f1_macro"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(results_df["subgrupo"], results_df["f1_macro"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Subgrupo")
    ax.set_ylabel("F1-Macro")
    ax.set_title(title)
    ax.axhline(mean_f1, linestyle="--", color="gray", label=f"Media ({mean_f1:.3f})")
    for bar, val in zip(bars, results_df["f1_macro"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _check_gap(results_df, feature_name, max_gap):
    if results_df.empty:
        return
    gap = results_df["f1_macro"].max() - results_df["f1_macro"].min()
    status = "OK" if gap <= max_gap else "ATENCAO"
    print(f"  [{status}] Variacao de F1-Macro entre subgrupos de {feature_name}: {gap:.4f} (limite: {max_gap})")


def run_bias_analysis(model_path="models/churn_rf_calibrated.joblib"):
    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    model = joblib.load(model_path)
    x_test, y_test = _get_test_data()

    all_results = []

    # --- Genero ---
    genero_results = _f1_by_categorical(model, x_test, y_test, "genero")
    genero_df = pd.DataFrame(genero_results)
    all_results.extend(genero_results)
    _plot_subgroup_f1(genero_df, "F1-Macro por Genero", "reports/bias_genero.png", MAX_F1_GAP_GENDER)
    print("\n=== Analise de Vies: Genero ===")
    print(genero_df.to_string(index=False))
    _check_gap(genero_df, "genero", MAX_F1_GAP_GENDER)

    # --- Renda Mensal ---
    renda_results = _f1_by_numeric_quartile(model, x_test, y_test, "renda_mensal")
    renda_df = pd.DataFrame(renda_results)
    all_results.extend(renda_results)
    _plot_subgroup_f1(renda_df, "F1-Macro por Faixa de Renda", "reports/bias_renda.png", MAX_F1_GAP_INCOME)
    print("\n=== Analise de Vies: Renda Mensal ===")
    print(renda_df.to_string(index=False))
    _check_gap(renda_df, "renda_mensal", MAX_F1_GAP_INCOME)

    # --- Score de Credito ---
    score_results = _f1_by_numeric_quartile(model, x_test, y_test, "score_credito")
    score_df = pd.DataFrame(score_results)
    all_results.extend(score_results)
    _plot_subgroup_f1(score_df, "F1-Macro por Score de Credito", "reports/bias_score_credito.png", MAX_F1_GAP_SCORE)
    print("\n=== Analise de Vies: Score de Credito ===")
    print(score_df.to_string(index=False))
    _check_gap(score_df, "score_credito", MAX_F1_GAP_SCORE)

    # --- Save consolidated results ---
    all_df = pd.DataFrame(all_results)
    all_df.to_csv("reports/bias_analysis.csv", index=False)
    print("\nResultados salvos em reports/bias_analysis.csv")
    print("Graficos: reports/bias_genero.png | reports/bias_renda.png | reports/bias_score_credito.png")

    return all_df


if __name__ == "__main__":
    run_bias_analysis()
