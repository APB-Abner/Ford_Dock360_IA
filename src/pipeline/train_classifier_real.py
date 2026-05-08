"""
Classificador de Perfil — Caminho B (dataset real Ford)

Substitui train_classifier.py original. Treina classificador multiclasse para
prever o perfil_cluster usando APENAS features disponiveis no momento da venda.

Features de compra disponiveis no dataset real:
  - modelo (categorico)
  - ano_modelo (numerico)
  - dealer_code (categorico de alta cardinalidade)
  - dias_ate_entrega (numerico)
  - idade_veiculo_meses (numerico - calculado em relacao a data ref)

Modelos comparados:
  1. Logistic Regression (baseline)
  2. Decision Tree max_depth=4 (explicavel)
  3. Random Forest (producao)

Anti-leakage: features comportamentais (qtde_revisoes, meses_*, n_dealers,
km_max, pct_agenda) sao PROIBIDAS aqui pois nao existem no momento da compra.

Saidas:
  models/perfil_rf_classifier.joblib
  reports/model_comparison.csv
  reports/decision_tree.png
  reports/feature_importance.png
  reports/confusion_matrix_rf.png
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.pipeline.config import LEAKAGE_BEHAVIORAL, RANDOM_STATE, TEST_SIZE


VINS_PATH = "data/processed/vins_agregados.csv"
LABELS_PATH = "data/processed/cluster_labels.csv"

# Features disponiveis no MOMENTO DA COMPRA (sem leakage)
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
        raise FileNotFoundError(f"Rode antes: python -m src.pipeline.feature_engineering_real")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Rode antes: python -m src.pipeline.clustering_real")

    vins = pd.read_csv(VINS_PATH)
    labels = pd.read_csv(LABELS_PATH)

    df = vins.merge(labels[["cliente_id", "perfil_cluster"]],
                    left_on="VIN_Hash", right_on="cliente_id", how="inner")

    feature_cols = PURCHASE_FEATURES_NUMERIC + PURCHASE_FEATURES_CATEGORICAL
    x = df[feature_cols].copy()
    y = df["perfil_cluster"].copy()

    # Filtrar linhas sem perfil
    mask = y.notna()
    x, y = x[mask], y[mask]

    check_leakage(x)
    print(f"Dataset: {len(x):,} VINs | Features: {feature_cols}")
    print(f"Distribuicao de y:\n{y.value_counts(normalize=True).round(3)}")
    return x, y


def assert_metrics_not_suspicious(f1_macro):
    if f1_macro > 0.92:
        raise ValueError(
            f"F1 Macro = {f1_macro:.4f} acima de 0.92 — suspeita de leakage. "
            f"Auditar features do X de treino."
        )


def train_all_models():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    x, y = _load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = _build_preprocessor()

    models = {
        "LogReg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_features="sqrt", min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    pipelines = {}

    for name, model in models.items():
        print(f"\n=== Treinando {name} ===")
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Cross-validation no treino
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

        # Fit final + avaliacao no teste
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        f1_test = f1_score(y_test, y_pred, average="macro")

        rows.append({
            "model": name,
            "f1_macro_cv_mean": cv_scores.mean(),
            "f1_macro_cv_std": cv_scores.std(),
            "f1_macro_test": f1_test,
        })
        pipelines[name] = pipeline

        print(f"  CV F1 Macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test F1 Macro: {f1_test:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")

    comparison = pd.DataFrame(rows).sort_values("f1_macro_test", ascending=False)
    comparison.to_csv("reports/model_comparison.csv", index=False)
    print("\n=== Comparativo final ===")
    print(comparison.to_string(index=False))

    # Anti-leakage check
    best_f1 = comparison["f1_macro_test"].max()
    assert_metrics_not_suspicious(best_f1)

    # Salvar Random Forest (modelo de producao)
    rf_pipeline = pipelines["RandomForest"]
    joblib.dump(rf_pipeline, "models/perfil_rf_classifier.joblib", compress=3)
    print("\nSalvo: models/perfil_rf_classifier.joblib")

    # Visualizacoes
    _plot_decision_tree(pipelines["DecisionTree"])
    _plot_feature_importance(rf_pipeline)
    _plot_confusion_matrix(rf_pipeline, x_test, y_test)

    return comparison


def _get_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    return [n.split("__", 1)[-1] for n in names]


def _plot_decision_tree(pipeline):
    tree = pipeline.named_steps["model"]
    feature_names = _get_feature_names(pipeline)

    plt.figure(figsize=(28, 14))
    plot_tree(
        tree, feature_names=feature_names,
        class_names=[str(c) for c in tree.classes_],
        max_depth=4, filled=True, rounded=True, fontsize=8,
    )
    plt.tight_layout()
    plt.savefig("reports/decision_tree.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Salvo: reports/decision_tree.png")


def _plot_feature_importance(pipeline):
    rf = pipeline.named_steps["model"]
    feature_names = _get_feature_names(pipeline)
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv("reports/feature_importance.csv", index=False)

    top15 = importance.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top15["feature"], top15["importance"], color="#003478")
    plt.xlabel("Importancia")
    plt.title("Top 15 Features — Random Forest (Perfil)")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Salvo: reports/feature_importance.png")


def _plot_confusion_matrix(pipeline, x_test, y_test):
    y_pred = pipeline.predict(x_test)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de Confusao — Random Forest (Perfil)")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Salvo: reports/confusion_matrix_rf.png")


if __name__ == "__main__":
    train_all_models()
