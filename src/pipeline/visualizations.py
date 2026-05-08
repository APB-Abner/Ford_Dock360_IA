"""Visualizacoes simples para o classificador experimental de segmento pos-venda."""

import matplotlib
matplotlib.use("Agg")

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.pipeline.config import (
    RANDOM_STATE,
    TEST_SIZE,
)
from src.pipeline.train_classifier_real import EXPERIMENT_FEATURES_CATEGORICAL, EXPERIMENT_FEATURES_NUMERIC
from src.pipeline.train_classifier_real import _build_preprocessor as build_preprocessor_perfil
from src.pipeline.train_churn_real import check_leakage

VINS_PATH = "data/processed/snapshots_pos_venda.csv"
SEGMENTOS_PATH = "data/processed/segmentos_pos_venda.csv"


def _load_data(input_path, target_col):
    df = pd.read_csv(input_path).merge(pd.read_csv(SEGMENTOS_PATH), on="VIN_Hash", how="inner")
    feature_cols = EXPERIMENT_FEATURES_NUMERIC + EXPERIMENT_FEATURES_CATEGORICAL
    x = df[feature_cols].copy()
    y = df[target_col]
    check_leakage(x)
    return x, y


def _feature_names(preprocessor):
    names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[-1] for name in names]


def plot_decision_tree(pipeline, output_path="reports/decision_tree.png"):
    os.makedirs("reports", exist_ok=True)
    features = _feature_names(pipeline.named_steps["preprocessor"])
    model = pipeline.named_steps["model"]

    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=features,
        class_names=[str(c) for c in model.classes_],
        max_depth=3,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipeline, output_path="reports/feature_importance_segmento_experimental.png"):
    os.makedirs("reports", exist_ok=True)
    model = pipeline.named_steps["model"]
    features = _feature_names(pipeline.named_steps["preprocessor"])
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"].iloc[::-1], importance["importance"].iloc[::-1], color="#003478")
    plt.xlabel("Importancia")
    plt.title("Feature Importance - Segmento Experimental")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    importance.to_csv(output_path.replace(".png", ".csv"), index=False)
    return importance


def plot_confusion_matrix(pipeline, x_test, y_test, output_path="reports/confusion_matrix_segmento_experimental.png"):
    os.makedirs("reports", exist_ok=True)
    y_pred = pipeline.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de Confusao - Segmento Experimental")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(input_path=VINS_PATH, target_col="segmento_pos_venda"):
    if not os.path.exists(input_path):
        print(f"Arquivo nao encontrado: {input_path}")
        return

    x, y = _load_data(input_path, target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor_perfil()

    # Arvore de Decisao para visualizacao
    tree_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    tree_pipe.fit(x_train, y_train)
    plot_decision_tree(tree_pipe)

    # RandomForest para importancia e matriz
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    ])
    rf_pipe.fit(x_train, y_train)
    plot_feature_importance(rf_pipe)
    plot_confusion_matrix(rf_pipe, x_test, y_test)

    print("Visualizacoes geradas em reports/")


if __name__ == "__main__":
    main()
