import matplotlib
matplotlib.use("Agg")

import os
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

try:
    from src.pipeline.preprocessor import build_preprocessor
    from src.pipeline.train_classifier import LEAKAGE_COLS, check_leakage
except ModuleNotFoundError:
    base_dir = os.path.dirname(__file__)
    build_preprocessor = SourceFileLoader(
        "preprocessor",
        os.path.join(base_dir, "preprocessor.py"),
    ).load_module().build_preprocessor
    train_classifier = SourceFileLoader(
        "train_classifier",
        os.path.join(base_dir, "train_classifier.py"),
    ).load_module()
    LEAKAGE_COLS = train_classifier.LEAKAGE_COLS
    check_leakage = train_classifier.check_leakage


RANDOM_STATE = 42
LABELS = ["abandono", "economico", "esquecido", "fiel"]


def _add_plano_manutencao(df):
    if "plano_manutencao" in df.columns:
        return df

    required = {"share_revisoes_rede_24m", "qtde_revisoes_24m"}
    if not required.issubset(df.columns):
        return df

    df = df.copy()
    df["plano_manutencao"] = "fora_rede"
    df.loc[df["share_revisoes_rede_24m"] >= 0.70, "plano_manutencao"] = "com_plano"
    df.loc[
        (df["share_revisoes_rede_24m"] >= 0.30)
        & (df["share_revisoes_rede_24m"] < 0.70),
        "plano_manutencao",
    ] = "avulso_rede"
    df.loc[
        (df["share_revisoes_rede_24m"] >= 0.55)
        & (df["qtde_revisoes_24m"] >= 2),
        "plano_manutencao",
    ] = "com_plano"
    return df


def _split_columns(x):
    numeric_cols = []
    categorical_cols = []
    binary_cols = []

    for col in x.columns:
        unique_count = x[col].dropna().nunique()
        if unique_count <= 2:
            binary_cols.append(col)
        elif x[col].dtype == "object" or str(x[col].dtype) == "category":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return numeric_cols, categorical_cols, binary_cols


def _load_data(input_path, target_col):
    df = pd.read_csv(input_path)
    df = _add_plano_manutencao(df)

    drop_cols = ["id_cliente", "cliente_id", "modelo_veiculo", target_col]
    drop_cols += [col for col in LEAKAGE_COLS if col != target_col]

    y = df[target_col]
    x = df.drop(columns=[col for col in drop_cols if col in df.columns])
    check_leakage(x)
    return x, y


def _feature_names(preprocessor):
    names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[-1] for name in names]


def _make_pipeline(model, x):
    numeric_cols, categorical_cols, binary_cols = _split_columns(x)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, binary_cols)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def plot_decision_tree(pipeline, output_path="reports/decision_tree"):
    os.makedirs("reports", exist_ok=True)
    features = _feature_names(pipeline.named_steps["preprocessor"])
    model = pipeline.named_steps["model"]

    plt.figure(figsize=(28, 14))
    plot_tree(
        model,
        feature_names=features,
        class_names=[str(label) for label in model.classes_],
        max_depth=4,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipeline, output_path="reports/feature_importance"):
    os.makedirs("reports", exist_ok=True)
    model = pipeline.named_steps["model"]
    features = _feature_names(pipeline.named_steps["preprocessor"])
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top15 = importance.head(15).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top15["feature"], top15["importance"], color="#003478")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    importance.to_csv(f"{output_path}.csv", index=False)
    top3 = importance.head(3)["feature"].tolist()
    if not any(feature.startswith("plano_manutencao") for feature in top3):
        print(f"AVISO: plano_manutencao nao ficou no top 3: {top3}")

    return importance


def plot_confusion_matrix(
    pipeline,
    x_test,
    y_test,
    output_path="reports/confusion_matrix_rf",
):
    os.makedirs("reports", exist_ok=True)
    y_pred = pipeline.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    display.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(
    input_path="data/raw/ford_clientes_historico_completo.csv",
    target_col="perfil_latente",
):
    x, y = _load_data(input_path, target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    tree = _make_pipeline(
        DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        x,
    )
    tree.fit(x_train, y_train)
    plot_decision_tree(tree)

    rf = _make_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        x,
    )
    rf.fit(x_train, y_train)
    importance = plot_feature_importance(rf)
    plot_confusion_matrix(rf, x_test, y_test)

    print("Criado reports/decision_tree")
    print("Criado reports/feature_importance")
    print("Criado reports/feature_importance.csv")
    print("Criado reports/confusion_matrix_rf")
    print("Top 3 features:")
    print(importance.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
