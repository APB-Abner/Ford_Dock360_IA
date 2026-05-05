from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols, categorical_cols, binary_cols):
    numeric_cols = [col for col in numeric_cols if col != "modelo_veiculo"]
    categorical_cols = [col for col in categorical_cols if col != "modelo_veiculo"]
    binary_cols = [col for col in binary_cols if col != "modelo_veiculo"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("binary", "passthrough", binary_cols),
        ],
        remainder="drop",
    )
