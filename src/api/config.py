from pydantic_settings import BaseSettings, SettingsConfigDict


MIN_SECRET_KEY_LENGTH = 32
EXAMPLE_SECRET_KEY = "changeme-local-only"


class Settings(BaseSettings):
    SECRET_KEY: str
    JWT_ISSUER: str = "ford-vinguard-api"
    JWT_AUDIENCE: str = "ford-vinguard-api"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    DEMO_TOKEN_SECRET: str | None = None
    DEMO_TOKEN_SUBJECT: str = "demo-consultor"

    # Paths/names expected by the API runtime. Relative paths use the repo root.
    DATA_PATH: str = "data/raw/ford_complaints_top3_por_modelo.csv"
    MODELS_DIR: str = "models"
    CHURN_MODEL_FILENAME: str = "churn_pos_venda_rf_calibrated.joblib"
    PERFIL_MODEL_FILENAME: str = "segmento_pos_venda_classifier_experimental.joblib"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def validate_secret_key(secret_key: str) -> None:
    if len(secret_key) < MIN_SECRET_KEY_LENGTH or secret_key == EXAMPLE_SECRET_KEY:
        raise SystemExit(
            "SECRET_KEY invalida: configure uma chave com pelo menos 32 caracteres "
            "e diferente de 'changeme-local-only'."
        )


settings = Settings()  # type: ignore[call-arg]
validate_secret_key(settings.SECRET_KEY)
