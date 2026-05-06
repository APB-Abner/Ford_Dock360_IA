from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


MIN_SECRET_KEY_LENGTH = 32
EXAMPLE_SECRET_KEY = "changeme-local-only"


class Settings(BaseSettings):
    SECRET_KEY: str

    # Paths (defaults relative to workspace root)
    DATA_PATH: str = "data/raw/ford_complaints_top3_por_modelo.csv"
    MODELS_DIR: str = str(Path(__file__).resolve().parents[2] / 'models')

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def validate_secret_key(secret_key: str) -> None:
    if len(secret_key) < MIN_SECRET_KEY_LENGTH or secret_key == EXAMPLE_SECRET_KEY:
        raise SystemExit(
            "SECRET_KEY invalida: configure uma chave com pelo menos 32 caracteres "
            "e diferente de 'changeme-local-only'."
        )


settings = Settings()  # type: ignore[call-arg]
validate_secret_key(settings.SECRET_KEY)
