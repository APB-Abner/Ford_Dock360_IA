from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SECRET_KEY: str = "MUST_BE_SET_IN_ENVIRONMENT"
    
    # Paths (defaults relative to workspace root)
    DATA_PATH: str = "data/raw/ford_complaints_top3_por_modelo.csv"
    MODELS_DIR: str = "models"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
