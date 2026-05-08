from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    viewer = "viewer"
    analyst = "analyst"
    admin = "admin"


class ChurnLabelEnum(str, Enum):
    churn = "churn"
    no_churn = "no_churn"


class RiskLevelEnum(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class PredictRequest(BaseModel):
    features: dict
    modelo_veiculo: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "ano_modelo": 2023,
                    "dias_ate_entrega": 11,
                    "idade_veiculo_meses": 18.5,
                    "modelo": "RANGER"
                },
                "modelo_veiculo": "Ranger"
            }
        }
    }


class PredictResponse(BaseModel):
    prediction: ChurnLabelEnum
    churn_probability: float
    risk_level: RiskLevelEnum
    perfil_previsto: Optional[str] = None
    probabilidades_perfil: Optional[dict] = None
    acao_recomendada: Optional[str] = None
    historico_problemas: Optional[list] = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., max_length=500)
