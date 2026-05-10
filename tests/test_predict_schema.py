import pytest
from pydantic import ValidationError
from src.api.models.schemas import (
    ChurnLabelEnum,
    PREDICT_FEATURES_EXAMPLE,
    PredictRequest,
    PredictResponse,
    RiskLevelEnum,
)

def test_predict_request_valid():
    data = {
        "features": PREDICT_FEATURES_EXAMPLE,
        "modelo_veiculo": "Ka",
    }
    req = PredictRequest(**data)
    assert req.features.modelo == "KA"

def test_predict_request_missing_features():
    data = {"modelo_veiculo": "Ranger"}
    with pytest.raises(ValidationError):
        PredictRequest(**data)

def test_predict_response_valid():
    data = {
        "prediction": "no_churn",
        "churn_probability": 0.23,
        "risk_level": "low",
        "perfil_previsto": "fiel",
        "probabilidades_perfil": {"fiel": 0.8, "economico": 0.2},
        "acao_recomendada": "Nenhuma acao ativa.",
        "historico_problemas": []
    }
    resp = PredictResponse(**data)
    assert resp.prediction == ChurnLabelEnum.no_churn
    assert resp.risk_level == RiskLevelEnum.low
