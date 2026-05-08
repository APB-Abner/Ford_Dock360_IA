import pytest
from pydantic import ValidationError
from src.api.models.schemas import PredictRequest, PredictResponse, ChurnLabelEnum, RiskLevelEnum

def test_predict_request_valid():
    data = {
        "features": {
            "ano_modelo": 2023,
            "dias_ate_entrega": 11,
            "idade_veiculo_meses": 18.5,
            "modelo": "RANGER"
        },
        "modelo_veiculo": "Ranger"
    }
    req = PredictRequest(**data)
    assert req.features["modelo"] == "RANGER"

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
