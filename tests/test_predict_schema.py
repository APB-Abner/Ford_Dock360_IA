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
    req = PredictRequest(features=PREDICT_FEATURES_EXAMPLE)
    assert req.features.modelo == "KA"


def test_predict_request_missing_features():
    with pytest.raises(ValidationError):
        PredictRequest()


def test_predict_response_valid():
    resp = PredictResponse(
        prediction="no_churn",
        churn_probability=0.23,
        risk_level="low",
        perfil_previsto="recorrente",
        probabilidades_perfil={"recorrente": 0.8, "inativo": 0.2},
        acao_recomendada="Nenhuma acao ativa de recuperacao.",
    )
    assert resp.prediction == ChurnLabelEnum.no_churn
    assert resp.risk_level == RiskLevelEnum.low
