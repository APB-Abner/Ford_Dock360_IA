import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.models.schemas import PredictRequest, PredictResponse
from src.api.services.predictor import PredictorService


class FakeChurnModel:
    feature_names_in_ = ["idade", "renda"]

    def __init__(self):
        self.predict_proba_calls = 0

    def predict_proba(self, frame):
        self.predict_proba_calls += 1
        assert len(frame) == 2
        return np.array([[0.8, 0.2], [0.25, 0.75]])


class FakePerfilModel:
    classes_ = np.array(["fiel", "abandono"])

    def __init__(self):
        self.predict_proba_calls = 0

    def predict_proba(self, frame):
        self.predict_proba_calls += 1
        assert len(frame) == 2
        return np.array([[0.9, 0.1], [0.2, 0.8]])


def test_predict_batch_retorna_lista_de_respostas_vetorizada():
    service = PredictorService()
    original_model_churn = service.model_churn
    original_model_perfil = service.model_perfil
    original_feature_names = service.feature_names
    churn_model = FakeChurnModel()
    perfil_model = FakePerfilModel()
    service.model_churn = churn_model
    service.model_perfil = perfil_model
    service.feature_names = ["idade", "renda"]

    items = [
        PredictRequest(features={"idade": 30, "renda": 5000}, modelo_veiculo="Ranger"),
        PredictRequest(features={"idade": 55, "renda": 9000}, modelo_veiculo="Ka"),
    ]

    try:
        responses = service.predict_batch(items)

        assert churn_model.predict_proba_calls == 1
        assert perfil_model.predict_proba_calls == 1
        assert isinstance(responses, list)
        assert len(responses) == 2
        assert all(isinstance(response, PredictResponse) for response in responses)
        assert responses[0].prediction == "no_churn"
        assert responses[1].prediction == "churn"
    finally:
        service.model_churn = original_model_churn
        service.model_perfil = original_model_perfil
        service.feature_names = original_feature_names
