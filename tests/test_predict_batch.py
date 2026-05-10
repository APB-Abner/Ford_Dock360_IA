import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.models.schemas import PredictRequest, PredictResponse
from src.api.services.predictor import PredictorService


class FakeChurnModel:
    feature_names_in_ = [
        "ano_modelo",
        "qtde_revisoes_ate_corte",
        "meses_desde_ultimo_servico_ate_corte",
        "meses_relacionamento_ate_corte",
        "n_dealers_usados_ate_corte",
        "km_max_ate_corte",
        "pct_agenda_ate_corte",
        "intervalo_medio_revisoes_dias_ate_corte",
        "dias_ate_primeira_revisao",
        "idade_veiculo_meses_ate_corte",
        "modelo",
    ]

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
    service.feature_names = FakeChurnModel.feature_names_in_

    features_1 = {
        "ano_modelo": 2023,
        "qtde_revisoes_ate_corte": 1,
        "meses_desde_ultimo_servico_ate_corte": 6.0,
        "meses_relacionamento_ate_corte": 18.0,
        "n_dealers_usados_ate_corte": 1,
        "km_max_ate_corte": 23400,
        "pct_agenda_ate_corte": 0.8,
        "intervalo_medio_revisoes_dias_ate_corte": 190.0,
        "dias_ate_primeira_revisao": 160,
        "idade_veiculo_meses_ate_corte": 18.0,
        "modelo": "RANGER",
    }
    features_2 = {
        "ano_modelo": 2021,
        "qtde_revisoes_ate_corte": 3,
        "meses_desde_ultimo_servico_ate_corte": 16.0,
        "meses_relacionamento_ate_corte": 42.0,
        "n_dealers_usados_ate_corte": 2,
        "km_max_ate_corte": 67200,
        "pct_agenda_ate_corte": 0.55,
        "intervalo_medio_revisoes_dias_ate_corte": 240.0,
        "dias_ate_primeira_revisao": 190,
        "idade_veiculo_meses_ate_corte": 42.0,
        "modelo": "KA",
    }

    items = [
        PredictRequest(features=features_1, modelo_veiculo="Ranger"),
        PredictRequest(features=features_2, modelo_veiculo="Ka"),
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
