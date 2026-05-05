import os
import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi import HTTPException
from app.models.schemas import ChurnLabelEnum, PredictResponse, RiskLevelEnum

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.pipeline.complaints_loader import get_top3_por_modelo
    _COMPLAINTS_OK = True
except ImportError:
    _COMPLAINTS_OK = False

_ACOES = {
    "abandono": "Ativar Pulse Loop imediato. Oferecer plano de manutencao com desconto antes da primeira revisao.",
    "esquecido": "Configurar lembrete 45 dias antes do vencimento. Verificar disponibilidade de agenda.",
    "economico": "Apresentar tabela comparativa custo oficial vs externo. Oferta de pacote economico com preco fixo.",
    "fiel": "Nenhuma acao ativa. Registrar para agradecimento apos proxima revisao.",
}


class PredictorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_churn = None
            cls._instance.model_perfil = None
            cls._instance.feature_names = None
        return cls._instance

    def _model_path(self, filename):
        env_key = "MODEL_PATH_" + filename.upper().replace(".", "_").replace("-", "_")
        return Path(os.environ.get(env_key, _PROJECT_ROOT / "models" / filename))

    def _load_churn(self):
        if self.model_churn is None:
            path = self._model_path("churn_rf_calibrated.joblib")
            if not path.exists():
                raise HTTPException(status_code=503, detail=f"Modelo churn nao encontrado: {path}")
            self.model_churn = joblib.load(path)
            self.feature_names = list(getattr(self.model_churn, "feature_names_in_", []))
        return self.model_churn

    def _load_perfil(self):
        if self.model_perfil is None:
            path = self._model_path("perfil_rf_classifier.joblib")
            if path.exists():
                self.model_perfil = joblib.load(path)
        return self.model_perfil

    def _build_frame(self, features):
        self._load_churn()
        if not self.feature_names:
            estimator = getattr(self.model_churn, "estimator", None)
            preprocessor = getattr(estimator, "named_steps", {}).get("preprocessor")
            self.feature_names = list(getattr(preprocessor, "feature_names_in_", []))
        if self.feature_names:
            missing = [c for c in self.feature_names if c not in features]
            if missing:
                raise HTTPException(status_code=422, detail={"missing_features": missing})
            features = {c: features[c] for c in self.feature_names}
        return pd.DataFrame([features])

    def _predict_churn(self, frame):
        model = self._load_churn()
        prob = float(model.predict_proba(frame)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(frame)[0])
        label = ChurnLabelEnum.churn if prob >= 0.5 else ChurnLabelEnum.no_churn
        risk = RiskLevelEnum.high if prob >= 0.70 else (RiskLevelEnum.medium if prob >= 0.40 else RiskLevelEnum.low)
        return label, round(prob, 6), risk

    def _predict_perfil(self, frame):
        model = self._load_perfil()
        if model is None:
            return None, None
        try:
            probs = model.predict_proba(frame)[0]
            classes = model.classes_
            perfil = str(classes[probs.argmax()])
            return perfil, {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        except Exception:
            return None, None

    def predict(self, features, modelo_veiculo=None):
        frame = self._build_frame(features)
        label, prob_churn, risk = self._predict_churn(frame)
        perfil, prob_perfil = self._predict_perfil(frame)
        historico = []
        if _COMPLAINTS_OK and modelo_veiculo:
            try:
                historico = get_top3_por_modelo(modelo_veiculo)
            except Exception:
                pass
        return PredictResponse(
            prediction=label,
            churn_probability=prob_churn,
            risk_level=risk,
            perfil_previsto=perfil,
            probabilidades_perfil=prob_perfil,
            acao_recomendada=_ACOES.get(perfil) if perfil else None,
            historico_problemas=historico,
        )


predictor_service = PredictorService()
