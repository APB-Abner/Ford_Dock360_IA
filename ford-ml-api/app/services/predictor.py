import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

from app.models.schemas import ChurnLabelEnum, PredictResponse, RiskLevelEnum


class PredictorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.feature_names = None
        return cls._instance

    def _model_path(self):
        default_path = Path(__file__).resolve().parents[3] / "models" / "churn_rf_calibrated.joblib"
        return Path(os.environ.get("MODEL_PATH", default_path))

    def load_model(self):
        if self.model is None:
            model_path = self._model_path()
            if not model_path.exists():
                raise HTTPException(status_code=503, detail=f"Modelo nao encontrado: {model_path}")
            self.model = joblib.load(model_path)
            self.feature_names = list(getattr(self.model, "feature_names_in_", []))
        return self.model

    def _build_frame(self, features):
        self.load_model()
        if not self.feature_names:
            estimator = getattr(self.model, "estimator", None)
            preprocessor = getattr(estimator, "named_steps", {}).get("preprocessor")
            self.feature_names = list(getattr(preprocessor, "feature_names_in_", []))

        if self.feature_names:
            missing = [col for col in self.feature_names if col not in features]
            if missing:
                raise HTTPException(status_code=422, detail={"missing_features": missing})
            features = {col: features[col] for col in self.feature_names}
        return pd.DataFrame([features])

    def predict(self, features):
        model = self.load_model()
        frame = self._build_frame(features)

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(frame)[0][1])
        else:
            probability = float(model.predict(frame)[0])

        label = ChurnLabelEnum.churn if probability >= 0.5 else ChurnLabelEnum.no_churn
        if probability >= 0.70:
            risk = RiskLevelEnum.high
        elif probability >= 0.40:
            risk = RiskLevelEnum.medium
        else:
            risk = RiskLevelEnum.low

        return PredictResponse(
            prediction=label,
            churn_probability=round(probability, 6),
            risk_level=risk,
        )


predictor_service = PredictorService()
