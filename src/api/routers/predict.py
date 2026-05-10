from fastapi import APIRouter, Depends
from src.api.models.schemas import BatchPredictRequest, PredictRequest, PredictResponse, RoleEnum
from src.api.security.auth import require_role
from src.api.services.predictor import predictor_service

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, role: str = Depends(require_role(RoleEnum.analyst))):
    return predictor_service.predict(request.features.model_dump(), modelo_veiculo=request.modelo_veiculo)


@router.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest, role: str = Depends(require_role(RoleEnum.analyst))):
    return predictor_service.predict_batch(request.items)
