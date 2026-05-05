from fastapi import APIRouter, Depends

from app.models.schemas import BatchPredictRequest, PredictRequest, PredictResponse, RoleEnum
from app.security.auth import require_role
from app.services.predictor import predictor_service


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, role: str = Depends(require_role(RoleEnum.viewer))):
    return predictor_service.predict(request.features)


@router.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest, role: str = Depends(require_role(RoleEnum.analyst))):
    return [predictor_service.predict(item.features) for item in request.items]
