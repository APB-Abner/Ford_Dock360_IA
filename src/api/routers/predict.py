from fastapi import APIRouter, Depends, Request

from src.api.main import limiter
from src.api.models.schemas import BatchPredictRequest, PredictRequest, PredictResponse, RoleEnum
from src.api.security.auth import require_role
from src.api.services.predictor import predictor_service

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
@limiter.limit("60/minute")
async def predict(request: Request, body: PredictRequest, role: str = Depends(require_role(RoleEnum.analyst))):
    return predictor_service.predict(body.features.model_dump())


@router.post("/predict/batch", response_model=list[PredictResponse])
@limiter.limit("20/minute")
async def predict_batch(request: Request, body: BatchPredictRequest, role: str = Depends(require_role(RoleEnum.analyst))):
    return predictor_service.predict_batch(body.items)
