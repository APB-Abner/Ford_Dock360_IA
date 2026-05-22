import hmac
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request

from src.api.limiter import limiter

from src.api.config import settings
from src.api.models.schemas import DemoTokenResponse, RoleEnum
from src.api.security.auth import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/demo-token", response_model=DemoTokenResponse)
@limiter.limit("10/minute")
async def create_demo_token(
    request: Request,
    x_demo_token_secret: Annotated[str, Header(alias="X-Demo-Token-Secret")],
):
    if not settings.DEMO_TOKEN_SECRET:
        raise HTTPException(status_code=503, detail="DEMO_TOKEN_SECRET nao configurado")

    if not hmac.compare_digest(x_demo_token_secret, settings.DEMO_TOKEN_SECRET):
        raise HTTPException(status_code=403, detail="Segredo de demo invalido")

    role = RoleEnum.analyst
    token = create_access_token(settings.DEMO_TOKEN_SUBJECT, role)
    return DemoTokenResponse(
        access_token=token,
        role=role,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )
