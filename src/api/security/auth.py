from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.api.config import settings


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
bearer_scheme = HTTPBearer()
ROLE_LEVELS = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
}


def _role_value(role: Any) -> str:
    return role.value if hasattr(role, "value") else str(role)


def create_access_token(subject: str, role: Any) -> str:
    expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "role": _role_value(role),
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "exp": expires_at,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def _current_role(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            options={
                "require_exp": True,
                "require_iss": True,
                "require_aud": True,
                "require_sub": True,
            },
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Token JWT invalido")

    if not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Token JWT invalido")

    role = payload.get("role")
    if role not in ROLE_LEVELS:
        raise HTTPException(status_code=403, detail="Role invalida")
    return role


def require_role(required_role: Any):
    def dependency(role: str = Depends(_current_role)):
        if ROLE_LEVELS[role] < ROLE_LEVELS[_role_value(required_role)]:
            raise HTTPException(status_code=403, detail="Permissao insuficiente")
        return role

    return dependency
