import os

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.models.schemas import RoleEnum


ALGORITHM = "HS256"
bearer_scheme = HTTPBearer()
ROLE_LEVELS = {
    RoleEnum.viewer.value: 1,
    RoleEnum.analyst.value: 2,
    RoleEnum.admin.value: 3,
}


def _secret_key():
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise HTTPException(status_code=500, detail="SECRET_KEY nao configurada")
    return secret_key


def _current_role(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        payload = jwt.decode(credentials.credentials, _secret_key(), algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token JWT invalido")

    role = payload.get("role")
    if role not in ROLE_LEVELS:
        raise HTTPException(status_code=403, detail="Role invalida")
    return role


def require_role(required_role: RoleEnum):
    def dependency(role: str = Depends(_current_role)):
        if ROLE_LEVELS[role] < ROLE_LEVELS[required_role.value]:
            raise HTTPException(status_code=403, detail="Permissao insuficiente")
        return role

    return dependency
