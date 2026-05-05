
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import settings
from app.models.schemas import RoleEnum


ALGORITHM = "HS256"
bearer_scheme = HTTPBearer()
ROLE_LEVELS = {
    RoleEnum.viewer.value: 1,
    RoleEnum.analyst.value: 2,
    RoleEnum.admin.value: 3,
}


def _current_role(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[ALGORITHM])
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
