from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.config import settings

router = APIRouter(tags=["health"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / settings.MODELS_DIR
_REQUIRED = [settings.CHURN_MODEL_FILENAME]
_OPTIONAL = [settings.PERFIL_MODEL_FILENAME]
_MIN_KEY_LEN = 32
_EXAMPLE_KEY = "changeme-local-only"


def _check_secret_key() -> tuple[bool, str]:
    key = settings.SECRET_KEY
    if len(key) < _MIN_KEY_LEN:
        return False, f"SECRET_KEY ausente ou com menos de {_MIN_KEY_LEN} caracteres"
    if key == _EXAMPLE_KEY:
        return False, "SECRET_KEY invalida: valor de exemplo detectado"
    return True, "ok"


def _check_artifacts() -> tuple[bool, dict[str, str], list[str]]:
    checks: dict[str, str] = {}
    failed: list[str] = []

    if not _MODELS_DIR.exists():
        return False, {"models_dir": f"diretorio ausente: {_MODELS_DIR}"}, ["models_dir"]

    checks["models_dir"] = "ok"

    for name in _REQUIRED:
        path = _MODELS_DIR / name
        if not path.exists():
            checks[name] = "nao encontrado"
            failed.append(name)
        else:
            try:
                path.stat()
                checks[name] = "ok"
            except OSError as exc:
                checks[name] = f"inacessivel: {exc}"
                failed.append(name)

    for name in _OPTIONAL:
        path = _MODELS_DIR / name
        if not path.exists():
            checks[name] = "ausente (opcional)"
        else:
            try:
                path.stat()
                checks[name] = "ok"
            except OSError as exc:
                checks[name] = f"inacessivel: {exc}"

    return len(failed) == 0, checks, failed


@router.get("/health", response_model=None)
def health() -> dict | JSONResponse:
    key_ok, key_msg = _check_secret_key()
    artifacts_ok, artifact_checks, failed = _check_artifacts()

    all_ok = key_ok and artifacts_ok

    payload: dict = {
        "status": "ok" if all_ok else "degraded",
        "checks": {
            "secret_key": "ok" if key_ok else key_msg,
            "artifacts": artifact_checks,
        },
    }

    if not all_ok:
        errors = []
        if not key_ok:
            errors.append(key_msg)
        if not artifacts_ok:
            errors.append(f"artefatos com falha: {failed}")
        payload["errors"] = errors
        return JSONResponse(status_code=503, content=payload)

    return payload
