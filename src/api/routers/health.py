from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.config import settings

router = APIRouter(tags=["health"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / settings.MODELS_DIR


def _models_loaded() -> dict[str, bool]:
    if not _MODELS_DIR.exists():
        return {"churn": False, "kmeans": False}

    churn = (_MODELS_DIR / settings.CHURN_MODEL_FILENAME).exists()
    kmeans = (_MODELS_DIR / settings.PERFIL_MODEL_FILENAME).exists()
    return {"churn": churn, "kmeans": kmeans}


@router.get("/health", response_model=None)
def health() -> dict | JSONResponse:
    loaded = _models_loaded()
    artifacts_ok = all(loaded.values())

    payload = {
        "status": "ok" if artifacts_ok else "degraded",
        "checks": {
            "secret_key": True,
            "artifacts": artifacts_ok,
        },
        "models_loaded": loaded,
    }

    if not artifacts_ok:
        return JSONResponse(status_code=503, content=payload)

    return payload
