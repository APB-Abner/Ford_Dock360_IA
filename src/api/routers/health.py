from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.config import settings

router = APIRouter(tags=["health"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / settings.MODELS_DIR


def _models_loaded() -> dict[str, bool]:
    churn = (_MODELS_DIR / settings.CHURN_MODEL_FILENAME).exists()
    kmeans = (_MODELS_DIR / settings.PERFIL_MODEL_FILENAME).exists()
    return {"churn": churn, "kmeans": kmeans}


@router.get("/health", response_model=None)
def health() -> dict | JSONResponse:
    loaded = _models_loaded()
    all_ok = all(loaded.values())

    payload = {
        "status": "ok" if all_ok else "degraded",
        "models_loaded": loaded,
    }

    if not all_ok:
        return JSONResponse(status_code=503, content=payload)

    return payload
