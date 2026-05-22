from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.routers import auth, health, predict

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Ford VinGuard ML API", redoc_url=None, openapi_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(auth.router)
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Ford VinGuard ML API",
        "docs": "/docs",
        "readiness": "/health",
    }
