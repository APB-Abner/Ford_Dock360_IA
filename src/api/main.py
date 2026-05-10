from fastapi import FastAPI

from src.api.routers import auth, health, predict


app = FastAPI(title="Ford VinGuard ML API")
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
