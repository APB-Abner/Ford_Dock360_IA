from fastapi import FastAPI

from src.api.routers import health, predict


app = FastAPI(title="Ford VinGuard ML API")
app.include_router(health.router)
app.include_router(predict.router)
