from fastapi import FastAPI

from app.routers import health, predict


app = FastAPI(title="Ford VinGuard ML API")
app.include_router(health.router)
app.include_router(predict.router)
