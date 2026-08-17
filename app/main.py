from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_eval import router as eval_router
from app.api.routes_route import router as route_router
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Adaptive Model Router",
    description="Routes tasks to the weakest/cheapest AI model likely to solve them reliably, escalating only when genuinely needed.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route_router, tags=["routing"])
app.include_router(eval_router, tags=["evaluation"])

_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if _frontend_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_frontend_dir), html=True), name="dashboard")


@app.get("/")
def root():
    return {"service": "adaptive-model-router", "docs": "/docs", "dashboard": "/dashboard"}
