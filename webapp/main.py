from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from webapp.database import init_db
from webapp.middleware import RateLimitMiddleware
from webapp.routers import health_router, auth_router, jobs_router, stripe_router
from webapp.services.worker import shutdown as shutdown_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    shutdown_worker()


app = FastAPI(title="Noion - AI耳コピサービス", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)

app.include_router(health_router.router)
app.include_router(auth_router.router)
app.include_router(jobs_router.router)
app.include_router(stripe_router.router)

frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
