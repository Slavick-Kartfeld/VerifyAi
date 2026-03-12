from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.core.database import engine, Base
from app.api.routes import router

# ── Rate limiter — keyed by IP ────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/day", "60/hour"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="VerifyAI",
    description="Forensic media authentication — multi-agent AI pipeline",
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the production frontend + localhost dev. Tighten in Sprint 3 with JWT.
ALLOWED_ORIGINS = [
    "https://verifyai-3.onrender.com",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],          # only what we actually use
    allow_headers=["Content-Type", "Authorization", "X-Client-ID"],
)

# ── Rate limiting middleware + handler ────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def landing():
    return FileResponse(str(static_dir / "landing.html"))


@app.get("/app")
async def main_app():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}
