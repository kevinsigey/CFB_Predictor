"""main.py — FastAPI app entry point."""

import asyncio
import logging
import shutil
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # This loads variables from your .env file into os.environ

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse

from ..config import PUBLIC_MODE, PUBLIC_SNAPSHOT_POLL_SECONDS, TRACKING_DB_BACKUP_PATH, TRACKING_DB_PATH
from .facts import router as facts_router
from .signals import router as signals_router
from .routes import (
    current_season_and_week,
    refresh_public_snapshot_from_remote,
    router,
    background_tracking_tick,
)

logger = logging.getLogger(__name__)

_TRACKING_INTERVAL_SECONDS = 3600


def _restore_tracking_db() -> None:
    """Copy the last known-good tracking db from the persistent cache
    mount into place before anything opens it -- see
    config.py::TRACKING_DB_BACKUP_PATH for why this is a file copy
    instead of pointing the live db at the mount directly."""
    if TRACKING_DB_BACKUP_PATH.exists() and not TRACKING_DB_PATH.exists():
        try:
            shutil.copy2(TRACKING_DB_BACKUP_PATH, TRACKING_DB_PATH)
        except Exception:
            logger.exception("tracking db restore failed")


def _backup_tracking_db() -> None:
    """Copy the live tracking db out to the persistent cache mount.
    Safe to call right after a tick: every connection in tracking/store.py
    is opened and closed per-call (contextlib.closing), so there's never
    a connection left open on the local file at this point."""
    if TRACKING_DB_PATH.exists():
        try:
            shutil.copy2(TRACKING_DB_PATH, TRACKING_DB_BACKUP_PATH)
        except Exception:
            logger.exception("tracking db backup failed")


async def _run_tracking_tick():
    try:
        season, week = current_season_and_week()
        await asyncio.to_thread(background_tracking_tick, season, week)
        await asyncio.to_thread(_backup_tracking_db)
    except Exception:
        logger.exception("background_tracking_tick failed")


async def _tracking_loop():
    # Tick once immediately, THEN sleep -- this app scales to zero between
    # requests, so a sleep-first loop with a 1-hour interval essentially
    # never got to run: the container rarely stays warm continuously for a
    # full hour, so it was cold-started, served some requests, and scaled
    # back down long before the first sleep ever elapsed (confirmed live:
    # games had finished and been resolved for a while, but nothing had
    # ever reconciled them). Ticking on every cold start instead means
    # every burst of traffic gets at least one real reconciliation pass.
    await _run_tracking_tick()
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        await _run_tracking_tick()


async def _public_snapshot_poll_loop():
    while True:
        await asyncio.to_thread(refresh_public_snapshot_from_remote)
        await asyncio.sleep(PUBLIC_SNAPSHOT_POLL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Restore the tracking db from its persistent backup BEFORE anything
    # (including the first tracking tick, below) can touch it -- the live
    # db is on local ephemeral disk, wiped by every cold start, otherwise.
    _restore_tracking_db()

    # The public deployment serves games/predictions/player-props from
    # public_snapshot.py's precomputed file (see routes.py's PUBLIC_MODE
    # branches). The tracking loop still runs regardless -- it's what
    # feeds the track-record/verdict history, is already throttled by its
    # own TTLs (see data/games.py's _CURRENT_SEASON_TTL_SECONDS), and skipping
    # it would leave that feature permanently empty on the public deployment.
    tasks = [asyncio.create_task(_tracking_loop())]
    if PUBLIC_MODE:
        tasks.append(asyncio.create_task(_public_snapshot_poll_loop()))
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(title="CFB Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Track-record and prop payloads are large, repetitive JSON; gzip cuts them
# several-fold for any client that sends Accept-Encoding: gzip.
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    # Always logged server-side, regardless of mode. The raw traceback is
    # only ever returned to the caller outside PUBLIC_MODE (local/private
    # dev) -- the public deployment is internet-reachable, and a raw
    # traceback in an HTTP response is a real information-disclosure risk
    # (file paths, code structure, and whatever local values happened to be
    # in scope when the exception was raised), not just a debugging nicety.
    logger.exception("Unhandled exception during request %s %s", request.method, request.url.path)
    if PUBLIC_MODE:
        return PlainTextResponse(status_code=500, content="Internal Server Error")
    error_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return PlainTextResponse(status_code=500, content=f"DEBUG_TRACEBACK:\n{error_trace}")


app.include_router(router)
# The explainer service calls {SPORT_API}/facts/{id} on the API root, so this
# router carries no /api prefix. /facts/upcoming is declared before
# /facts/{game_id} inside facts.py so it isn't swallowed by the path parameter.
app.include_router(facts_router)

# GET /api/signals/{game_id} — spec §3's per-fixture signal payloads, rendered by
# the shared `SignalRows` under the facts block. This one DOES carry the /api
# prefix, unlike `facts_router` above: the explainer service is what calls
# `/facts/{id}`, and it addresses the API root, whereas this endpoint is read by
# this site's own frontend, which reaches the backend through `/api` in both
# production (`client.ts`'s `BASE_URL`) and development (`vite.config.ts`
# proxies only '/api'). F1 shipped this router mounted at the root and the page
# could not call it; the tests here pin the prefix.
app.include_router(signals_router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "CFB Predictor API",
        "docs": "/docs",
    }
