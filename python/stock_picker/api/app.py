"""The FastAPI app object -- separate from main.py so it's importable by tests
without needing uvicorn (mirrors the rest of the project's thin-main-py pattern)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from stock_picker.api.routes import router

DEV_ORIGIN = "http://localhost:5173"  # the Vite dev server


def _web_dist() -> Path:
    root = Path(os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd()))
    return root / "typescript" / "dist"


app = FastAPI(title="stock-picker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[DEV_ORIGIN],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(router)

_dist = _web_dist()
if _dist.is_dir():
    # After /api so the SPA cannot swallow JSON routes. html=True serves
    # index.html for / so the iPhone Home Screen app has a document root.
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
