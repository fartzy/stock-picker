"""The FastAPI app object -- separate from main.py so it's importable by tests
without needing uvicorn (mirrors the rest of the project's thin-main-py pattern)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stock_picker.api.routes import router

DEV_ORIGIN = "http://localhost:5173"  # the Vite dev server

app = FastAPI(title="stock-picker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[DEV_ORIGIN],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(router)
