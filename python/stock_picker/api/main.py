"""Entrypoint: run the FastAPI backend serving the feature catalog/registry.

Run with: bazel run //python/stock_picker/api:main
"""

from __future__ import annotations

import uvicorn

from stock_picker.api.app import app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000  # kept in sync with typescript/vite.config.ts's API_PROXY_TARGET


def main() -> None:
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
