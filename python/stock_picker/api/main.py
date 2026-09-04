"""Entrypoint: run the FastAPI backend serving the feature catalog/registry.

Run with: bazel run //python/stock_picker/api:main
"""

from __future__ import annotations

import uvicorn

from stock_picker.api.app import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
