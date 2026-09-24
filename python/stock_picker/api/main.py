"""Entrypoint: run the FastAPI backend serving the feature catalog/registry.

Run with: bazel run //python/stock_picker/api:main
"""

from __future__ import annotations

import socket

import uvicorn

from stock_picker.api.app import app
from stock_picker.log import get_logger

# 0.0.0.0 so an iPhone on the same Wi-Fi can open the hosted UI. Vite still
# proxies /api to this port on loopback.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000  # kept in sync with typescript/vite.config.ts's API_PROXY_TARGET

logger = get_logger(__name__)


def _lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> None:
    logger.info("phone Home Screen: http://%s:%s", _lan_ip(), DEFAULT_PORT)
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
