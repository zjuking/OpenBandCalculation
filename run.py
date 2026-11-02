#!/usr/bin/env python3
"""
Deployment entry point.

Hosting sandboxes inject a single ``PORT`` and expect the service bound to
0.0.0.0; this puts the ``src`` layout on the path and starts uvicorn there,
so the app runs the same way whether or not it was pip-installed first.

    python run.py            # http://127.0.0.1:8000
    PORT=8080 python run.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from web.app import app                                  # noqa: E402


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", 8000)), log_level="info")


if __name__ == "__main__":
    main()
