"""
pytest configuration for the OBPDS test suite.

The verification code has two shapes.  The ``validate_*.py`` scripts are
self-contained programs (a custom ``check()`` + ``all_ok`` + ``sys.exit``)
whose stdout is the reproducible ``paper/cpc/data/*.out`` artefact, so they
stay as scripts and are driven from ``test_physics.py`` through a subprocess.
``web_smoke`` is run against an in-process ``TestClient`` so it needs no
server or port.  The three markers below let one command run the whole thing
while still picking out a single class:

    python -m pytest                 # everything
    python -m pytest -m unit         # fast pure-function checks
    python -m pytest -m physics      # solver validation (slow)
    python -m pytest -m api          # web-API contract tests
"""

import os
import sys

# The package lives under src/; put it on the path so the tests import
# ``obpds`` and ``web.app`` regardless of whether ``pip install -e`` has been
# run in this environment.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unit: fast pure-function checks (Fermi table, Bernoulli, ...)")
    config.addinivalue_line(
        "markers",
        "physics: solver validation against closed-form results (slow)")
    config.addinivalue_line(
        "markers",
        "api: web-API contract tests through an in-process TestClient")


@pytest.fixture(scope="session", autouse=True)
def _shutdown_worker_pool():
    """Shut down the web app's background thread pool after the API tests.

    ``web.app`` owns a module-level ``ThreadPoolExecutor``.  Leaving it
    running at interpreter exit is harmless, but pytest reports stray threads
    on some versions; closing it here keeps the run clean.
    """
    yield
    import web.app as wa
    wa._POOL.shutdown(wait=True)
