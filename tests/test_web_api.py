"""
pytest entry for the web-API contract tests (``marker = api``).

Drives ``web.app.app`` through ``fastapi.testclient.TestClient`` -- no server,
no port, no network -- and reuses the exact same contract checks as the
standalone ``web_smoke.py`` script (see :mod:`web_checks`).  Failures are
collected and reported together, one line per failed contract, instead of
aborting at the first.

Run only these with ``python -m pytest -m api``.
"""

import pytest
from fastapi.testclient import TestClient

import web.app as wa
from web_checks import run_all


@pytest.mark.api
def test_web_api_contract():
    client = TestClient(wa.app)

    def get(path):
        r = client.get(path)
        r.raise_for_status()
        return r.json()

    def post(path, payload):
        r = client.post(path, json=payload)
        r.raise_for_status()
        return r.json()

    failures = []

    def check(name, cond, extra=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        if not cond:
            failures.append(f"{name} {extra}".strip())

    run_all(get, post, check)
    assert not failures, (
        f"{len(failures)} web-API contract failure(s):\n  "
        + "\n  ".join(failures))
