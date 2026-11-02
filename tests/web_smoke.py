"""
End-to-end smoke test of the web API against a running server.

    python tests/web_smoke.py [base_url]

Posts each preset through /api/solve and /api/sweep and checks that the
payload the front end actually draws on is present and finite.  The checks
themselves live in :mod:`web_checks` so the same body is driven in-process by
``test_web_api.py``; this script only supplies the ``urllib`` transport.
"""
from __future__ import annotations

import json
import sys
import urllib.request

from web_checks import run_all

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")


run_all(get, post, check)

print("\n" + ("WEB SMOKE PASSED" if ok else "WEB SMOKE FAILED"))
sys.exit(0 if ok else 1)
