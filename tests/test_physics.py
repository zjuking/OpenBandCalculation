"""
pytest entry for the solver validation suites (``marker = physics``).

Each ``validate_*.py`` is a self-contained program whose stdout is the
reproducible ``paper/cpc/data/*.out`` artefact, so it is run as a subprocess
rather than re-imported.  A non-zero exit means at least one closed-form
check failed; the subprocess stdout/stderr are echoed back for diagnosis.

Run only these with ``python -m pytest -m physics``.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYTHON = sys.executable

SCRIPTS = ["validate_physics.py", "validate_quantum.py", "validate_strain.py"]


@pytest.mark.physics
@pytest.mark.parametrize("script", SCRIPTS)
def test_physics_suite(script):
    path = os.path.join(HERE, script)
    r = subprocess.run([PYTHON, path], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (
        f"{script} failed (exit {r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
