"""
pytest entry for the fast pure-function checks (``marker = unit``).

``validate_fermi.py`` pins the tabulated Fermi--Dirac integral against the
Gauss--Legendre quadrature (nodes exact to round-off, interpolation error
bounded, out-of-table fallback exact, and the H(phi,0) = F_{1/2} identity).
It is pure NumPy -- no device solve -- so it belongs in the ``unit`` class,
not the slow ``physics`` class.

Run only these with ``python -m pytest -m unit``.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYTHON = sys.executable


@pytest.mark.unit
def test_fermi_table_validation():
    path = os.path.join(HERE, "validate_fermi.py")
    r = subprocess.run([PYTHON, path], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (
        f"validate_fermi.py failed (exit {r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
