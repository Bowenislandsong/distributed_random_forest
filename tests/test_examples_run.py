"""Ensure runnable example scripts exit successfully (smoke)."""

import subprocess
import sys
from pathlib import Path


def test_benchmark_cli_quick_mode():
    root = Path(__file__).resolve().parent.parent
    script = root / "examples" / "benchmark_public_dataset.py"
    r = subprocess.run(
        [sys.executable, str(script), "--quick"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = (r.stdout + r.stderr).lower()
    assert "wdbc_breast_cancer" in r.stdout
    assert "test accuracy" in out or "accuracy" in out
