# -*- coding: utf-8 -*-
"""Validate ACE Triangle V8 project structure and Python syntax."""
from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "app.py",
    "requirements.txt",
    "README.md",
    "scripts/scheduled_scan.py",
    "scripts/run_tests.py",
    ".github/workflows/ace_daily_scan.yml",
    "ace_engine/__init__.py",
    "ace_engine/data_provider.py",
    "ace_engine/indicators.py",
    "ace_engine/triangle_engine.py",
    "ace_engine/signal_engine.py",
    "ace_engine/risk_engine.py",
    "ace_engine/backtest_engine.py",
    "ace_engine/database.py",
    "ace_engine/report_engine.py",
    "ace_engine/paper_trading.py",
    "tests/test_engine.py",
]


def main() -> int:
    ok = True
    for rel in REQUIRED:
        path = ROOT / rel
        if path.exists():
            print(f"OK   {rel}")
        else:
            print(f"MISS {rel}")
            ok = False
    for path in [ROOT / "app.py", ROOT / "scripts" / "scheduled_scan.py", ROOT / "scripts" / "run_tests.py", *sorted((ROOT / "ace_engine").glob("*.py")), *sorted((ROOT / "tests").glob("test_*.py"))]:
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"PYOK {path.relative_to(ROOT)}")
        except Exception as exc:
            print(f"PYFAIL {path.relative_to(ROOT)}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
