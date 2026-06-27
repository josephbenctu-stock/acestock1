# -*- coding: utf-8 -*-
"""Run V8 unit tests without requiring pytest as a dependency."""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_DIR = ROOT / "tests"


def main() -> int:
    failures = []
    for path in sorted(TEST_DIR.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            for name in dir(mod):
                if name.startswith("test_") and callable(getattr(mod, name)):
                    getattr(mod, name)()
                    print(f"PASS {path.name}::{name}")
        except Exception:
            failures.append((path.name, traceback.format_exc()))
    if failures:
        for name, err in failures:
            print(f"FAIL {name}\n{err}")
        return 1
    print("All V8 tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
