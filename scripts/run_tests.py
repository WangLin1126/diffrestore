"""Entry point for the Phase-0 numerical gates.  Usage: python smdc/scripts/run_tests.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.gates import run_all  # noqa: E402

if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
