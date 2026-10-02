"""Backward-compatible wrapper for the live production evaluation command."""

from __future__ import annotations

import sys

from membership_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval-live", *sys.argv[1:]]))
