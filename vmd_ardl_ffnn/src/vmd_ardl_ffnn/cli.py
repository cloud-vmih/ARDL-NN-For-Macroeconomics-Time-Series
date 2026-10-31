"""Canonical installed CLI for run-scoped experiments."""

from .advanced_runner import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
