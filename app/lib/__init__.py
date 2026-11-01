"""Shared helpers for the Gold Lens Streamlit app."""

from __future__ import annotations

__all__ = ["PROJECT_ROOT"]

from pathlib import Path

#: Repository root, resolved from this file so no absolute path is hard-coded.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
