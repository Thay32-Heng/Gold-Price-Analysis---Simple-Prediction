"""Shared helpers for the Gold Lens Streamlit app."""

from __future__ import annotations

__all__ = ["PROJECT_ROOT", "md_safe"]

from pathlib import Path

#: Repository root, resolved from this file so no absolute path is hard-coded.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def md_safe(text: str) -> str:
    """Escape ``$`` so Streamlit renders it literally instead of as LaTeX.

    Streamlit enables ``$...$`` maths in Markdown, so any string that happens
    to contain two dollar amounts — ``"($1,418.5 vs $40.2 MAE)"`` — loses both
    signs and gets re-rendered in a math font with wide gaps. Library code
    should keep returning plain text (the CLI prints it too), so escaping
    belongs here, at the point where the string becomes Markdown.
    """
    return text.replace("$", r"\$")
