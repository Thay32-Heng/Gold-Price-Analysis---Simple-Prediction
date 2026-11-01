"""Markdown safety helpers for the Gold Lens Streamlit app."""

from __future__ import annotations

__all__ = ["md_safe"]


def md_safe(text: str) -> str:
    """Escape ``$`` so Streamlit renders it literally instead of as LaTeX.

    Streamlit enables ``$...$`` maths in Markdown, so any string that happens
    to contain two dollar amounts — ``"($1,418.5 vs $40.2 MAE)"`` — loses both
    signs and gets re-rendered in a math font with wide gaps. Library code
    should keep returning plain text (the CLI prints it too), so escaping
    belongs here, at the point where the string becomes Markdown.

    This lives in its own module rather than in ``lib/__init__.py`` on purpose:
    a running Streamlit server caches imported packages in ``sys.modules``, so
    a new symbol added to ``lib`` is invisible to a process that started before
    the change. A brand-new submodule is always read from disk, which makes a
    half-reloaded server keep working instead of crashing on import.
    """
    return text.replace("$", r"\$")
