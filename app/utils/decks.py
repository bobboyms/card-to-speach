"""Deck-related utility helpers."""

from __future__ import annotations


def normalize_deck_type(deck_type: str) -> str:
    """Normalize deck type values coming from database or payloads."""
    normalized = (deck_type or "").strip().lower()
    if normalized == "speach":
        return "speech"
    return normalized

