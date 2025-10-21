"""Pydantic schemas for request and response payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class DeckCreate(BaseModel):
    name: str = Field(..., min_length=1)


class DeckRename(BaseModel):
    new_name: str = Field(..., min_length=1)


class DeckOut(BaseModel):
    name: str


class CardCreate(BaseModel):
    content: Dict[str, Any] = Field(..., description="Structured card content")
    deck: str = Field(..., min_length=1, description="Name of the existing deck")
    tags: Optional[List[str]] = None

    @field_validator("content")
    @classmethod
    def _content_non_empty(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("content must be a non-empty JSON object")
        return value


class CardUpdate(BaseModel):
    content: Optional[Dict[str, Any]] = Field(None, description="New JSON content")
    deck: Optional[str] = Field(None, min_length=1, description="New deck (optional)")
    tags: Optional[List[str]] = None

    @field_validator("content")
    @classmethod
    def _content_non_empty_update(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is not None and (not isinstance(value, dict) or not value):
            raise ValueError("content must be a non-empty JSON object")
        return value


class CardOut(BaseModel):
    id: int
    content: Dict[str, Any]
    deck: Optional[str]
    tags: Optional[List[str]]
    repetitions: int
    interval: int
    efactor: float
    due: str
    last_reviewed: Optional[str]
    lapses: int


class ReviewIn(BaseModel):
    grade: int = Field(..., ge=0, le=5, description="SM-2 grade: 0–5")

    @field_validator("grade")
    @classmethod
    def _grade_range(cls, value: int) -> int:
        if not (0 <= value <= 5):
            raise ValueError("grade must be between 0 and 5")
        return value


class ReviewButtonIn(BaseModel):
    button: Literal["again", "hard", "good", "easy"]


class ReviewOut(BaseModel):
    card: CardOut
    next_due: str
    interval_days: int
    efactor: float
    repetitions: int
    lapses: int
    reviewed_card_id: int


class CardsPage(BaseModel):
    items: List[CardOut]
    total: int
    limit: int
    offset: int
    next_offset: Optional[int]
