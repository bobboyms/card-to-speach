"""Pydantic schemas for request and response payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, field_validator

# from app.evaluate import DEFAULT_MODEL


def _normalize_deck_type(deck_type: str) -> str:
    normalized = (deck_type or "").strip().lower()
    if normalized == "speach":
        return "speech"
    return normalized


class DeckCreate(BaseModel):
    name: str = Field(..., min_length=1)
    type: Literal["speech", "shadowing", "speach"] = Field(
        "speech", description="Deck modality defining the learning flow."
    )

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalized = _normalize_deck_type(value)
        if normalized not in {"speech", "shadowing"}:
            raise ValueError("type must be 'speech' or 'shadowing'")
        return normalized


class DeckRename(BaseModel):
    new_name: str = Field(..., min_length=1)


class DeckOut(BaseModel):
    public_id: str
    name: str
    type: Literal["speech", "shadowing"]
    due_cards: int
    total_cards: int


class CardCreate(BaseModel):
    content: Dict[str, Any] = Field(..., description="Structured card content")
    deck_id: str = Field(..., min_length=1, description="Public UUID of the existing deck")
    tags: Optional[List[str]] = None

    @field_validator("content")
    @classmethod
    def _content_non_empty(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("content must be a non-empty JSON object")
        return value


class CardUpdate(BaseModel):
    content: Optional[Dict[str, Any]] = Field(None, description="New JSON content")
    deck_id: Optional[str] = Field(None, min_length=1, description="Target deck public UUID")
    tags: Optional[List[str]] = None

    @field_validator("content")
    @classmethod
    def _content_non_empty_update(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is not None and (not isinstance(value, dict) or not value):
            raise ValueError("content must be a non-empty JSON object")
        return value


class CardOut(BaseModel):
    public_id: str
    content: Dict[str, Any]
    deck_id: Optional[str]
    deck_name: Optional[str]
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
    reviewed_card_public_id: str


class CardsPage(BaseModel):
    items: List[CardOut]
    total: int
    limit: int
    offset: int
    next_offset: Optional[int]

class EvalRequest(BaseModel):
    target_text: str = Field(..., description="Frase-alvo em inglês")
    audio_b64: str = Field(..., description="Áudio codificado em base64 (pode ser data URL)")
    phoneme_fmt: Literal["ipa"] = Field("ipa", description="Formato de fonemas suportado: somente 'ipa'")
    # model_repo: str = Field(DEFAULT_MODEL, description="Repositório/modelo mlX Whisper")

class IntelligibilityOut(BaseModel):
    score: float
    word_accuracy_rate: float


class WordAnalysisOut(BaseModel):
    target_word: Optional[str]
    target_phones: List[Any] = Field(default_factory=list)
    produced_phone: List[Any] = Field(default_factory=list)
    per: float
    gop: float
    ops: List[str] = Field(default_factory=list)
    pronunciation_quality: Optional[str]
    audio_b64: Optional[str]


class PhoneticAnalysisOut(BaseModel):
    fluency_level: Optional[str]
    words: List[WordAnalysisOut]


class EvalResponse(BaseModel):
    intelligibility: IntelligibilityOut
    phonetic_analysis: PhoneticAnalysisOut
    meta: Dict[str, Any]
