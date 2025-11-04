#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anki Mini – FastAPI + Decks + Learning Steps (Anki-like) + Learn-Ahead
----------------------------------------------------------------------

• Scheduler: SM-2 (supermemo2 v3 – functional API: first_review/review)
• Learning Steps: Again/Hard/Good/Easy (minutes) before graduation
• Learn-Ahead: bring learning cards forward within a window (with fallback if the queue is empty)
• Storage: SQLite (database file lives next to this script)
• Timezone: UTC (due_ts stored as ISO datetime; due stored as YYYY-MM-DD)

Endpoints (Cards):
  - POST   /cards                              → create card (deck UUID required, JSON content)
  - GET    /cards?deck_id=...&limit=...        → list deck cards (browse)
  - GET    /cards/due?deck_id=...&limit&offset → list due cards (UTC) with pagination
  - GET    /cards/{card_public_id}             → card details
  - PATCH  /cards/{card_public_id}             → update content/tags and optionally move deck
  - DELETE /cards/{card_public_id}             → delete card
  - POST   /cards/{card_public_id}/review      → review a specific card (again/hard/good/easy buttons)

Endpoints (Decks):
  - POST   /decks                              → create deck (name)
  - GET    /decks                              → list decks
  - PATCH  /decks/{name}                       → rename deck
  - DELETE /decks/{name}                       → delete deck (existing cards keep deck_name = NULL)

Endpoints (Reviews):
  - GET    /reviews/next?deck=...              → peek next due card (order: real due → learn-ahead → fallback)
  - POST   /reviews/next?deck=...              → apply SM-2 grade to the next due card (same selection as GET)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import Body, FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config, time_utils
from app.db import db_manager
from app.services.evaluate import evaluate_pronunciation
from app.repositories import CardRepository, DeckRepository
from app.schemas import (
    CardCreate,
    CardOut,
    CardUpdate,
    CardsPage,
    DeckCreate,
    DeckOut,
    DeckRename,
    ReviewButtonIn,
    ReviewIn,
    ReviewOut, EvalResponse, EvalRequest,
)
from app.services.other_services import DeckService, CardService, ReviewService
from app.utils.b64 import b64_to_temp_audio_file

# ---------------------------------
# Configuration (database and learning policy)
# ---------------------------------
DB = config.DB_PATH
MIN_EF = config.MIN_EF

# Learning policy (tunable)
LEARNING_STEPS_MIN: List[int] = config.LEARNING_STEPS_MIN
GRADUATE_GOOD_DAYS: int = config.GRADUATE_GOOD_DAYS
GRADUATE_EASY_DAYS: int = config.GRADUATE_EASY_DAYS
LEARN_AHEAD_MIN: int = config.LEARN_AHEAD_MIN
LEARN_AHEAD_IF_EMPTY: bool = config.LEARN_AHEAD_IF_EMPTY

# --- Scheduling normalization (for graduated cards) ---
NORMALIZE_TO_DAY_START: bool = config.NORMALIZE_TO_DAY_START
TZ = config.TIMEZONE

# Time helpers re-exported for backwards compatibility with tests
utc_now = time_utils.utc_now
utc_today = time_utils.utc_today
_normalize_due_day = time_utils.normalize_due_day


def init_db() -> None:
    """Initialize database schema using the current database path."""
    db_manager.set_path(DB)
    db_manager.initialize()


# Instantiate repositories
deck_repository = DeckRepository(db_manager)
card_repository = CardRepository(db_manager)


# Instantiate services with dynamic providers so monkeypatching on this module still works
deck_service = DeckService(deck_repository)
card_service = CardService(
    card_repository,
    deck_service,
    lambda: utc_now(),
)
review_service = ReviewService(
    card_repository,
    card_service,
    deck_service,
    lambda: utc_now(),
    lambda: utc_today(),
    lambda days, ref: _normalize_due_day(days, ref),
    lambda: LEARNING_STEPS_MIN,
    lambda: GRADUATE_GOOD_DAYS,
    lambda: GRADUATE_EASY_DAYS,
    lambda: NORMALIZE_TO_DAY_START,
    lambda: LEARN_AHEAD_MIN,
    lambda: LEARN_AHEAD_IF_EMPTY,
    lambda: MIN_EF,
)


# ---------------
# FastAPI (app)
# ---------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook: ensure database is initialized before serving requests."""
    init_db()
    yield


app = FastAPI(title="Anki Mini API", version="3.3 (normalize day start)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------
# Endpoints – Decks
# ------------------------
@app.post("/decks", response_model=DeckOut, status_code=201)
def create_deck(payload: DeckCreate):
    """Create a new deck with an explicit type ('speach' or 'shadowing')."""
    return deck_service.create(payload)


@app.get("/decks", response_model=List[DeckOut])
def list_decks():
    """List all decks."""
    return deck_service.list()


@app.patch("/decks/{public_id}", response_model=DeckOut)
def rename_deck(public_id: str, payload: DeckRename):
    """Rename an existing deck identified by its public UUID."""
    return deck_service.rename(public_id, payload)


@app.delete("/decks/{public_id}", status_code=204)
def delete_deck(public_id: str):
    """Delete the specified deck by its public UUID."""
    deck_service.delete(public_id)
    return


# ------------------------
# Endpoints – Cards
# ------------------------
@app.post("/cards", response_model=CardOut, status_code=201)
def create_card(payload: CardCreate):
    """Create a card in learning mode."""
    return card_service.create(payload)


@app.get("/cards", response_model=List[CardOut])
def list_cards(
    deck_id: str = Query(..., description="Deck public UUID"),
    limit: int = Query(50, ge=1, le=500),
):
    """List cards for a deck ordered by due timestamp."""
    return card_service.list_by_deck(deck_id, limit)


@app.get("/cards/due", response_model=CardsPage)
def list_due(
    deck_id: str = Query(..., description="Deck public UUID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return paginated due cards for a deck."""
    return card_service.list_due(deck_id, limit, offset)


@app.get("/cards/{card_public_id}", response_model=CardOut)
def get_card(card_public_id: str):
    """Fetch a card by public UUID."""
    return card_service.get(card_public_id)


@app.patch("/cards/{card_public_id}", response_model=CardOut)
def update_card(card_public_id: str, payload: CardUpdate):
    """Update the content, deck, or tags of a card."""
    return card_service.update(card_public_id, payload)


@app.delete("/cards/{card_public_id}", status_code=204)
def delete_card(card_public_id: str):
    """Remove a card from the collection."""
    card_service.delete(card_public_id)
    return


# ------------------------
# Endpoints – Reviews
# ------------------------
@app.get("/reviews/next", response_model=CardOut)
def peek_next_due(deck_id: str = Query(..., description="Deck public UUID")):
    """Return the next due card for the requested deck using due, learn-ahead, and fallback rules."""
    return review_service.peek_next_due(deck_id)


@app.post("/reviews/next", response_model=ReviewOut)
def review_next_due_grade(deck_id: str = Query(..., description="Deck public UUID"), payload: ReviewIn = ...):
    """Apply an SM-2 grade to the next due card (same selection criteria as the GET endpoint)."""
    return review_service.review_next_due_by_grade(deck_id, payload.grade)


@app.post("/cards/{card_public_id}/review", response_model=ReviewOut)
def review_card_button(card_public_id: str, payload: ReviewButtonIn = Body(...)):
    """Review a specific card by mapping the button press to the corresponding SM-2 grade."""
    return review_service.review_card_button(card_public_id, payload.button)


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    if req.phoneme_fmt not in {"ipa", "ascii", "arpabet"}:
        raise HTTPException(status_code=400, detail="phoneme_fmt deve ser 'ipa', 'ascii' ou 'arpabet'")
    # Decodificar base64 -> arquivo temporário
    audio_path = b64_to_temp_audio_file(req.audio_b64)
    try:
        raw_results = evaluate_pronunciation(
            audio_path=audio_path,
            target_text=req.target_text,
            phoneme_fmt=req.phoneme_fmt,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha na avaliação: {e}")
    finally:
        # limpe o temporário
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return _format_evaluation_response(raw_results, req.phoneme_fmt)

def _format_evaluation_response(results: Dict[str, Any], phoneme_fmt: str) -> Dict[str, Any]:
    """Adapt the internal evaluation payload to the public API schema."""
    words: List[Dict[str, Any]] = list(results.get("words") or [])
    intelligibility_score = float(results.get("intelligibility", 0.0))
    word_accuracy_rate = float(results.get("word_accuracy_rate", 0.0))
    fluency_level = results.get("fluency_level")

    return {
        "intelligibility": {
            "score": intelligibility_score,
            "word_accuracy_rate": word_accuracy_rate,
        },
        "phonetic_analysis": {
            "fluency_level": fluency_level,
            "words": words,
        },
        "meta": {
            "phoneme_fmt": phoneme_fmt,
        },
    }


# ------------------------
# Healthcheck
# ------------------------
@app.get("/health")
def health():
    """Simple health check endpoint with current UTC timestamp."""
    return {"status": "ok", "utc": utc_now().isoformat()}


# ------------------------
# Local execution
# ------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
