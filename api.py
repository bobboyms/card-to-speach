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

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List

from fastapi import Body, Depends, FastAPI, HTTPException, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
from starlette.responses import StreamingResponse

from app import config, time_utils
from app.db import db_manager
from app.services.chat_service import ChatService
from app.services.evaluate import evaluate_pronunciation, format_eval_response
from app.repositories import CardRepository, DeckRepository, UserRepository, AuthRepository
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
    ReviewIn,
    ReviewOut, EvalResponse, EvalRequest, AudioB64,
    GoogleAuthRequest, Token, UserOut, UserUpdate,
)
from app.services.other_services import DeckService, CardService, ReviewService
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.text_to_speech import TextToSpeach
from app.utils.b64 import b64_to_temp_audio_file, mp3_to_base64

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

load_dotenv()

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Initialize database schema using the current database path."""
    db_manager.set_path(DB)
    db_manager.initialize()


def get_deck_repository() -> DeckRepository:
    return DeckRepository(db_manager)


def get_card_repository() -> CardRepository:
    return CardRepository(db_manager)


def get_deck_service(
    repo: DeckRepository = Depends(get_deck_repository),
) -> DeckService:
    return DeckService(repo)

def get_text_to_speech_service(
) -> TextToSpeach:
    return TextToSpeach()

def get_card_service(
    repo: CardRepository = Depends(get_card_repository),
    deck_service: DeckService = Depends(get_deck_service),
    text_to_speech_service: TextToSpeach = Depends(get_text_to_speech_service)
) -> CardService:
    return CardService(
        repo,
        deck_service,
        text_to_speech_service,
        utc_now,
    )


def get_review_service(
    card_repo: CardRepository = Depends(get_card_repository),
    card_service: CardService = Depends(get_card_service),
    deck_service: DeckService = Depends(get_deck_service),
) -> ReviewService:
    return ReviewService(
        card_repo,
        card_service,
        deck_service,
        utc_now,
        utc_today,
        _normalize_due_day,
        lambda: LEARNING_STEPS_MIN,
        lambda: GRADUATE_GOOD_DAYS,
        lambda: GRADUATE_EASY_DAYS,
        lambda: NORMALIZE_TO_DAY_START,
        lambda: LEARN_AHEAD_MIN,
        lambda: LEARN_AHEAD_IF_EMPTY,
        lambda: MIN_EF,
    )


def get_chat_services() -> ChatService:
    return ChatService()

def get_evaluate_pronunciation() -> Callable[[str, str, str], Dict[str, Any]]:
    return evaluate_pronunciation


def get_user_repository() -> UserRepository:
    return UserRepository(db_manager)


def get_user_service(
    repo: UserRepository = Depends(get_user_repository),
) -> UserService:
    return UserService(repo)


def get_auth_repository() -> AuthRepository:
    return AuthRepository(db_manager)


def get_auth_service(
    user_service: UserService = Depends(get_user_service),
    auth_repository: AuthRepository = Depends(get_auth_repository),
) -> AuthService:
    return AuthService(user_service, auth_repository)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/google", auto_error=False)


async def verify_auth_global(
    request: Request,
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
):
    # List of paths that do not require authentication
    allowed_paths = [
        "/auth/google",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]

    if request.url.path in allowed_paths:
        return

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        if auth_service.is_token_revoked(token):
            raise ValueError("Token has been revoked")
        auth_service.verify_jwt(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_id(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> str:
    """Extract and return the user_id from the JWT token."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        if auth_service.is_token_revoked(token):
            raise ValueError("Token has been revoked")
        payload = auth_service.verify_jwt(token)
        user_id = payload.get("user_id")
        if not user_id:
            raise ValueError("Token does not contain user_id")
        return user_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------
# FastAPI (app)
# ---------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook: ensure database is initialized before serving requests."""
    init_db()
    yield


app = FastAPI(
    title="Anki Mini API",
    version="3.3 (normalize day start)",
    lifespan=lifespan,
    dependencies=[Depends(verify_auth_global)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------
# Endpoints – Decks
# ------------------------
@app.post("/decks", response_model=DeckOut, status_code=201)
def create_deck(
    payload: DeckCreate,
    deck_service: DeckService = Depends(get_deck_service),
    user_id: str = Depends(get_current_user_id),
):
    """Create a new deck with an explicit type ('speech' or 'shadowing')."""
    return deck_service.create(payload, user_id)


@app.get("/decks", response_model=List[DeckOut])
def list_decks(
    deck_service: DeckService = Depends(get_deck_service),
    user_id: str = Depends(get_current_user_id),
):
    """List all decks."""
    return deck_service.list(user_id)


@app.patch("/decks/{public_id}", response_model=DeckOut)
def rename_deck(
    public_id: str,
    payload: DeckRename,
    deck_service: DeckService = Depends(get_deck_service),
):
    """Rename an existing deck identified by its public UUID."""
    return deck_service.rename(public_id, payload)


@app.delete("/decks/{public_id}", status_code=204)
def delete_deck(
    public_id: str,
    deck_service: DeckService = Depends(get_deck_service),
):
    """Delete the specified deck by its public UUID."""
    deck_service.delete(public_id)
    return


# ------------------------
# Endpoints – Cards
# ------------------------
@app.post("/cards", response_model=CardOut, status_code=201)
def create_card(
    payload: CardCreate,
    card_service: CardService = Depends(get_card_service),
    user_id: str = Depends(get_current_user_id),
):
    """Create a card in learning mode."""
    return card_service.create(payload, user_id)


@app.get("/cards", response_model=List[CardOut])
def list_cards(
    deck_id: str = Query(..., description="Deck public UUID"),
    limit: int = Query(50, ge=1, le=500),
    card_service: CardService = Depends(get_card_service),
    user_id: str = Depends(get_current_user_id),
):
    """List cards for a deck ordered by due timestamp."""
    return card_service.list_by_deck(deck_id, limit, user_id)


@app.get("/cards/due", response_model=CardsPage)
def list_due(
    deck_id: str = Query(..., description="Deck public UUID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    card_service: CardService = Depends(get_card_service),
    user_id: str = Depends(get_current_user_id),
):
    """Return paginated due cards for a deck."""
    return card_service.list_due(deck_id, limit, offset, user_id)


@app.get("/cards/{card_public_id}", response_model=CardOut)
def get_card(
    card_public_id: str,
    card_service: CardService = Depends(get_card_service),
):
    """Fetch a card by public UUID."""
    return card_service.get(card_public_id)


@app.patch("/cards/{card_public_id}", response_model=CardOut)
def update_card(
    card_public_id: str,
    payload: CardUpdate,
    card_service: CardService = Depends(get_card_service),
):
    """Update the content, deck, or tags of a card."""
    return card_service.update(card_public_id, payload)


@app.delete("/cards/{card_public_id}", status_code=204)
def delete_card(
    card_public_id: str,
    card_service: CardService = Depends(get_card_service),
):
    """Remove a card from the collection."""
    card_service.delete(card_public_id)
    return

@app.get("/audio/{audio_id}", response_model=AudioB64)
def get_audio(
    audio_id: str,
):
    # Basic validation to prevent path traversal
    if ".." in audio_id or "/" in audio_id or "\\" in audio_id:
        raise HTTPException(status_code=400, detail="Invalid audio ID")
        
    b64 = mp3_to_base64(str("temp_files/" + audio_id))
    return AudioB64(
        audio_id=audio_id,
        b64=b64,
    )

# ------------------------
# Endpoints – Reviews
# ------------------------
@app.get("/reviews/next", response_model=CardOut)
def peek_next_due(
    deck_id: str = Query(..., description="Deck public UUID"),
    review_service: ReviewService = Depends(get_review_service),
    user_id: str = Depends(get_current_user_id),
):
    """Return the next due card for the requested deck using due, learn-ahead, and fallback rules."""
    return review_service.peek_next_due(deck_id, user_id)


@app.post("/reviews/next", response_model=ReviewOut)
def review_next_due_grade(
    deck_id: str = Query(..., description="Deck public UUID"),
    payload: ReviewIn = ...,
    review_service: ReviewService = Depends(get_review_service),
    user_id: str = Depends(get_current_user_id),
):
    """Apply an SM-2 grade to the next due card (same selection criteria as the GET endpoint)."""
    return review_service.review_next_due_by_grade(deck_id, payload.grade, user_id)


@app.post("/cards/{card_public_id}/review", response_model=ReviewOut)
def review_card_button(
    card_public_id: str,
    payload: ReviewButtonIn = Body(...),
    review_service: ReviewService = Depends(get_review_service),
):
    """Review a specific card by mapping the button press to the corresponding SM-2 grade."""
    return review_service.review_card_button(card_public_id, payload.button)


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(
    req: EvalRequest,
    evaluate_fn: Callable[[str, str, str], Dict[str, Any]] = Depends(get_evaluate_pronunciation),
):
    # Decodificar base64 -> arquivo temporário
    audio_path = b64_to_temp_audio_file(req.audio_b64)
    print(f"Audio path: {audio_path}")
    try:
        raw_results = evaluate_fn(
            audio_path=audio_path,
            target_text=req.target_text,
            phoneme_fmt=req.phoneme_fmt,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("Pronunciation evaluation failed due to runtime error")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during pronunciation evaluation")
        raise HTTPException(status_code=500, detail="Falha interna ao avaliar a pronúncia") from exc
    finally:
        # limpe o temporário
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return format_eval_response(raw_results, req.phoneme_fmt)


@app.post("/chat-stream")
def chat_stream(
        payload: dict = Body(...),
        chat_service: ChatService = Depends(get_chat_services),
        user_id: str = Depends(get_current_user_id),
):
    """
    Recebe:
      {
        "history": [{"role": "user"|"assistant", "content": "..."}],
        "user_message": "texto"
      }
    Responde com um stream de texto (chunks) da resposta final.
    """
    history = payload.get("history", [])
    user_message = payload.get("user_message", "")

    return StreamingResponse(
        chat_service.generate_answer_stream(history, user_message, user_id),
        media_type="text/plain",
    )


# ------------------------
# Endpoints – Auth
# ------------------------
@app.post("/auth/google", response_model=Token)
def login_google(
    payload: GoogleAuthRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Receives a Google ID token, verifies it, creates/retrieves the user,
    and returns an application JWT.
    """
    try:
        google_user = auth_service.verify_google_token(payload.token)
        user = auth_service.get_or_create_user(google_user)
        access_token = auth_service.create_access_token(
            data={"sub": user.public_id, "user_id": user.public_id, "email": user.email}
        )

        return {"access_token": access_token, "token_type": "bearer"}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/users", response_model=UserOut)
def get_user_info(
    user_id: str = Depends(get_current_user_id),
    user_service: UserService = Depends(get_user_service),
):
    """
    Returns information about the currently authenticated user.
    The user ID is extracted from the JWT token.
    """
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/users", response_model=UserOut)
def update_user_info(
    payload: UserUpdate,
    user_id: str = Depends(get_current_user_id),
    user_service: UserService = Depends(get_user_service),
):
    """
    Updates information for the currently authenticated user (partial update).
    The user ID is extracted from the JWT token.
    """
    updated_user = user_service.update_user(user_id, payload)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user


@app.post("/logout", status_code=204)
def logout(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Invalidate the current token by adding it to the blacklist.
    """
    auth_service.revoke_token(token)
    return

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

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
