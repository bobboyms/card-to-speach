#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anki Mini – FastAPI + Decks + Learning Steps (Anki-like) + Learn-Ahead
----------------------------------------------------------------------

• Scheduler: SM-2 (supermemo2 v3 – API funcional: first_review/review)
• Learning Steps: Again/Hard/Good/Easy (minutos) antes da graduação
• Learn-Ahead: adianta learning dentro de uma janela (e fallback se fila vazia)
• Storage: SQLite (arquivo ao lado deste script)
• Tempo: UTC (due_ts em datetime ISO UTC; due em YYYY-MM-DD)

Endpoints (Cards):
  - POST   /cards                              → criar cartão (deck obrigatório)
  - GET    /cards?deck=...&limit=...           → listar cartões de um deck (browse)
  - GET    /cards/due?deck=...&limit&offset    → listar cartões devidos (UTC) com paginação
  - GET    /cards/{card_id}                    → detalhes de um cartão
  - PATCH  /cards/{card_id}                    → atualizar frente/verso/tags e (opcional) mover de deck
  - DELETE /cards/{card_id}                    → remover cartão
  - POST   /cards/{card_id}/review             → revisar um cartão (again/hard/good/easy)

Endpoints (Decks):
  - POST   /decks                              → criar deck (nome)
  - GET    /decks                              → listar decks
  - PATCH  /decks/{name}                       → renomear deck
  - DELETE /decks/{name}                       → deletar deck (cards ficam com deck_name = NULL)

Endpoints (Reviews):
  - GET    /reviews/next?deck=...              → espiar o próximo devido (ordem: devido real → learn-ahead → fallback)
  - POST   /reviews/next?deck=...              → aplicar grade SM-2 no próximo devido (mesma seleção acima)
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# supermemo2 v3 – API funcional
from supermemo2 import first_review, review

# ---------------------------------
# Config (DB e política de learning)
# ---------------------------------
DB = str(Path(__file__).with_name("anki_mini.db"))  # caminho absoluto ao lado do arquivo
MIN_EF = 1.3

# Learning policy (configurável)
LEARNING_STEPS_MIN = [1, 10]    # passos em minutos (ex.: 1m, 10m)
GRADUATE_GOOD_DAYS = 1          # Good no último passo → X dias
GRADUATE_EASY_DAYS = 4          # Easy em qualquer momento → X dias
LEARN_AHEAD_MIN   = 20          # minutos: pode adiantar learning se faltar ≤ X (comportamento Anki)
LEARN_AHEAD_IF_EMPTY = True     # se fila vazia, puxa o learning mais próximo mesmo fora da janela

# --- Normalização de agendamento (apenas para graduados) ---
NORMALIZE_TO_DAY_START = True   # agenda revisões diárias para 00:00 do dia devido
TZ = timezone.utc               # fuso usado na normalização

def _day_start(dt: datetime) -> datetime:
    d = dt.date()
    return datetime(d.year, d.month, d.day, tzinfo=TZ)

def _normalize_due_day(days: int, ref: datetime) -> datetime:
    """00:00 (TZ) do dia que cai em 'ref.date() + days'."""
    due_day = ref.date() + timedelta(days=days)
    return datetime(due_day.year, due_day.month, due_day.day, tzinfo=TZ)

# ---------
# Time (UTC)
# ---------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_today() -> date:
    return utc_now().date()

# ---------------------------
# DB helpers / initialização
# ---------------------------
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON;")
    return c

def _column_exists(c: sqlite3.Connection, table: str, col: str) -> bool:
    cols = c.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in cols)

def init_db():
    with conn() as c:
        # Decks
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS decks (
                name TEXT PRIMARY KEY
            );
            """
        )
        # Cards
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                front TEXT NOT NULL,
                back  TEXT NOT NULL,
                deck_name TEXT REFERENCES decks(name) ON UPDATE CASCADE ON DELETE SET NULL,
                tags  TEXT,
                repetitions INTEGER NOT NULL DEFAULT 0,
                interval    INTEGER NOT NULL DEFAULT 0,
                efactor     REAL    NOT NULL DEFAULT 2.5,
                due         TEXT    NOT NULL,               -- YYYY-MM-DD
                last_reviewed TEXT,
                lapses INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        # Índices
        c.execute("CREATE INDEX IF NOT EXISTS idx_cards_due ON cards(due);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_name);")

        # Migrations: learning flags e due_ts (datetime)
        if not _column_exists(c, 'cards', 'is_learning'):
            c.execute("ALTER TABLE cards ADD COLUMN is_learning INTEGER NOT NULL DEFAULT 1")
        if not _column_exists(c, 'cards', 'learn_step_idx'):
            c.execute("ALTER TABLE cards ADD COLUMN learn_step_idx INTEGER NOT NULL DEFAULT 0")
        if not _column_exists(c, 'cards', 'due_ts'):
            c.execute("ALTER TABLE cards ADD COLUMN due_ts TEXT")
            # backfill de due_ts com meia-noite UTC da data 'due'
            rows = c.execute("SELECT id, due FROM cards").fetchall()
            for r in rows:
                due_dt = datetime.combine(
                    datetime.fromisoformat(r["due"]).date(),
                    datetime.min.time(),
                    tzinfo=timezone.utc
                )
                c.execute("UPDATE cards SET due_ts=? WHERE id=?",
                          (due_dt.isoformat(timespec='seconds'), r["id"]))
        # índice de due_ts
        c.execute("CREATE INDEX IF NOT EXISTS idx_cards_due_ts ON cards(due_ts);")

# -----------------------
# Pydantic Schemas (I/O)
# -----------------------
# Decks
class DeckCreate(BaseModel):
    name: str = Field(..., min_length=1)

class DeckRename(BaseModel):
    new_name: str = Field(..., min_length=1)

class DeckOut(BaseModel):
    name: str

# Cards
class CardCreate(BaseModel):
    front: str = Field(..., min_length=1)
    back: str = Field(..., min_length=1)
    deck: str = Field(..., min_length=1, description="Nome do deck existente")
    tags: Optional[List[str]] = None

class CardUpdate(BaseModel):
    front: Optional[str] = Field(None, min_length=1)
    back: Optional[str] = Field(None, min_length=1)
    deck: Optional[str] = Field(None, min_length=1, description="Novo deck (opcional)")
    tags: Optional[List[str]] = None

class CardOut(BaseModel):
    id: int
    front: str
    back: str
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
    def _grade_range(cls, v: int) -> int:
        if not (0 <= v <= 5):
            raise ValueError("grade deve estar entre 0 e 5")
        return v

class ReviewButtonIn(BaseModel):
    button: Literal['again','hard','good','easy']

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

# ---------------
# FastAPI (app)
# ---------------
@asynccontextmanager
async def lifespan(app: FastAPI):
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
# Helpers (domain logic)
# ------------------------
def _row_to_cardout(row: sqlite3.Row) -> CardOut:
    tags_list = row["tags"].split(",") if row["tags"] else None
    return CardOut(
        id=row["id"],
        front=row["front"],
        back=row["back"],
        deck=row["deck_name"],
        tags=[t for t in (tags_list or []) if t != ""],
        repetitions=row["repetitions"],
        interval=row["interval"],
        efactor=float(row["efactor"]),
        due=row["due"],
        last_reviewed=row["last_reviewed"],
        lapses=row["lapses"],
    )

def _require_deck_exists(deck_name: str):
    with conn() as c:
        r = c.execute("SELECT name FROM decks WHERE name=?", (deck_name,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail=f"Deck '{deck_name}' não existe")

def _get_card(card_id: int) -> sqlite3.Row:
    with conn() as c:
        r = c.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Cartão não encontrado")
    return r

def _set_due(c: sqlite3.Connection, card_id: int, due_dt: datetime):
    due_iso = due_dt.isoformat(timespec='seconds')
    due_date = due_dt.date().isoformat()
    c.execute("UPDATE cards SET due_ts=?, due=? WHERE id=?", (due_iso, due_date, card_id))

# ---------- Learning steps (botões) ----------
BUTTON_TO_GRADE = {"again": 0, "hard": 3, "good": 4, "easy": 5}

def _review_with_button(row: sqlite3.Row, button: str) -> ReviewOut:
    now = utc_now()
    is_learning = int(row["is_learning"]) == 1
    step_idx = int(row["learn_step_idx"])

    if is_learning:
        steps = LEARNING_STEPS_MIN
        with conn() as c:
            if button == 'again':
                step_idx = 0
                _set_due(c, row["id"], now + timedelta(minutes=steps[0]))
            elif button == 'hard':
                step_idx = max(0, min(step_idx, len(steps)-1))
                _set_due(c, row["id"], now + timedelta(minutes=steps[step_idx]))
            elif button == 'good':
                step_idx += 1
                if step_idx < len(steps):
                    _set_due(c, row["id"], now + timedelta(minutes=steps[step_idx]))
                else:
                    # gradua com GOOD → normaliza para início do dia, se habilitado
                    c.execute(
                        "UPDATE cards SET is_learning=0, repetitions=?, interval=?, efactor=? WHERE id=?",
                        (1, GRADUATE_GOOD_DAYS, max(float(row["efactor"]) or 2.5, MIN_EF), row["id"])
                    )
                    next_dt = _normalize_due_day(GRADUATE_GOOD_DAYS, now) if NORMALIZE_TO_DAY_START \
                              else now + timedelta(days=GRADUATE_GOOD_DAYS)
                    _set_due(c, row["id"], next_dt)
            elif button == 'easy':
                # gradua com EASY (pula etapas) → normaliza para início do dia, se habilitado
                c.execute(
                    "UPDATE cards SET is_learning=0, repetitions=?, interval=?, efactor=? WHERE id=?",
                    (1, GRADUATE_EASY_DAYS, max(float(row["efactor"]) or 2.5, MIN_EF), row["id"])
                )
                next_dt = _normalize_due_day(GRADUATE_EASY_DAYS, now) if NORMALIZE_TO_DAY_START \
                          else now + timedelta(days=GRADUATE_EASY_DAYS)
                _set_due(c, row["id"], next_dt)

            # persistir índice do passo e last_reviewed
            c.execute(
                "UPDATE cards SET learn_step_idx=?, last_reviewed=? WHERE id=?",
                (min(step_idx, max(0, len(steps)-1)), now.isoformat(timespec='seconds'), row["id"])
            )
            updated = c.execute("SELECT * FROM cards WHERE id=?", (row["id"],)).fetchone()

        due_dt = datetime.fromisoformat(updated["due_ts"]) if updated["due_ts"] else now
        return ReviewOut(
            card=_row_to_cardout(updated),
            next_due=updated["due"],
            interval_days=max(0, (due_dt.date() - utc_today()).days),
            efactor=float(updated["efactor"]),
            repetitions=int(updated["repetitions"]),
            lapses=int(updated["lapses"]),
            reviewed_card_id=int(row["id"])
        )

    # Graduado → SM-2
    q = BUTTON_TO_GRADE[button]
    return _apply_review(row, q)

# ---------- SM-2 (grades) ----------
def _apply_review(row: sqlite3.Row, grade: int) -> ReviewOut:
    """Aplica revisão SM-2 (supermemo2 v3). Usa due_ts para agendar."""
    if row["repetitions"] == 0 and row["interval"] == 0:
        r = first_review(grade)
    else:
        r = review(grade, float(row["efactor"]), int(row["interval"]), int(row["repetitions"]))

    new_interval = int(r["interval"])      # dias
    new_reps = int(r["repetitions"])
    new_ef = max(float(r["easiness"]), MIN_EF)

    now = utc_now()
    if grade <= 2:
        next_dt = now  # reforço imediato (mesmo dia)
        lapses = (row["lapses"] or 0) + 1
    else:
        lapses = row["lapses"] or 0
        next_dt = _normalize_due_day(new_interval, now) if NORMALIZE_TO_DAY_START \
                  else now + timedelta(days=new_interval)

    with conn() as c:
        c.execute(
            """
            UPDATE cards
            SET repetitions=?, interval=?, efactor=?, last_reviewed=?, lapses=?
            WHERE id=?
            """,
            (new_reps, new_interval, new_ef, (r.get("review_datetime") or now.isoformat(timespec='seconds')), lapses, row["id"])
        )
        _set_due(c, row["id"], next_dt)
        updated = c.execute("SELECT * FROM cards WHERE id=?", (row["id"],)).fetchone()

    return ReviewOut(
        card=_row_to_cardout(updated),
        next_due=updated["due"],
        interval_days=new_interval,
        efactor=new_ef,
        repetitions=new_reps,
        lapses=int(lapses),
        reviewed_card_id=int(row["id"]),
    )

# ------------------------
# Endpoints – Decks
# ------------------------
@app.post("/decks", response_model=DeckOut, status_code=201)
def create_deck(payload: DeckCreate):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nome do deck vazio")
    with conn() as c:
        exists = c.execute("SELECT 1 FROM decks WHERE name=?", (name,)).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="Deck já existe")
        c.execute("INSERT INTO decks(name) VALUES(?)", (name,))
    return DeckOut(name=name)

@app.get("/decks", response_model=List[DeckOut])
def list_decks():
    with conn() as c:
        rows = c.execute("SELECT name FROM decks ORDER BY name").fetchall()
    return [DeckOut(name=r["name"]) for r in rows]

@app.patch("/decks/{name}", response_model=DeckOut)
def rename_deck(name: str, payload: DeckRename):
    new_name = payload.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Novo nome inválido")
    with conn() as c:
        cur = c.execute("UPDATE decks SET name=? WHERE name=?", (new_name, name))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Deck não encontrado")
    return DeckOut(name=new_name)

@app.delete("/decks/{name}", status_code=204)
def delete_deck(name: str):
    with conn() as c:
        cur = c.execute("DELETE FROM decks WHERE name=?", (name,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Deck não encontrado")
    return

# ------------------------
# Endpoints – Cards
# ------------------------
@app.post("/cards", response_model=CardOut, status_code=201)
def create_card(payload: CardCreate):
    deck_name = payload.deck.strip()
    _require_deck_exists(deck_name)
    now = utc_now()
    tags_csv = ",".join(t.strip() for t in (payload.tags or []) if t.strip()) or None
    with conn() as c:
        cur = c.execute(
            """
            INSERT INTO cards(front, back, deck_name, tags, repetitions, interval, efactor, due,
                              last_reviewed, lapses, is_learning, learn_step_idx, due_ts)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                payload.front,
                payload.back,
                deck_name,
                tags_csv,
                0, 0, 2.5,
                now.date().isoformat(),
                None,
                0,
                1,    # is_learning
                0,    # learn_step_idx
                now.isoformat(timespec='seconds'),  # due_ts = agora (entra na fila já)
            ),
        )
        new_id = cur.lastrowid
        row = c.execute("SELECT * FROM cards WHERE id=?", (new_id,)).fetchone()
    return _row_to_cardout(row)

@app.get("/cards", response_model=List[CardOut])
def list_cards(
    deck: str = Query(..., description="Nome do deck"),
    limit: int = Query(50, ge=1, le=500),
):
    _require_deck_exists(deck)
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM cards WHERE deck_name = ? ORDER BY due_ts, id LIMIT ?",
            (deck, limit),
        ).fetchall()
    return [_row_to_cardout(r) for r in rows]

@app.get("/cards/due", response_model=CardsPage)
def list_due(
    deck: str = Query(..., description="Nome do deck"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    _require_deck_exists(deck)
    now_iso = utc_now().isoformat(timespec='seconds')
    with conn() as c:
        total = c.execute(
            "SELECT COUNT(*) AS c FROM cards WHERE due_ts <= ? AND deck_name = ?",
            (now_iso, deck),
        ).fetchone()[0]
        rows = c.execute(
            "SELECT * FROM cards WHERE due_ts <= ? AND deck_name = ? ORDER BY due_ts, id LIMIT ? OFFSET ?",
            (now_iso, deck, limit, offset),
        ).fetchall()
    items = [_row_to_cardout(r) for r in rows]
    next_off = offset + limit if offset + limit < total else None
    return CardsPage(items=items, total=int(total), limit=limit, offset=offset, next_offset=next_off)

@app.get("/cards/{card_id}", response_model=CardOut)
def get_card(card_id: int):
    return _row_to_cardout(_get_card(card_id))

@app.patch("/cards/{card_id}", response_model=CardOut)
def update_card(card_id: int, payload: CardUpdate):
    row = _get_card(card_id)
    new_front = payload.front if payload.front is not None else row["front"]
    new_back  = payload.back  if payload.back  is not None else row["back"]
    if payload.deck is not None:
        deck_name = payload.deck.strip()
        _require_deck_exists(deck_name)
        new_deck = deck_name
    else:
        new_deck = row["deck_name"]
    if payload.tags is None:
        new_tags = row["tags"]
    else:
        new_tags = ",".join(t.strip() for t in payload.tags if t.strip()) or None

    with conn() as c:
        c.execute(
            "UPDATE cards SET front=?, back=?, deck_name=?, tags=? WHERE id=?",
            (new_front, new_back, new_deck, new_tags, card_id),
        )
        updated = c.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
    return _row_to_cardout(updated)

@app.delete("/cards/{card_id}", status_code=204)
def delete_card(card_id: int):
    _get_card(card_id)  # 404 se não existir
    with conn() as c:
        c.execute("DELETE FROM cards WHERE id=?", (card_id,))
    return

# ------------------------
# Endpoints – Reviews
# ------------------------
@app.get("/reviews/next", response_model=CardOut)
def peek_next_due(deck: str = Query(..., description="Nome do deck")):
    """Retorna o próximo cartão devido do deck.
    Ordem:
      1) due_ts <= agora (devidos reais)
      2) is_learning=1 AND due_ts <= agora + LEARN_AHEAD_MIN (learn-ahead)
      3) (opcional) se nada encontrado e LEARN_AHEAD_IF_EMPTY, pega o learning mais próximo
    """
    _require_deck_exists(deck)
    now = utc_now()
    now_iso = now.isoformat(timespec='seconds')
    ahead_iso = (now + timedelta(minutes=LEARN_AHEAD_MIN)).isoformat(timespec='seconds')
    with conn() as c:
        # 1) devidos reais
        row = c.execute(
            "SELECT * FROM cards WHERE due_ts <= ? AND deck_name = ? ORDER BY due_ts, id LIMIT 1",
            (now_iso, deck),
        ).fetchone()
        if not row:
            # 2) learn-ahead (janela)
            row = c.execute(
                """
                SELECT * FROM cards
                WHERE is_learning = 1 AND deck_name = ? AND due_ts <= ?
                ORDER BY due_ts, id LIMIT 1
                """,
                (deck, ahead_iso),
            ).fetchone()
        if not row and LEARN_AHEAD_IF_EMPTY:
            # 3) fallback: puxa learning mais próximo mesmo fora da janela
            row = c.execute(
                "SELECT * FROM cards WHERE is_learning = 1 AND deck_name = ? ORDER BY due_ts, id LIMIT 1",
                (deck,),
            ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Nenhum cartão devido agora")
    return _row_to_cardout(row)

@app.post("/reviews/next", response_model=ReviewOut)
def review_next_due_grade(deck: str = Query(..., description="Nome do deck"), payload: ReviewIn = ...):
    """Aplica grade SM-2 no próximo cartão do deck (mesma seleção do GET)."""
    _require_deck_exists(deck)
    now = utc_now()
    now_iso = now.isoformat(timespec='seconds')
    ahead_iso = (now + timedelta(minutes=LEARN_AHEAD_MIN)).isoformat(timespec='seconds')
    with conn() as c:
        row = c.execute(
            "SELECT * FROM cards WHERE due_ts <= ? AND deck_name = ? ORDER BY due_ts, id LIMIT 1",
            (now_iso, deck),
        ).fetchone()
        if not row:
            row = c.execute(
                """
                SELECT * FROM cards
                WHERE is_learning = 1 AND deck_name = ? AND due_ts <= ?
                ORDER BY due_ts, id LIMIT 1
                """,
                (deck, ahead_iso),
            ).fetchone()
        if not row and LEARN_AHEAD_IF_EMPTY:
            row = c.execute(
                "SELECT * FROM cards WHERE is_learning = 1 AND deck_name = ? ORDER BY due_ts, id LIMIT 1",
                (deck,),
            ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Nenhum cartão devido agora")
    return _apply_review(row, payload.grade)

@app.post("/cards/{card_id}/review", response_model=ReviewOut)
def review_card_button(card_id: int, payload: ReviewButtonIn = Body(...)):
    """Revisar um cartão específico usando botões (again/hard/good/easy)."""
    row = _get_card(card_id)
    return _review_with_button(row, payload.button)

# ------------------------
# Healthcheck
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok", "utc": utc_now().isoformat()}

# ------------------------
# Execução local
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
