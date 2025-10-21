"""Repository layer encapsulating raw database interactions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple
import sqlite3

from .db import DatabaseManager


class DeckRepository:
    """Persistence layer responsible for deck CRUD operations."""

    def __init__(self, db: DatabaseManager):
        self._db = db

    def exists(self, name: str) -> bool:
        return self.find_by_name(name) is not None

    def find_by_name(self, name: str, *, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
        query = "SELECT name FROM decks WHERE name=?"
        if conn is None:
            with self._db.connect() as connection:
                return connection.execute(query, (name,)).fetchone()
        return conn.execute(query, (name,)).fetchone()

    def insert(self, name: str) -> None:
        with self._db.connect() as connection:
            connection.execute("INSERT INTO decks(name) VALUES(?)", (name,))

    def list_all(self) -> List[sqlite3.Row]:
        with self._db.connect() as connection:
            return connection.execute("SELECT name FROM decks ORDER BY name").fetchall()

    def update_name(self, current: str, new_name: str) -> int:
        with self._db.connect() as connection:
            cursor = connection.execute("UPDATE decks SET name=? WHERE name=?", (new_name, current))
        return cursor.rowcount

    def delete(self, name: str) -> int:
        with self._db.connect() as connection:
            cursor = connection.execute("DELETE FROM decks WHERE name=?", (name,))
        return cursor.rowcount


class CardRepository:
    """Persistence layer responsible for card CRUD operations and review state updates."""

    def __init__(self, db: DatabaseManager):
        self._db = db

    @contextmanager
    def transaction(self):
        with self._db.connect() as connection:
            yield connection

    def insert(
        self,
        *,
        content_json: str,
        deck_name: str,
        tags_csv: Optional[str],
        repetitions: int,
        interval: int,
        efactor: float,
        due_date: str,
        last_reviewed: Optional[str],
        lapses: int,
        is_learning: int,
        learn_step_idx: int,
        due_ts: str,
    ) -> sqlite3.Row:
        with self._db.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO cards(content, deck_name, tags, repetitions, interval, efactor, due,
                                  last_reviewed, lapses, is_learning, learn_step_idx, due_ts)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    content_json,
                    deck_name,
                    tags_csv,
                    repetitions,
                    interval,
                    efactor,
                    due_date,
                    last_reviewed,
                    lapses,
                    is_learning,
                    learn_step_idx,
                    due_ts,
                ),
            )
            return connection.execute("SELECT * FROM cards WHERE id=?", (cursor.lastrowid,)).fetchone()

    def list_by_deck(self, deck: str, limit: int) -> List[sqlite3.Row]:
        with self._db.connect() as connection:
            return connection.execute(
                "SELECT * FROM cards WHERE deck_name = ? ORDER BY due_ts, id LIMIT ?",
                (deck, limit),
            ).fetchall()

    def list_due(self, deck: str, now_iso: str, limit: int, offset: int) -> Tuple[int, List[sqlite3.Row]]:
        with self._db.connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS c FROM cards WHERE due_ts <= ? AND deck_name = ?",
                (now_iso, deck),
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM cards WHERE due_ts <= ? AND deck_name = ? ORDER BY due_ts, id LIMIT ? OFFSET ?",
                (now_iso, deck, limit, offset),
            ).fetchall()
        return int(total), rows

    def fetch_by_id(self, card_id: int, *, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
        query = "SELECT * FROM cards WHERE id=?"
        if conn is None:
            with self._db.connect() as connection:
                return connection.execute(query, (card_id,)).fetchone()
        return conn.execute(query, (card_id,)).fetchone()

    def update_card(self, card_id: int, *, content: str, deck_name: Optional[str], tags: Optional[str]) -> sqlite3.Row:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE cards SET content=?, deck_name=?, tags=? WHERE id=?",
                (content, deck_name, tags, card_id),
            )
            return connection.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()

    def delete(self, card_id: int) -> int:
        with self._db.connect() as connection:
            cursor = connection.execute("DELETE FROM cards WHERE id=?", (card_id,))
        return cursor.rowcount

    def set_due(self, conn: sqlite3.Connection, card_id: int, due_dt: datetime) -> None:
        due_iso = due_dt.isoformat(timespec="seconds")
        due_date = due_dt.date().isoformat()
        conn.execute(
            "UPDATE cards SET due_ts=?, due=? WHERE id=?",
            (due_iso, due_date, card_id),
        )

    def graduate_card(self, conn: sqlite3.Connection, card_id: int, repetitions: int, interval: int, efactor: float) -> None:
        conn.execute(
            "UPDATE cards SET is_learning=0, repetitions=?, interval=?, efactor=? WHERE id=?",
            (repetitions, interval, efactor, card_id),
        )

    def update_learning_progress(self, conn: sqlite3.Connection, card_id: int, learn_step_idx: int, last_reviewed_iso: str) -> None:
        conn.execute(
            "UPDATE cards SET learn_step_idx=?, last_reviewed=? WHERE id=?",
            (learn_step_idx, last_reviewed_iso, card_id),
        )

    def update_review_metrics(
        self,
        conn: sqlite3.Connection,
        card_id: int,
        repetitions: int,
        interval: int,
        efactor: float,
        last_reviewed_iso: str,
        lapses: int,
    ) -> None:
        conn.execute(
            """
            UPDATE cards
            SET repetitions=?, interval=?, efactor=?, last_reviewed=?, lapses=?
            WHERE id=?
            """,
            (repetitions, interval, efactor, last_reviewed_iso, lapses, card_id),
        )

    def select_next_for_review(
        self,
        deck: str,
        now_iso: str,
        ahead_iso: str,
        allow_learning_fallback: bool,
    ) -> Optional[sqlite3.Row]:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM cards WHERE due_ts <= ? AND deck_name = ? ORDER BY due_ts, id LIMIT 1",
                (now_iso, deck),
            ).fetchone()
            if row:
                return row
            row = connection.execute(
                """
                SELECT * FROM cards
                WHERE is_learning = 1 AND deck_name = ? AND due_ts <= ?
                ORDER BY due_ts, id LIMIT 1
                """,
                (deck, ahead_iso),
            ).fetchone()
            if row:
                return row
            if allow_learning_fallback:
                row = connection.execute(
                    "SELECT * FROM cards WHERE is_learning = 1 AND deck_name = ? ORDER BY due_ts, id LIMIT 1",
                    (deck,),
                ).fetchone()
            return row
