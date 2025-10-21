"""Database connection helpers and schema management."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from .config import DB_PATH


class DatabaseManager:
    """Manage SQLite connections and schema lifecycle for the application."""

    def __init__(self, db_path: str):
        """Store the initial database path."""
        self._db_path = db_path

    def set_path(self, db_path: str) -> None:
        """Update the database path (used by tests to point to temporary files)."""
        self._db_path = db_path

    def connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection with foreign keys enforced."""
        conn_obj = sqlite3.connect(self._db_path)
        conn_obj.row_factory = sqlite3.Row
        conn_obj.execute("PRAGMA foreign_keys = ON;")
        return conn_obj

    def initialize(self) -> None:
        """Ensure all tables, columns, and indexes required by the app are present."""
        with self.connect() as connection:
            self._ensure_decks_table(connection)
            self._ensure_cards_table(connection)
            self._migrate_front_back_to_content(connection)
            self._ensure_learning_columns(connection)
            self._ensure_due_ts_column(connection)
            self._ensure_indexes(connection)

    @staticmethod
    def _column_exists(conn_obj: sqlite3.Connection, table: str, column: str) -> bool:
        """Return True when the given column exists in the specified table."""
        rows = conn_obj.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in rows)

    @staticmethod
    def _ensure_decks_table(conn_obj: sqlite3.Connection) -> None:
        """Create the decks table if it does not exist."""
        conn_obj.execute(
            """
            CREATE TABLE IF NOT EXISTS decks (
                name TEXT PRIMARY KEY
            );
            """
        )

    @staticmethod
    def _ensure_cards_table(conn_obj: sqlite3.Connection) -> None:
        """Create the cards table (current schema) when missing."""
        conn_obj.execute(
            """
            CREATE TABLE IF NOT EXISTS cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                deck_name TEXT REFERENCES decks(name) ON UPDATE CASCADE ON DELETE SET NULL,
                tags  TEXT,
                repetitions INTEGER NOT NULL DEFAULT 0,
                interval    INTEGER NOT NULL DEFAULT 0,
                efactor     REAL    NOT NULL DEFAULT 2.5,
                due         TEXT    NOT NULL,
                last_reviewed TEXT,
                lapses INTEGER NOT NULL DEFAULT 0,
                is_learning INTEGER NOT NULL DEFAULT 1,
                learn_step_idx INTEGER NOT NULL DEFAULT 0,
                due_ts TEXT
            );
            """
        )

    def _migrate_front_back_to_content(self, conn_obj: sqlite3.Connection) -> None:
        """Migrate legacy schemas with front/back columns to the unified JSON content column."""
        has_content = self._column_exists(conn_obj, "cards", "content")
        has_front = self._column_exists(conn_obj, "cards", "front")
        has_back = self._column_exists(conn_obj, "cards", "back")
        if not has_content and (has_front or has_back):
            conn_obj.execute("ALTER TABLE cards RENAME TO cards_legacy")
            conn_obj.execute(
                """
                CREATE TABLE cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    deck_name TEXT REFERENCES decks(name) ON UPDATE CASCADE ON DELETE SET NULL,
                    tags  TEXT,
                    repetitions INTEGER NOT NULL DEFAULT 0,
                    interval    INTEGER NOT NULL DEFAULT 0,
                    efactor     REAL    NOT NULL DEFAULT 2.5,
                    due         TEXT    NOT NULL,
                    last_reviewed TEXT,
                    lapses INTEGER NOT NULL DEFAULT 0,
                    is_learning INTEGER NOT NULL DEFAULT 1,
                    learn_step_idx INTEGER NOT NULL DEFAULT 0,
                    due_ts TEXT
                );
                """
            )
            legacy_rows = conn_obj.execute("SELECT * FROM cards_legacy").fetchall()
            for row in legacy_rows:
                available_columns = set(row.keys())
                content_payload = {}
                if "front" in available_columns:
                    content_payload["front"] = row["front"]
                if "back" in available_columns:
                    content_payload["back"] = row["back"]
                conn_obj.execute(
                    """
                    INSERT INTO cards(
                        id, content, deck_name, tags, repetitions, interval, efactor,
                        due, last_reviewed, lapses, is_learning, learn_step_idx, due_ts
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["id"],
                        json.dumps(content_payload),
                        row["deck_name"],
                        row["tags"],
                        row["repetitions"],
                        row["interval"],
                        row["efactor"],
                        row["due"],
                        row["last_reviewed"],
                        row["lapses"],
                        row["is_learning"] if "is_learning" in available_columns else 1,
                        row["learn_step_idx"] if "learn_step_idx" in available_columns else 0,
                        row["due_ts"] if "due_ts" in available_columns else None,
                    ),
                )
            conn_obj.execute("DROP TABLE cards_legacy")
        elif not has_content:
            conn_obj.execute("ALTER TABLE cards ADD COLUMN content TEXT")
            conn_obj.execute("UPDATE cards SET content='{}' WHERE content IS NULL")

    def _ensure_learning_columns(self, conn_obj: sqlite3.Connection) -> None:
        """Add learning helper columns when the schema still lacks them."""
        if not self._column_exists(conn_obj, "cards", "is_learning"):
            conn_obj.execute("ALTER TABLE cards ADD COLUMN is_learning INTEGER NOT NULL DEFAULT 1")
        if not self._column_exists(conn_obj, "cards", "learn_step_idx"):
            conn_obj.execute("ALTER TABLE cards ADD COLUMN learn_step_idx INTEGER NOT NULL DEFAULT 0")

    def _ensure_due_ts_column(self, conn_obj: sqlite3.Connection) -> None:
        """Add the due_ts column and backfill existing rows if necessary."""
        if not self._column_exists(conn_obj, "cards", "due_ts"):
            conn_obj.execute("ALTER TABLE cards ADD COLUMN due_ts TEXT")
            rows = conn_obj.execute("SELECT id, due FROM cards").fetchall()
            for row in rows:
                due_dt = datetime.combine(
                    datetime.fromisoformat(row["due"]).date(),
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                conn_obj.execute(
                    "UPDATE cards SET due_ts=? WHERE id=?",
                    (due_dt.isoformat(timespec="seconds"), row["id"]),
                )

    @staticmethod
    def _ensure_indexes(conn_obj: sqlite3.Connection) -> None:
        """Create the indexes required for fast due queries."""
        conn_obj.execute("CREATE INDEX IF NOT EXISTS idx_cards_due ON cards(due);")
        conn_obj.execute("CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_name);")
        conn_obj.execute("CREATE INDEX IF NOT EXISTS idx_cards_due_ts ON cards(due_ts);")


db_manager = DatabaseManager(DB_PATH)
