"""Domain services built on top of repositories."""

from __future__ import annotations

import json
from datetime import datetime, date, timedelta
from typing import Callable, Dict, Any, Optional, List

from fastapi import HTTPException
from supermemo2 import first_review, review

from .repositories import DeckRepository, CardRepository
from .schemas import (
    DeckCreate,
    DeckRename,
    DeckOut,
    CardCreate,
    CardUpdate,
    CardOut,
    CardsPage,
    ReviewOut,
)

BUTTON_TO_GRADE = {"again": 0, "hard": 3, "good": 4, "easy": 5}


class DeckService:
    """Provide deck-related persistence operations and validation utilities."""

    def __init__(self, repo: DeckRepository):
        """Store the repository used for deck persistence."""
        self._repo = repo

    def ensure_exists(self, deck_name: str) -> None:
        """Raise 404 if the given deck does not exist."""
        if not self._repo.exists(deck_name):
            raise HTTPException(status_code=404, detail=f"Deck '{deck_name}' does not exist")

    def create(self, payload: DeckCreate) -> DeckOut:
        """Create a new deck and return its representation."""
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Deck name cannot be empty")
        if self._repo.exists(name):
            raise HTTPException(status_code=409, detail="Deck already exists")
        self._repo.insert(name)
        return DeckOut(name=name)

    def list(self) -> List[DeckOut]:
        """Return all decks ordered alphabetically."""
        rows = self._repo.list_all()
        return [DeckOut(name=row["name"]) for row in rows]

    def rename(self, name: str, payload: DeckRename) -> DeckOut:
        """Rename a deck; raise 404 when the source deck does not exist."""
        new_name = payload.new_name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="New deck name is invalid")
        rowcount = self._repo.update_name(name, new_name)
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Deck not found")
        return DeckOut(name=new_name)

    def delete(self, name: str) -> None:
        """Delete a deck; raise 404 when the deck is missing."""
        rowcount = self._repo.delete(name)
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Deck not found")


class CardService:
    """Handle card CRUD operations, serialization, and conversions."""

    def __init__(
        self,
        repo: CardRepository,
        deck_service: DeckService,
        utc_now: Callable[[], datetime],
    ):
        """Store dependencies required to manage cards."""
        self._repo = repo
        self._deck_service = deck_service
        self._utc_now = utc_now

    def create(self, payload: CardCreate) -> CardOut:
        """Insert a new card in learning mode and return its API model."""
        deck_name = payload.deck.strip()
        self._deck_service.ensure_exists(deck_name)
        now = self._utc_now()
        tags_csv = self._serialize_tags(payload.tags)
        content_json = self._serialize_content(payload.content)
        row = self._repo.insert(
            content_json=content_json,
            deck_name=deck_name,
            tags_csv=tags_csv,
            repetitions=0,
            interval=0,
            efactor=2.5,
            due_date=now.date().isoformat(),
            last_reviewed=None,
            lapses=0,
            is_learning=1,
            learn_step_idx=0,
            due_ts=now.isoformat(timespec="seconds"),
        )
        return self._row_to_cardout(row)

    def list_by_deck(self, deck: str, limit: int) -> List[CardOut]:
        """Return cards for a deck ordered by due timestamp."""
        self._deck_service.ensure_exists(deck)
        rows = self._repo.list_by_deck(deck, limit)
        return [self._row_to_cardout(row) for row in rows]

    def list_due(self, deck: str, limit: int, offset: int) -> CardsPage:
        """Return paginated due cards for a deck."""
        self._deck_service.ensure_exists(deck)
        now_iso = self._utc_now().isoformat(timespec="seconds")
        total, rows = self._repo.list_due(deck, now_iso, limit, offset)
        items = [self._row_to_cardout(row) for row in rows]
        next_off = offset + limit if offset + limit < total else None
        return CardsPage(items=items, total=int(total), limit=limit, offset=offset, next_offset=next_off)

    def get(self, card_id: int) -> CardOut:
        """Fetch a card by ID and return its API model."""
        row = self._repo.fetch_by_id(card_id)
        if not row:
            raise HTTPException(status_code=404, detail="Card not found")
        return self._row_to_cardout(row)

    def update(self, card_id: int, payload: CardUpdate) -> CardOut:
        """Update card content, deck, or tags and return the updated model."""
        row = self._repo.fetch_by_id(card_id)
        if not row:
            raise HTTPException(status_code=404, detail="Card not found")
        if payload.content is not None:
            new_content = self._serialize_content(payload.content)
        else:
            new_content = row["content"]

        if payload.deck is not None:
            deck_name = payload.deck.strip()
            self._deck_service.ensure_exists(deck_name)
            new_deck = deck_name
        else:
            new_deck = row["deck_name"]

        new_tags = self._serialize_tags(payload.tags) if payload.tags is not None else row["tags"]

        updated = self._repo.update_card(card_id, content=new_content, deck_name=new_deck, tags=new_tags)
        return self._row_to_cardout(updated)

    def delete(self, card_id: int) -> None:
        """Delete a card by ID; raise 404 when not found."""
        deleted = self._repo.delete(card_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Card not found")

    def fetch_row(self, card_id: int) -> Dict[str, Any]:
        """Return the raw database row for a card (used by review logic)."""
        row = self._repo.fetch_by_id(card_id)
        if not row:
            raise HTTPException(status_code=404, detail="Card not found")
        return row

    def to_card_out(self, row: Dict[str, Any]) -> CardOut:
        """Convert a database row into the CardOut schema."""
        return self._row_to_cardout(row)

    @staticmethod
    def _serialize_tags(tags: Optional[List[str]]) -> Optional[str]:
        """Turn a list of tags into a normalized CSV value."""
        if tags is None:
            return None
        cleaned = [tag.strip() for tag in tags if tag.strip()]
        return ",".join(cleaned) or None

    @staticmethod
    def _serialize_content(content: Dict[str, Any]) -> str:
        """Serialize arbitrary card content into JSON."""
        try:
            return json.dumps(content)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="Card content must be serializable JSON"
            ) from exc

    @staticmethod
    def _parse_content(raw: Optional[str]) -> Dict[str, Any]:
        """Parse stored JSON content back into a dictionary."""
        if not raw:
            return {}
        try:
            parsed: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail="Invalid card content stored in the database"
            ) from exc
        return parsed

    def _row_to_cardout(self, row: Dict[str, Any]) -> CardOut:
        """Build a CardOut instance from a database row."""
        tags_list = row["tags"].split(",") if row["tags"] else None
        parsed_content = self._parse_content(row["content"])
        return CardOut(
            id=row["id"],
            content=parsed_content,
            deck=row["deck_name"],
            tags=[tag for tag in (tags_list or []) if tag],
            repetitions=row["repetitions"],
            interval=row["interval"],
            efactor=float(row["efactor"]),
            due=row["due"],
            last_reviewed=row["last_reviewed"],
            lapses=row["lapses"],
        )


class ReviewService:
    """Implement review selection, scheduling, and SM-2 updates."""

    def __init__(
        self,
        card_repo: CardRepository,
        card_service: CardService,
        deck_service: DeckService,
        utc_now: Callable[[], datetime],
        utc_today: Callable[[], date],
        normalize_due_day: Callable[[int, datetime], datetime],
        learning_steps: Callable[[], List[int]],
        graduate_good_days: Callable[[], int],
        graduate_easy_days: Callable[[], int],
        normalize_to_day_start: Callable[[], bool],
        learn_ahead_minutes: Callable[[], int],
        learn_ahead_if_empty: Callable[[], bool],
        min_ef: Callable[[], float],
    ):
        """Store references to collaborating services."""
        self._card_repo = card_repo
        self._card_service = card_service
        self._deck_service = deck_service
        self._utc_now = utc_now
        self._utc_today = utc_today
        self._normalize_due_day = normalize_due_day
        self._learning_steps = learning_steps
        self._graduate_good_days = graduate_good_days
        self._graduate_easy_days = graduate_easy_days
        self._normalize_to_day_start = normalize_to_day_start
        self._learn_ahead_minutes = learn_ahead_minutes
        self._learn_ahead_if_empty = learn_ahead_if_empty
        self._min_ef = min_ef

    def peek_next_due(self, deck: str) -> CardOut:
        """Return the next card that should be studied for the given deck."""
        row = self._select_next_row(deck)
        return self._card_service.to_card_out(row)

    def review_next_due_by_grade(self, deck: str, grade: int) -> ReviewOut:
        """Apply an SM-2 grade to the next due card."""
        row = self._select_next_row(deck)
        return self._apply_review(row, grade)

    def review_card_button(self, card_id: int, button: str) -> ReviewOut:
        """Apply a review action (again/hard/good/easy) to a specific card."""
        row = self._card_service.fetch_row(card_id)
        return self._review_with_button(row, button)

    def _select_next_row(self, deck: str) -> Dict[str, Any]:
        """Select the next card row according to due rules and learn-ahead policy."""
        self._deck_service.ensure_exists(deck)
        now = self._utc_now()
        now_iso = now.isoformat(timespec="seconds")
        ahead_iso = (now + timedelta(minutes=self._learn_ahead_minutes())).isoformat(timespec="seconds")
        row = self._card_repo.select_next_for_review(deck, now_iso, ahead_iso, self._learn_ahead_if_empty())
        if not row:
            raise HTTPException(status_code=404, detail="No card due right now")
        return row

    def _review_with_button(self, row: Dict[str, Any], button: str) -> ReviewOut:
        """Handle reviews initiated via buttons, including learning steps."""
        now = self._utc_now()
        is_learning = int(row["is_learning"]) == 1
        step_idx = int(row["learn_step_idx"])

        if is_learning:
            steps = self._learning_steps()
            with self._card_repo.transaction() as connection:
                if button == "again":
                    step_idx = 0
                    self._card_repo.set_due(connection, row["id"], now + timedelta(minutes=steps[0]))
                elif button == "hard":
                    step_idx = max(0, min(step_idx, len(steps) - 1))
                    self._card_repo.set_due(connection, row["id"], now + timedelta(minutes=steps[step_idx]))
                elif button == "good":
                    step_idx += 1
                    if step_idx < len(steps):
                        self._card_repo.set_due(connection, row["id"], now + timedelta(minutes=steps[step_idx]))
                    else:
                        self._card_repo.graduate_card(
                            connection,
                            row["id"],
                            repetitions=1,
                            interval=self._graduate_good_days(),
                            efactor=max(float(row["efactor"]) or 2.5, self._min_ef()),
                        )
                        next_dt = (
                            self._normalize_due_day(self._graduate_good_days(), now)
                            if self._normalize_to_day_start()
                            else now + timedelta(days=self._graduate_good_days())
                        )
                        self._card_repo.set_due(connection, row["id"], next_dt)
                elif button == "easy":
                    self._card_repo.graduate_card(
                        connection,
                        row["id"],
                        repetitions=1,
                        interval=self._graduate_easy_days(),
                        efactor=max(float(row["efactor"]) or 2.5, self._min_ef()),
                    )
                    next_dt = (
                        self._normalize_due_day(self._graduate_easy_days(), now)
                        if self._normalize_to_day_start()
                        else now + timedelta(days=self._graduate_easy_days())
                    )
                    self._card_repo.set_due(connection, row["id"], next_dt)

                self._card_repo.update_learning_progress(
                    connection,
                    row["id"],
                    min(step_idx, max(0, len(steps) - 1)),
                    now.isoformat(timespec="seconds"),
                )
                updated = self._card_repo.fetch_by_id(row["id"], conn=connection)

            due_dt = datetime.fromisoformat(updated["due_ts"]) if updated["due_ts"] else now
            return ReviewOut(
                card=self._card_service.to_card_out(updated),
                next_due=updated["due"],
                interval_days=max(0, (due_dt.date() - self._utc_today()).days),
                efactor=float(updated["efactor"]),
                repetitions=int(updated["repetitions"]),
                lapses=int(updated["lapses"]),
                reviewed_card_id=int(row["id"]),
            )

        grade = BUTTON_TO_GRADE[button]
        return self._apply_review(row, grade)

    def _apply_review(self, row: Dict[str, Any], grade: int) -> ReviewOut:
        """Apply SM-2 scheduling to a card row based on the given grade."""
        if row["repetitions"] == 0 and row["interval"] == 0:
            review_result = first_review(grade)
        else:
            review_result = review(
                grade,
                float(row["efactor"]),
                int(row["interval"]),
                int(row["repetitions"]),
            )

        new_interval = int(review_result["interval"])
        new_repetitions = int(review_result["repetitions"])
        new_efactor = max(float(review_result["easiness"]), self._min_ef())

        now = self._utc_now()
        if grade <= 2:
            next_due_dt = now
            lapses = (row["lapses"] or 0) + 1
        else:
            lapses = row["lapses"] or 0
            next_due_dt = (
                self._normalize_due_day(new_interval, now)
                if self._normalize_to_day_start()
                else now + timedelta(days=new_interval)
            )

        with self._card_repo.transaction() as connection:
            self._card_repo.update_review_metrics(
                connection,
                row["id"],
                new_repetitions,
                new_interval,
                new_efactor,
                review_result.get("review_datetime") or now.isoformat(timespec="seconds"),
                lapses,
            )
            self._card_repo.set_due(connection, row["id"], next_due_dt)
            updated = self._card_repo.fetch_by_id(row["id"], conn=connection)

        return ReviewOut(
            card=self._card_service.to_card_out(updated),
            next_due=updated["due"],
            interval_days=new_interval,
            efactor=new_efactor,
            repetitions=new_repetitions,
            lapses=int(lapses),
            reviewed_card_id=int(row["id"]),
        )
