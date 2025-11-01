"""Domain services built on top of repositories."""

from __future__ import annotations

import json
from datetime import datetime, date, timedelta
from typing import Callable, Dict, Any, Optional, List, Mapping
from uuid import uuid4

from fastapi import HTTPException
from supermemo2 import first_review, review

from .repositories import DeckRepository, CardRepository
from .time_utils import utc_now
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
        """Raise 404 if the given deck does not exist by name."""
        if not self._repo.exists(deck_name):
            raise HTTPException(status_code=404, detail=f"Deck '{deck_name}' does not exist")

    def get_by_public_id(self, public_id: str) -> Mapping[str, Any]:
        """Return deck information by its public identifier."""
        row = self._repo.find_by_public_id(public_id)
        if not row:
            raise HTTPException(status_code=404, detail="Deck not found")
        return row

    def create(self, payload: DeckCreate) -> DeckOut:
        """Create a new deck and return its representation."""
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Deck name cannot be empty")
        if self._repo.exists(name):
            raise HTTPException(status_code=409, detail="Deck already exists")
        public_id = str(uuid4())
        deck_row = self._repo.insert(name=name, deck_type=payload.type, public_id=public_id)
        return DeckOut(
            public_id=deck_row["public_id"],
            name=deck_row["name"],
            type=deck_row["type"],
            due_cards=0,
            total_cards=0,
        )

    def list(self) -> List[DeckOut]:
        """Return all decks ordered alphabetically along with card statistics."""
        now_iso = utc_now().isoformat(timespec="seconds")
        rows = self._repo.list_with_counts(now_iso)
        return [self._row_to_deck_out(row) for row in rows]

    def rename(self, public_id: str, payload: DeckRename) -> DeckOut:
        """Rename a deck using its public identifier; raise 404 when missing."""
        new_name = payload.new_name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="New deck name is invalid")
        deck_row = self._repo.find_by_public_id(public_id)
        if not deck_row:
            raise HTTPException(status_code=404, detail="Deck not found")
        rowcount = self._repo.update_name_by_public_id(public_id, new_name)
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Deck not found")
        now_iso = utc_now().isoformat(timespec="seconds")
        rows = self._repo.list_with_counts(now_iso)
        stats = next((row for row in rows if row["public_id"] == public_id), None)
        if stats:
            return self._row_to_deck_out(stats)
        deck_row = self._repo.find_by_public_id(public_id)
        if not deck_row:
            raise HTTPException(status_code=404, detail="Deck not found after rename")
        return DeckOut(
            public_id=deck_row["public_id"],
            name=deck_row["name"],
            type=deck_row["type"],
            due_cards=0,
            total_cards=0,
        )

    @staticmethod
    def _row_to_deck_out(row: Mapping[str, Any]) -> DeckOut:
        """Build a DeckOut instance from a repository row."""
        return DeckOut(
            public_id=row["public_id"],
            name=row["name"],
            type=row["type"],
            due_cards=int(row["due_cards"] or 0),
            total_cards=int(row["total_cards"] or 0),
        )

    def delete(self, public_id: str) -> None:
        """Delete a deck using its public identifier; raise 404 when missing."""
        rowcount = self._repo.delete_by_public_id(public_id)
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
        deck_id = payload.deck_id.strip()
        if not deck_id:
            raise HTTPException(status_code=400, detail="Deck public id cannot be empty")
        self._deck_service.get_by_public_id(deck_id)
        now = self._utc_now()
        tags_csv = self._serialize_tags(payload.tags)
        content_json = self._serialize_content(payload.content)
        card_public_id = str(uuid4())
        row = self._repo.insert(
            public_id=card_public_id,
            content_json=content_json,
            deck_id=deck_id,
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
        deck_id = deck.strip()
        if not deck_id:
            raise HTTPException(status_code=404, detail="Deck not found")
        self._deck_service.get_by_public_id(deck_id)
        rows = self._repo.list_by_deck(deck_id, limit)
        return [self._row_to_cardout(row) for row in rows]

    def list_due(self, deck: str, limit: int, offset: int) -> CardsPage:
        """Return paginated due cards for a deck."""
        deck_id = deck.strip()
        if not deck_id:
            raise HTTPException(status_code=404, detail="Deck not found")
        self._deck_service.get_by_public_id(deck_id)
        now_iso = self._utc_now().isoformat(timespec="seconds")
        total, rows = self._repo.list_due(deck_id, now_iso, limit, offset)
        items = [self._row_to_cardout(row) for row in rows]
        next_off = offset + limit if offset + limit < total else None
        return CardsPage(items=items, total=int(total), limit=limit, offset=offset, next_offset=next_off)

    def get(self, card_public_id: str) -> CardOut:
        """Fetch a card by public ID and return its API model."""
        public_id = card_public_id.strip()
        if not public_id:
            raise HTTPException(status_code=404, detail="Card not found")
        row = self._repo.fetch_by_public_id(public_id)
        if not row:
            raise HTTPException(status_code=404, detail="Card not found")
        return self._row_to_cardout(row)

    def update(self, card_public_id: str, payload: CardUpdate) -> CardOut:
        """Update card content, deck, or tags and return the updated model."""
        public_id = card_public_id.strip()
        if not public_id:
            raise HTTPException(status_code=404, detail="Card not found")
        row = self._repo.fetch_by_public_id(public_id)
        if not row:
            raise HTTPException(status_code=404, detail="Card not found")
        if payload.content is not None:
            new_content = self._serialize_content(payload.content)
        else:
            new_content = row["content"]

        if payload.deck_id is not None:
            deck_id = payload.deck_id.strip()
            if not deck_id:
                raise HTTPException(status_code=400, detail="Deck public id cannot be empty")
            self._deck_service.get_by_public_id(deck_id)
            new_deck_id = deck_id
        else:
            new_deck_id = row["deck_id"]

        new_tags = self._serialize_tags(payload.tags) if payload.tags is not None else row["tags"]

        updated = self._repo.update_card_by_public_id(
            public_id, content=new_content, deck_id=new_deck_id, tags=new_tags
        )
        return self._row_to_cardout(updated)

    def delete(self, card_public_id: str) -> None:
        """Delete a card by public ID; raise 404 when not found."""
        public_id = card_public_id.strip()
        if not public_id:
            raise HTTPException(status_code=404, detail="Card not found")
        deleted = self._repo.delete_by_public_id(public_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Card not found")

    def fetch_row(self, card_public_id: str) -> Dict[str, Any]:
        """Return the raw database row for a card (used by review logic)."""
        public_id = card_public_id.strip()
        if not public_id:
            raise HTTPException(status_code=404, detail="Card not found")
        row = self._repo.fetch_by_public_id(public_id)
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
        normalized_tags = [tag for tag in (tags_list or []) if tag]
        return CardOut(
            public_id=row["public_id"],
            content=parsed_content,
            deck_id=row["deck_id"],
            deck_name=row["deck_name"],
            tags=normalized_tags or None,
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

    def peek_next_due(self, deck_public_id: str) -> CardOut:
        """Return the next card that should be studied for the given deck."""
        row = self._select_next_row(deck_public_id)
        return self._card_service.to_card_out(row)

    def review_next_due_by_grade(self, deck_public_id: str, grade: int) -> ReviewOut:
        """Apply an SM-2 grade to the next due card."""
        row = self._select_next_row(deck_public_id)
        return self._apply_review(row, grade)

    def review_card_button(self, card_public_id: str, button: str) -> ReviewOut:
        """Apply a review action (again/hard/good/easy) to a specific card."""
        row = self._card_service.fetch_row(card_public_id)
        return self._review_with_button(row, button)

    def _select_next_row(self, deck_public_id: str) -> Dict[str, Any]:
        """Select the next card row according to due rules and learn-ahead policy."""
        deck_id = deck_public_id.strip()
        if not deck_id:
            raise HTTPException(status_code=404, detail="Deck not found")
        self._deck_service.get_by_public_id(deck_id)
        now = self._utc_now()
        now_iso = now.isoformat(timespec="seconds")
        ahead_iso = (now + timedelta(minutes=self._learn_ahead_minutes())).isoformat(timespec="seconds")
        row = self._card_repo.select_next_for_review(
            deck_id, now_iso, ahead_iso, self._learn_ahead_if_empty()
        )
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
                reviewed_card_public_id=row["public_id"],
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
            reviewed_card_public_id=row["public_id"],
        )
