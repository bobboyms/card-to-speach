"""Application configuration constants."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(BASE_DIR / "anki_mini.db")

MIN_EF: float = 1.3
LEARNING_STEPS_MIN: List[int] = [1, 10]
GRADUATE_GOOD_DAYS: int = 1
GRADUATE_EASY_DAYS: int = 4
LEARN_AHEAD_MIN: int = 20
LEARN_AHEAD_IF_EMPTY: bool = True
NORMALIZE_TO_DAY_START: bool = True
TIMEZONE = timezone.utc

# Google Auth & JWT
GOOGLE_CLIENT_ID = "391853776253-6gv4td2darv8ojosjs781fv9hugs86pl.apps.googleusercontent.com"
JWT_SECRET = "b2af1bfdb64b48824d7ccc4056413b2f"
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_MINUTES = 60
