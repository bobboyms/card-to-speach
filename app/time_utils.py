"""Utility helpers for timezone-aware scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta, date

from .config import TIMEZONE


def day_start(dt: datetime) -> datetime:
    """Return the start of day for the provided datetime using the configured timezone."""
    current_date = dt.date()
    return datetime(current_date.year, current_date.month, current_date.day, tzinfo=TIMEZONE)


def normalize_due_day(days: int, ref: datetime) -> datetime:
    """Normalize a reference date plus the given number of days to midnight in the configured timezone."""
    future_day = ref.date() + timedelta(days=days)
    return datetime(future_day.year, future_day.month, future_day.day, tzinfo=TIMEZONE)


def utc_now() -> datetime:
    """Return the current datetime in UTC."""
    return datetime.now(TIMEZONE)


def utc_today() -> date:
    """Return today's date in UTC."""
    return utc_now().date()
