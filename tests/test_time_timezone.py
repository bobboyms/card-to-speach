# tests/test_6_time_timezone.py
from datetime import datetime, timedelta, timezone
import sqlite3, api

def test_graduated_not_advanced_by_learn_ahead(app_client, fixed_now, monkeypatch):
    """
    Graduados não usam learn-ahead: só entram quando due_ts <= agora.
    """
    c = app_client
    c.post("/decks", json={"name":"geo"})
    card = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck":"geo"}).json()
    # Gradua com EASY → due = hoje+4 às 00:00
    c.post(f"/cards/{card['id']}/review", json={"button":"easy"})

    # Avança para antes do due (ontem)
    before_due = fixed_now - timedelta(days=1)
    monkeypatch.setattr(api, "utc_now", lambda: before_due, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: before_due.date(), raising=True)

    # Não deve haver nada devido
    r = c.get("/reviews/next", params={"deck":"geo"})
    assert r.status_code == 404
