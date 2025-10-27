# tests/test_5_sm2_edges_safety.py
from datetime import datetime, timedelta, timezone
import sqlite3, api

def test_grade_bounds_and_min_ef_and_lapses(app_client, fixed_now, monkeypatch):
    c = app_client
    deck = c.post("/decks", json={"name":"geo"}).json()
    card = c.post(
        "/cards",
        json={"content":{"front":"Q","back":"A"},"deck_id":deck["public_id"]},
    ).json()

    # gradua com EASY
    c.post(f"/cards/{card['public_id']}/review", json={"button":"easy"})

    # move relógio para o dia devido (para permitir revisão)
    due_day = fixed_now.date() + timedelta(days=api.GRADUATE_EASY_DAYS)
    now2 = datetime(due_day.year, due_day.month, due_day.day, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # notas inválidas
    r_bad1 = c.post("/reviews/next", params={"deck_id":deck["public_id"]}, json={"grade": -1})
    assert r_bad1.status_code in (400,422)
    r_bad2 = c.post("/reviews/next", params={"deck_id":deck["public_id"]}, json={"grade": 6})
    assert r_bad2.status_code in (400,422)

    # aplicar várias falhas para tentar derrubar efactor
    for _ in range(3):
        r = c.post("/reviews/next", params={"deck_id":deck["public_id"]}, json={"grade": 0})
        assert r.status_code == 200
        data = r.json()
        assert data["efactor"] >= api.MIN_EF  # piso mantido

    # lapses incrementou
    row = (
        sqlite3.connect(api.DB)
        .execute("SELECT lapses FROM cards WHERE public_id=?", (card["public_id"],))
        .fetchone()
    )
    assert row[0] >= 3
