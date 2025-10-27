# tests/test_3_selection_priority_filters.py
from datetime import datetime, timedelta, timezone
import sqlite3, api

def test_priority_due_over_learning_ahead_and_deck_isolation(app_client, fixed_now):
    c = app_client
    deck_geo = c.post("/decks", json={"name":"geo"}).json()
    deck_hist = c.post("/decks", json={"name":"hist"}).json()

    # geo: dois cards
    a = c.post(
        "/cards",
        json={"content":{"front":"A?","back":"a"},"deck_id":deck_geo["public_id"]},
    ).json()
    b = c.post(
        "/cards",
        json={"content":{"front":"B?","back":"b"},"deck_id":deck_geo["public_id"]},
    ).json()

    # hist: um card
    h = c.post(
        "/cards",
        json={"content":{"front":"H?","back":"h"},"deck_id":deck_hist["public_id"]},
    ).json()

    # b → GOOD (entra em learning 10m à frente = learning ahead)
    c.post(f"/cards/{b['public_id']}/review", json={"button":"good"})

    # a → forçar devido real (due_ts no passado)
    with sqlite3.connect(api.DB) as db:
        db.row_factory = sqlite3.Row
        past = (fixed_now - timedelta(seconds=1)).isoformat(timespec="seconds")
        db.execute(
            "UPDATE cards SET due_ts=?, due=? WHERE public_id=?",
            (past, fixed_now.date().isoformat(), a["public_id"]),
        )
        db.commit()

    # /reviews/next?deck=geo → deve retornar A (devido real), não B (learning ahead)
    r_geo = c.get("/reviews/next", params={"deck_id":deck_geo["public_id"]})
    assert r_geo.status_code == 200
    assert r_geo.json()["public_id"] == a["public_id"]

    # /reviews/next?deck=hist → não pode retornar card de geo
    r_hist = c.get("/reviews/next", params={"deck_id":deck_hist["public_id"]})
    assert r_hist.status_code == 200
    assert r_hist.json()["deck_name"] == "hist"
    assert r_hist.json()["public_id"] == h["public_id"]
