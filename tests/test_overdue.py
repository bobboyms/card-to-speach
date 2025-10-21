# tests/test_overdue.py
from datetime import datetime, timedelta, timezone
import sqlite3
import api

def _mk_deck(client, name="geo"):
    r = client.post("/decks", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()

def _mk_card(client, deck="geo", front="Q?", back="A"):
    r = client.post("/cards", json={"front": front, "back": back, "deck": deck})
    assert r.status_code == 201, r.text
    return r.json()

def _row(card_id):
    with sqlite3.connect(api.DB) as db:
        db.row_factory = sqlite3.Row
        return db.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()

def test_overdue_graduated_appear_and_priority_by_due_ts(app_client, fixed_now, monkeypatch):
    """
    Cenário:
      - 2 cards graduados (A, B)
      - Forçamos due_ts para o passado (A = mais antigo que B).
      - Avançamos 'agora' para depois desses due_ts.
    Expectativa:
      - /reviews/next retorna A primeiro (mais velho).
      - Após revisar A, /reviews/next retorna B.
    """
    c = app_client
    _mk_deck(c, "geo")

    # Cria dois cards
    A = _mk_card(c, deck="geo", front="A?", back="a")
    B = _mk_card(c, deck="geo", front="B?", back="b")

    # Gradua ambos rapidamente (EASY → sai do learning e vira revisão diária)
    r = c.post(f"/cards/{A['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text
    r = c.post(f"/cards/{B['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # Força 'due_ts' e 'due' para datas PASSADAS, A mais antigo que B
    with sqlite3.connect(api.DB) as db:
        db.row_factory = sqlite3.Row
        # A ficou devido há 2 dias
        dueA_day = (fixed_now.date() - timedelta(days=2)).isoformat()
        dueA_ts  = datetime.fromisoformat(dueA_day + "T00:00:00+00:00").isoformat(timespec="seconds")
        db.execute("UPDATE cards SET due=?, due_ts=? WHERE id=?", (dueA_day, dueA_ts, A["id"]))
        # B ficou devido há 1 dia
        dueB_day = (fixed_now.date() - timedelta(days=1)).isoformat()
        dueB_ts  = datetime.fromisoformat(dueB_day + "T00:00:00+00:00").isoformat(timespec="seconds")
        db.execute("UPDATE cards SET due=?, due_ts=? WHERE id=?", (dueB_day, dueB_ts, B["id"]))
        db.commit()

    # Avança o "agora" para depois de ambos (ex.: hoje + 1 dia)
    now2 = fixed_now + timedelta(days=1)
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # 1) Deve retornar A (o mais antigo)
    r1 = c.get("/reviews/next", params={"deck": "geo"})
    assert r1.status_code == 200, r1.text
    assert r1.json()["id"] == A["id"]

    # 2) Aplica uma revisão SM-2 no "próximo" (A) para tirá-lo da frente
    r2 = c.post("/reviews/next", params={"deck": "geo"}, json={"grade": 4})
    assert r2.status_code == 200, r2.text

    # 3) Agora o próximo atrasado deve ser B
    r3 = c.get("/reviews/next", params={"deck": "geo"})
    assert r3.status_code == 200, r3.text
    assert r3.json()["id"] == B["id"]
