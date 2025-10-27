# tests/test_normalize_toggle.py
from datetime import datetime, timedelta, timezone
import sqlite3
import api

def _mk_deck(client, name="geo"):
    r = client.post("/decks", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()

def _mk_card(client, deck_id, front="Q?", back="A"):
    r = client.post(
        "/cards",
        json={"content":{"front": front, "back": back}, "deck_id": deck_id},
    )
    assert r.status_code == 201, r.text
    return r.json()

def _row(card_public_id):
    with sqlite3.connect(api.DB) as db:
        db.row_factory = sqlite3.Row
        return db.execute("SELECT * FROM cards WHERE public_id=?", (card_public_id,)).fetchone()

def test_graduation_sem_normalizar_easy_agenda_mesmo_horario(app_client, fixed_now, monkeypatch):
    """
    Com NORMALIZE_TO_DAY_START=False:
      - EASY (graduação direta) agenda para 'agora + GRADUATE_EASY_DAYS'
        mantendo o mesmo horário (não 00:00).
    """
    c = app_client
    deck = _mk_deck(c, "geo")
    card = _mk_card(c, deck_id=deck["public_id"])

    # Desliga a normalização
    monkeypatch.setattr(api, "NORMALIZE_TO_DAY_START", False, raising=False)

    # EASY -> gradua direto
    r = c.post(f"/cards/{card['public_id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # due_ts esperado = fixed_now + GRADUATE_EASY_DAYS (mesmo horário 15:30:00)
    expected_dt = fixed_now + timedelta(days=api.GRADUATE_EASY_DAYS)
    row = _row(card["public_id"])
    assert row["due"] == expected_dt.date().isoformat()
    assert row["due_ts"] == expected_dt.isoformat(timespec="seconds")

def test_sm2_sem_normalizar_agenda_mesmo_horario(app_client, fixed_now, monkeypatch):
    """
    Com NORMALIZE_TO_DAY_START=False:
      - Uma revisão SM-2 bem-sucedida agenda para 'now + interval_days'
        mantendo o mesmo horário (sem arredondar para 00:00).
    """
    c = app_client
    deck = _mk_deck(c, "geo")
    card = _mk_card(c, deck_id=deck["public_id"])

    # Desliga a normalização
    monkeypatch.setattr(api, "NORMALIZE_TO_DAY_START", False, raising=False)

    # Gradua rápido via EASY
    r = c.post(f"/cards/{card['public_id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # Avança o relógio para DEPOIS do due_ts do graduado, para conseguir revisar no SM-2
    due_day = fixed_now + timedelta(days=api.GRADUATE_EASY_DAYS)
    now2 = due_day + timedelta(minutes=10)  # 10 minutos depois do horário original
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # Revisão SM-2 (grade=4)
    r2 = c.post("/reviews/next", params={"deck_id": deck["public_id"]}, json={"grade": 4})
    assert r2.status_code == 200, r2.text
    data = r2.json()
    k = data["interval_days"]
    assert k >= 1

    # Esperado: due_ts = now2 + k dias (mesmo horário de now2)
    expected_dt = now2 + timedelta(days=k)
    row = _row(card["public_id"])
    assert row["due"] == expected_dt.date().isoformat()
    assert row["due_ts"] == expected_dt.isoformat(timespec="seconds")
