# tests/test_normalize_day_start.py
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

def _get_row(card_id):
    with sqlite3.connect(api.DB) as db:
        db.row_factory = sqlite3.Row
        return db.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()

def test_easy_graduation_normaliza_para_meia_noite(app_client, fixed_now):
    c = app_client
    _mk_deck(c, "geo")
    card = _mk_card(c, deck="geo")

    # EASY → gradua direto para GRADUATE_EASY_DAYS (4 dias), normalizado 00:00
    r = c.post(f"/cards/{card['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    row = _get_row(card["id"])
    # due (YYYY-MM-DD) = hoje + 4
    expected_due = (fixed_now.date() + timedelta(days=api.GRADUATE_EASY_DAYS)).isoformat()
    assert row["due"] == expected_due

    # due_ts = expected_due 00:00:00Z
    expected_dt = datetime.fromisoformat(expected_due + "T00:00:00+00:00")
    assert row["due_ts"] == expected_dt.isoformat(timespec="seconds")

def test_good_ultimo_passo_graduation_meia_noite(app_client, fixed_now):
    c = app_client
    _mk_deck(c, "geo")
    card = _mk_card(c, deck="geo")

    # GOOD (1º passo -> 10m)
    r1 = c.post(f"/cards/{card['id']}/review", json={"button": "good"})
    assert r1.status_code == 200, r1.text
    # GOOD novamente (último passo) → gradua para GRADUATE_GOOD_DAYS (1 dia), 00:00
    r2 = c.post(f"/cards/{card['id']}/review", json={"button": "good"})
    assert r2.status_code == 200, r2.text

    row = _get_row(card["id"])
    expected_due = (fixed_now.date() + timedelta(days=api.GRADUATE_GOOD_DAYS)).isoformat()
    assert row["due"] == expected_due
    expected_dt = datetime.fromisoformat(expected_due + "T00:00:00+00:00")
    assert row["due_ts"] == expected_dt.isoformat(timespec="seconds")

def test_graduated_disponivel_ao_virar_o_dia(app_client, fixed_now, monkeypatch):
    c = app_client
    _mk_deck(c, "geo")
    card = _mk_card(c, deck="geo")

    # Gradua via EASY (4 dias)
    r = c.post(f"/cards/{card['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # Avança o relógio para exatamente o início do dia devido
    due_day = fixed_now.date() + timedelta(days=api.GRADUATE_EASY_DAYS)
    now2 = datetime(due_day.year, due_day.month, due_day.day, tzinfo=timezone.utc)
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # Agora o card deve aparecer em /reviews/next
    resp = c.get("/reviews/next", params={"deck": "geo"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == card["id"]

def test_sm2_pos_graduacao_tambem_normaliza_meia_noite(app_client, fixed_now, monkeypatch):
    c = app_client
    _mk_deck(c, "geo")
    card = _mk_card(c, deck="geo")

    # Gradua rápido via EASY (dia devido = today + 4, às 00:00)
    r = c.post(f"/cards/{card['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # "Amanhã do devido": move o relógio para a data de vencimento + 00:30 (depois das 00:00)
    due_day = fixed_now.date() + timedelta(days=api.GRADUATE_EASY_DAYS)
    now2 = datetime(due_day.year, due_day.month, due_day.day, 0, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # Faz uma revisão SM-2 (grade=4) pelo endpoint de NEXT
    r2 = c.post("/reviews/next", params={"deck": "geo"}, json={"grade": 4})
    assert r2.status_code == 200, r2.text
    data = r2.json()
    k = data["interval_days"]  # intervalo retornado pelo SM-2

    # due/due_ts devem estar normalizados para 00:00 do dia (now2.date() + k)
    expected_due_day = (now2.date() + timedelta(days=k)).isoformat()
    row = _get_row(card["id"])
    assert row["due"] == expected_due_day
    expected_dt = datetime.fromisoformat(expected_due_day + "T00:00:00+00:00")
    assert row["due_ts"] == expected_dt.isoformat(timespec="seconds")
