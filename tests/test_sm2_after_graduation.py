# tests/test_sm2_after_graduation.py
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

def test_sm2_success_normaliza_e_atualiza_campos(app_client, fixed_now, monkeypatch):
    """
    Após graduado, uma revisão bem-sucedida (grade>=3) deve:
      - usar SM-2 (atualizar interval/repetitions/efactor >= MIN_EF),
      - agendar para 'interval' dias normalizado às 00:00 UTC,
      - manter is_learning=0.
    """
    c = app_client
    _mk_deck(c, "geo")
    card = _mk_card(c, deck="geo")

    # Gradua rápido via EASY (is_learning=0, interval=GRADUATE_EASY_DAYS, due_ts = 00:00 do dia devido)
    r = c.post(f"/cards/{card['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # Avança relógio para o início do dia devido (para poder revisar no SM-2)
    due_day = fixed_now.date() + timedelta(days=api.GRADUATE_EASY_DAYS)
    now2 = datetime(due_day.year, due_day.month, due_day.day, 0, 5, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # Faz uma revisão SM-2 com grade=4
    r2 = c.post("/reviews/next", params={"deck": "geo"}, json={"grade": 4})
    assert r2.status_code == 200, r2.text
    data = r2.json()

    # Verificações de SM-2
    assert data["interval_days"] >= 1
    assert data["efactor"] >= api.MIN_EF
    assert data["repetitions"] >= 1

    # due/due_ts devem estar normalizados para 00:00 do (hoje+interval)
    expected_due_day = (now2.date() + timedelta(days=data["interval_days"])).isoformat()
    row = _row(card["id"])
    assert row["is_learning"] == 0
    assert row["due"] == expected_due_day
    expected_dt = datetime.fromisoformat(expected_due_day + "T00:00:00+00:00")
    assert row["due_ts"] == expected_dt.isoformat(timespec="seconds")

def test_sm2_failure_incrementa_lapses_e_agenda_hoje(app_client, fixed_now, monkeypatch):
    """
    Após graduado, uma falha (grade<=2) deve:
      - incrementar lapses,
      - manter is_learning=0 (não volta para learning),
      - agendar para 'hoje' (next_dt = now).
    """
    c = app_client
    _mk_deck(c, "geo")
    card = _mk_card(c, deck="geo")

    # Gradua via EASY
    r = c.post(f"/cards/{card['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # Vira o dia devido para permitir revisar
    due_day = fixed_now.date() + timedelta(days=api.GRADUATE_EASY_DAYS)
    now2 = datetime(due_day.year, due_day.month, due_day.day, 10, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # Aplica uma falha (grade=2)
    r2 = c.post("/reviews/next", params={"deck": "geo"}, json={"grade": 2})
    assert r2.status_code == 200, r2.text

    row = _row(card["id"])
    assert row["is_learning"] == 0
    # lapses incrementado
    assert row["lapses"] >= 1

    # Como a sua regra de falha agenda para 'now' (mesmo dia),
    # due deve ser a data de hoje (now2.date()), e due_ts ~ now2 (00:00 se normalizar, mas aqui é "hoje")
    assert row["due"] == now2.date().isoformat()
    # due_ts deve ser <= now2 (mesmo dia); não testamos igualdade exata por ser "now"
    assert datetime.fromisoformat(row["due_ts"]) <= now2

def test_botoes_em_graduado_mapeiam_para_grades_e_usam_sm2(app_client, fixed_now, monkeypatch):
    """
    Em cartões graduados, o endpoint de botões deve mapear:
      again→0, hard→3, good→4, easy→5, e seguir a mesma lógica SM-2.
    Validamos com 'again' (lapse) e 'easy' (intervalo alto).
    """
    c = app_client
    _mk_deck(c, "geo")
    card = _mk_card(c, deck="geo")

    # Gradua via EASY
    r = c.post(f"/cards/{card['id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text

    # Vira o dia devido
    due_day = fixed_now.date() + timedelta(days=api.GRADUATE_EASY_DAYS)
    now2 = datetime(due_day.year, due_day.month, due_day.day, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(api, "utc_now", lambda: now2, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now2.date(), raising=True)

    # 'again' em graduado -> grade=0 -> lapse e due = hoje
    r2 = c.post(f"/cards/{card['id']}/review", json={"button": "again"})
    assert r2.status_code == 200, r2.text
    row = _row(card["id"])
    assert row["lapses"] >= 1
    assert row["is_learning"] == 0
    assert row["due"] == now2.date().isoformat()

    # Avança um pouco o relógio dentro do mesmo dia e aplica 'easy' (grade=5)
    now3 = now2 + timedelta(hours=1)
    monkeypatch.setattr(api, "utc_now", lambda: now3, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: now3.date(), raising=True)

    r3 = c.post(f"/cards/{card['id']}/review", json={"button": "easy"})
    assert r3.status_code == 200, r3.text
    data = r3.json()
    assert data["interval_days"] >= 1
    assert data["efactor"] >= api.MIN_EF

    # Próximo due normalizado para 00:00 do (hoje + interval_days)
    expected_due_day = (now3.date() + timedelta(days=data["interval_days"])).isoformat()
    row2 = _row(card["id"])
    assert row2["due"] == expected_due_day
    expected_dt = datetime.fromisoformat(expected_due_day + "T00:00:00+00:00")
    assert row2["due_ts"] == expected_dt.isoformat(timespec="seconds")
