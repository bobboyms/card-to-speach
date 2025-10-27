# tests/test_reviews.py
from datetime import timedelta
import sqlite3
import api as anki_api

def _create_deck(client, name="geo"):
    r = client.post("/decks", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()

def _create_card(client, deck_id, front="Q?", back="A", tags=None):
    payload = {"content": {"front": front, "back": back}, "deck_id": deck_id}
    if tags is not None:
        payload["tags"] = tags
    r = client.post("/cards", json=payload)
    assert r.status_code == 201, r.text
    return r.json()

def _get_card(client, card_public_id):
    r = client.get(f"/cards/{card_public_id}")
    assert r.status_code == 200, r.text
    return r.json()

def _peek_next(client, deck_id, ok=True):
    r = client.get(f"/reviews/next?deck_id={deck_id}")
    if ok:
        assert r.status_code == 200, r.text
        return r.json()
    return r

def test_learning_good_entra_em_learn_ahead_imediato(app_client, fixed_now):
    """Cenário: 1 card novo, GOOD na 1ª vez -> próximo passo 10m.
       Como LEARN_AHEAD_MIN=20 e fila vazia, /reviews/next deve trazê-lo imediatamente.
    """
    c = app_client
    deck = _create_deck(c, "geo")
    card = _create_card(c, deck_id=deck["public_id"], front="Capital FR?", back="Paris")

    # Espiar primeiro devido: deve ser o próprio card novo (due_ts=agora)
    first = _peek_next(c, deck_id=deck["public_id"])
    assert first["public_id"] == card["public_id"]

    # GOOD (primeira vez) via botões
    r = c.post(f"/cards/{card['public_id']}/review", json={"button": "good"})
    assert r.status_code == 200, r.text

    # Imediatamente depois, sem avançar o relógio:
    # Deve reaparecer em /reviews/next graças ao learn-ahead (10m <= 20m)
    nxt = _peek_next(c, deck_id=deck["public_id"])
    assert nxt["public_id"] == card["public_id"]

def test_again_reinicia_para_primeiro_passo(app_client, fixed_now):
    """Again no learning deve voltar ao primeiro passo (1m)."""
    c = app_client
    deck = _create_deck(c, "geo")
    card = _create_card(c, deck_id=deck["public_id"])

    # Marca AGAIN
    r = c.post(f"/cards/{card['public_id']}/review", json={"button": "again"})
    assert r.status_code == 200, r.text

    # Verifica no banco: is_learning=1 e due_ts = now + 1m
    with sqlite3.connect(anki_api.DB) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM cards WHERE public_id=?", (card["public_id"],)).fetchone()
        assert row["is_learning"] == 1
        due_ts = row["due_ts"]
        assert due_ts is not None
        # deve ser exatamente +1 minuto
        expected = (fixed_now + timedelta(minutes=1)).isoformat(timespec="seconds")
        assert due_ts == expected

def test_easy_gradua_para_dias(app_client, fixed_now):
    """Easy deve graduar o card diretamente para GRADUATE_EASY_DAYS (4 dias)."""
    c = app_client
    deck = _create_deck(c, "geo")
    card = _create_card(c, deck_id=deck["public_id"])

    # EASY
    r = c.post(f"/cards/{card['public_id']}/review", json={"button": "easy"})
    assert r.status_code == 200, r.text
    out = r.json()["card"]
    assert out["interval"] == anki_api.GRADUATE_EASY_DAYS
    assert out["repetitions"] == 1

    # Verifica no banco: is_learning=0 e due = today + 4 dias
    with sqlite3.connect(anki_api.DB) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM cards WHERE public_id=?", (card["public_id"],)).fetchone()
        assert row["is_learning"] == 0
        assert row["interval"] == anki_api.GRADUATE_EASY_DAYS
        # due diário
        expected_due = (fixed_now + timedelta(days=anki_api.GRADUATE_EASY_DAYS)).date().isoformat()
        assert row["due"] == expected_due

def test_prioridade_reviews_next_devido_real_vem_primeiro(app_client, fixed_now):
    """Se existir um card devido real e outro em learning dentro da janela,
       /reviews/next deve priorizar o devido real.
    """
    c = app_client
    deck = _create_deck(c, "geo")

    # Card A (devido real já): vamos criar e alterar due_ts manualmente para fixed_now - 1s
    cardA = _create_card(c, deck_id=deck["public_id"], front="A?", back="a")
    # Card B (learning): GOOD 1ª vez → 10m
    cardB = _create_card(c, deck_id=deck["public_id"], front="B?", back="b")

    # B: GOOD para ir para 10m
    r = c.post(f"/cards/{cardB['public_id']}/review", json={"button": "good"})
    assert r.status_code == 200, r.text

    # Força A a ficar devido real (due_ts passado)
    with sqlite3.connect(anki_api.DB) as db:
        db.row_factory = sqlite3.Row
        # due_ts = fixed_now - 1s
        past = (fixed_now - timedelta(seconds=1)).isoformat(timespec="seconds")
        db.execute(
            "UPDATE cards SET due_ts=?, due=? WHERE public_id=?",
            (past, fixed_now.date().isoformat(), cardA["public_id"]),
        )
        db.commit()

    # Agora /reviews/next deve priorizar A (devido real) e não puxar B (learning dentro da janela)
    nxt = _peek_next(c, deck_id=deck["public_id"])
    assert nxt["public_id"] == cardA["public_id"]

def test_learn_ahead_fallback_quando_vazio_mas_fora_da_janela(app_client, fixed_now, monkeypatch):
    """Se LEARN_AHEAD_IF_EMPTY=True e não houver nada devido nem learning dentro da janela,
       /reviews/next deve pegar o learning mais próximo mesmo fora da janela.
    """
    c = app_client
    deck = _create_deck(c, "geo")
    card = _create_card(c, deck_id=deck["public_id"])

    # Aumenta a distância do próximo passo para 60 min (fora da janela de 20)
    monkeypatch.setattr(anki_api, "LEARNING_STEPS_MIN", [60, 120], raising=True)

    # GOOD 1ª vez → +60min
    r = c.post(f"/cards/{card['public_id']}/review", json={"button": "good"})
    assert r.status_code == 200, r.text

    # Nada devido (due_ts=+60m), nada dentro da janela (20m),
    # mas LEARN_AHEAD_IF_EMPTY=True => deve trazer esse card mesmo assim (fallback)
    nxt = _peek_next(c, deck_id=deck["public_id"])
    assert nxt["public_id"] == card["public_id"]
