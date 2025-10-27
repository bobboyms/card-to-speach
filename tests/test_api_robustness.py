# tests/test_8_api_robustness.py
def test_health_and_common_errors(app_client):
    c = app_client

    # /health
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # deck inexistente em GET /cards
    r2 = c.get("/cards", params={"deck_id":"nope"})
    assert r2.status_code == 404

    # criar deck e cartão
    deck = c.post("/decks", json={"name":"geo"}).json()
    r3 = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck_id":deck["public_id"]})
    assert r3.status_code == 201
    card = r3.json()

    # NOT FOUND em card
    r4 = c.get("/cards/00000000-0000-0000-0000-000000000000")
    assert r4.status_code == 404

    # 400/422 em payload inválido
    r5 = c.patch(f"/cards/{card['public_id']}", json={"content": {}})
    assert r5.status_code in (400,422)
