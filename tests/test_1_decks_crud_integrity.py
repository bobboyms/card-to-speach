# tests/test_1_decks_crud_integrity.py
def test_create_deck_ok_and_duplicate(app_client):
    c = app_client
    r1 = c.post("/decks", json={"name": "geo"})
    assert r1.status_code == 201
    r2 = c.post("/decks", json={"name": "geo"})
    assert r2.status_code == 409

def test_create_deck_empty_name(app_client):
    c = app_client
    r = c.post("/decks", json={"name": ""})
    assert r.status_code in (400, 422)

def test_rename_deck_propagates_to_cards(app_client):
    c = app_client
    c.post("/decks", json={"name": "geo"})
    c.post("/decks", json={"name": "hist"})
    card = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck":"geo"}).json()
    r = c.patch("/decks/geo", json={"new_name": "geografia"})
    assert r.status_code == 200
    rlist = c.get("/cards", params={"deck": "geografia"})
    assert rlist.status_code == 200
    assert any(x["id"] == card["id"] for x in rlist.json())
    r404 = c.get("/cards", params={"deck":"geo"})
    assert r404.status_code == 404

def test_delete_deck_sets_cards_null(app_client):
    c = app_client
    c.post("/decks", json={"name":"geo"})
    card = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck":"geo"}).json()
    r = c.delete("/decks/geo")
    assert r.status_code == 204
    r2 = c.get(f"/cards/{card['id']}")
    assert r2.status_code == 200
    assert r2.json()["deck"] is None
