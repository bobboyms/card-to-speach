# tests/test_1_decks_crud_integrity.py
def test_create_deck_ok_and_duplicate(app_client):
    c = app_client
    r1 = c.post("/decks", json={"name": "geo", "type": "speech"})
    assert r1.status_code == 201
    created = r1.json()
    assert created["type"] == "speech"
    r2 = c.post("/decks", json={"name": "geo", "type": "speech"})
    assert r2.status_code == 409

def test_create_deck_empty_name(app_client):
    c = app_client
    r = c.post("/decks", json={"name": "", "type": "speech"})
    assert r.status_code in (400, 422)

def test_rename_deck_propagates_to_cards(app_client):
    c = app_client
    deck_geo = c.post("/decks", json={"name": "geo", "type": "speech"}).json()
    c.post("/decks", json={"name": "hist", "type": "shadowing"})
    card = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck_id":deck_geo["public_id"]}).json()
    r = c.patch(f"/decks/{deck_geo['public_id']}", json={"new_name": "geografia"})
    assert r.status_code == 200
    renamed = r.json()
    assert renamed["type"] == "speech"
    rlist = c.get("/cards", params={"deck_id": deck_geo["public_id"]})
    assert rlist.status_code == 200
    cards = rlist.json()
    assert any(x["public_id"] == card["public_id"] for x in cards)
    assert all(x["deck_name"] == "geografia" for x in cards)

def test_delete_deck_sets_cards_null(app_client):
    c = app_client
    deck_geo = c.post("/decks", json={"name":"geo", "type": "speech"}).json()
    card = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck_id":deck_geo["public_id"]}).json()
    r = c.delete(f"/decks/{deck_geo['public_id']}")
    assert r.status_code == 204
    r2 = c.get(f"/cards/{card['public_id']}")
    assert r2.status_code == 200
    fetched = r2.json()
    assert fetched["deck_id"] is None
    assert fetched["deck_name"] is None
