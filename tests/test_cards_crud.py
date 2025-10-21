# tests/test_2_cards_crud.py
def test_create_card_validations_and_move_deck(app_client):
    c = app_client
    c.post("/decks", json={"name":"geo"})
    c.post("/decks", json={"name":"hist"})

    # deck inexistente
    r_bad = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck":"x"})
    assert r_bad.status_code == 404

    # criar ok
    r = c.post("/cards", json={"content":{"front":"Q1","back":"A1"},"deck":"geo","tags":["a","b"]})
    assert r.status_code == 201
    card = r.json()
    assert card["content"] == {"front":"Q1","back":"A1"}
    assert card["tags"] == ["a","b"]

    # patch: mover de deck e limpar tags
    r2 = c.patch(f"/cards/{card['id']}", json={"deck":"hist","tags":[]})
    assert r2.status_code == 200
    assert r2.json()["deck"] == "hist"


    # delete
    r3 = c.delete(f"/cards/{card['id']}")
    assert r3.status_code == 204
    r4 = c.get(f"/cards/{card['id']}")
    assert r4.status_code == 404
