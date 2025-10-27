# tests/test_2_cards_crud.py
def test_create_card_validations_and_move_deck(app_client):
    c = app_client
    deck_geo = c.post("/decks", json={"name":"geo"}).json()
    deck_hist = c.post("/decks", json={"name":"hist"}).json()

    # deck inexistente
    r_bad = c.post("/cards", json={"content":{"front":"Q","back":"A"},"deck_id":"x"})
    assert r_bad.status_code == 404

    # criar ok
    r = c.post(
        "/cards",
        json={
            "content":{"front":"Q1","back":"A1"},
            "deck_id":deck_geo["public_id"],
            "tags":["a","b"],
        },
    )
    assert r.status_code == 201
    card = r.json()
    assert card["content"] == {"front":"Q1","back":"A1"}
    assert card["tags"] == ["a","b"]
    assert card["deck_id"] == deck_geo["public_id"]
    assert card["deck_name"] == "geo"
    assert "public_id" in card

    # patch: mover de deck e limpar tags
    r2 = c.patch(
        f"/cards/{card['public_id']}",
        json={"deck_id":deck_hist["public_id"],"tags":[]},
    )
    assert r2.status_code == 200
    updated = r2.json()
    assert updated["deck_id"] == deck_hist["public_id"]
    assert updated["deck_name"] == "hist"
    assert updated["tags"] is None


    # delete
    r3 = c.delete(f"/cards/{card['public_id']}")
    assert r3.status_code == 204
    r4 = c.get(f"/cards/{card['public_id']}")
    assert r4.status_code == 404
