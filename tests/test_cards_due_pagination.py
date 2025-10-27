# tests/test_4_cards_due_pagination.py
def test_cards_due_pagination(app_client):
    c = app_client
    deck = c.post("/decks", json={"name":"geo"}).json()
    for i in range(5):
        c.post(
            "/cards",
            json={"content":{"front":f"Q{i}","back":"A"},"deck_id":deck["public_id"]},
        )

    page1 = c.get("/cards/due", params={"deck_id":deck["public_id"],"limit":2,"offset":0}).json()
    assert page1["total"] == 5
    assert len(page1["items"]) == 2
    assert page1["next_offset"] == 2

    page2 = c.get("/cards/due", params={"deck_id":deck["public_id"],"limit":2,"offset":2}).json()
    assert len(page2["items"]) == 2
    assert page2["next_offset"] == 4

    page3 = c.get("/cards/due", params={"deck_id":deck["public_id"],"limit":2,"offset":4}).json()
    assert len(page3["items"]) == 1
    assert page3["next_offset"] is None
