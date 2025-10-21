# tests/test_9_performance_basic.py
import time

def test_next_and_due_with_many_cards(app_client):
    c = app_client
    c.post("/decks", json={"name":"geo"})
    for i in range(200):
        r = c.post("/cards", json={"content":{"front":f"Q{i}","back":"A"},"deck":"geo"})
        assert r.status_code == 201

    t0 = time.monotonic()
    r1 = c.get("/reviews/next", params={"deck":"geo"})
    t1 = time.monotonic()
    assert r1.status_code == 200
    # sanity: deve responder "rápido" (ajuste se necessário)
    assert (t1 - t0) < 1.0

    r2 = c.get("/cards/due", params={"deck":"geo","limit":50,"offset":0})
    assert r2.status_code == 200
    js = r2.json()
    assert js["total"] >= 200
    assert len(js["items"]) == 50
