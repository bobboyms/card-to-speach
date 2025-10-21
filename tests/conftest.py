# tests/conftest.py
import os
import tempfile
from datetime import datetime, timezone
import importlib
import pytest
from fastapi.testclient import TestClient

@pytest.fixture()
def fixed_now():
    # 2025-10-20 15:30:00 UTC
    return datetime(2025, 10, 20, 15, 30, 0, tzinfo=timezone.utc)

@pytest.fixture()
def app_client(monkeypatch, fixed_now):
    import api
    importlib.reload(api)

    # DB temporário
    tmp_db = tempfile.NamedTemporaryFile(delete=False)
    tmp_db.close()
    monkeypatch.setattr(api, "DB", tmp_db.name, raising=True)

    # Congela o relógio
    monkeypatch.setattr(api, "utc_now", lambda: fixed_now, raising=True)
    monkeypatch.setattr(api, "utc_today", lambda: fixed_now.date(), raising=True)

    # Políticas para testes
    monkeypatch.setattr(api, "LEARNING_STEPS_MIN", [1, 10], raising=False)
    monkeypatch.setattr(api, "GRADUATE_GOOD_DAYS", 1, raising=False)
    monkeypatch.setattr(api, "GRADUATE_EASY_DAYS", 4, raising=False)
    monkeypatch.setattr(api, "LEARN_AHEAD_MIN", 20, raising=False)
    monkeypatch.setattr(api, "LEARN_AHEAD_IF_EMPTY", True, raising=False)
    monkeypatch.setattr(api, "NORMALIZE_TO_DAY_START", True, raising=False)

    api.init_db()
    client = TestClient(api.app)

    yield client

    os.unlink(tmp_db.name)
