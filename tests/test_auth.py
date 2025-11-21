import pytest
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from app.services.auth_service import AuthService
from app.schemas import UserOut
from app.db import DatabaseManager
from api import app, get_user_service, get_auth_repository
from app.services.user_service import UserService

client = TestClient(app)

from app.repositories import AuthRepository

@pytest.fixture
def mock_db_manager():
    return MagicMock(spec=DatabaseManager)

@pytest.fixture
def mock_user_service():
    return MagicMock(spec=UserService)

@pytest.fixture
def mock_auth_repository():
    repo = MagicMock(spec=AuthRepository)
    repo.is_token_revoked.return_value = False
    return repo

@pytest.fixture
def override_dependencies(mock_user_service, mock_auth_repository):
    app.dependency_overrides[get_user_service] = lambda: mock_user_service
    app.dependency_overrides[get_auth_repository] = lambda: mock_auth_repository
    yield
    app.dependency_overrides = {}

def test_login_google_success(override_get_auth_service):
    mock_user = UserOut(
        public_id="test-uuid",
        email="test@example.com",
        name="Test User",
        google_id="123456789",
        created_at="2023-01-01T00:00:00Z"
    )
    
def test_login_google_success(override_dependencies, mock_user_service):
    # Mock verify_google_token (we need to mock it on the AuthService instance or patch it)
    # Since we are using real AuthService, we need to patch verify_google_token or mock the id_token.verify_oauth2_token call
    with patch("app.services.auth_service.id_token.verify_oauth2_token") as mock_verify:
        mock_verify.return_value = {
            "sub": "valid-google-token",
            "email": "test@example.com",
            "name": "Test User"
        }
        
        mock_user_service.get_or_create_user.return_value = UserOut(
            public_id="test-uuid",
            email="test@example.com",
            name="Test User",
            google_id="valid-google-token",
            created_at="2023-01-01T00:00:00Z"
        )

        response = client.post("/auth/google", json={"token": "valid-google-token"})

    assert response.status_code == 200
    data = response.json()
    assert data["token_type"] == "bearer"
    assert "access_token" in data
    
    # Verify that create_access_token was called (implicitly by checking response)
    # Verify user service was called
    mock_user_service.get_or_create_user.assert_called_once()

def test_login_google_invalid_token(override_dependencies):
    with patch("app.services.auth_service.id_token.verify_oauth2_token") as mock_verify:
        mock_verify.side_effect = ValueError("Invalid token")
        response = client.post("/auth/google", json={"token": "invalid-token"})
    
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid Google token"}

def test_logout(override_dependencies, mock_auth_repository):
    # We need a valid token to call logout. 
    # Since we use real AuthService, we can generate one or mock verify_jwt.
    # But verify_auth_global uses auth_service.verify_jwt.
    # We can patch jwt.decode in AuthService to return a valid payload.
    
    with patch("app.services.auth_service.jwt.decode") as mock_decode:
        mock_decode.return_value = {"sub": "test-user"}
        mock_auth_repository.is_token_revoked.return_value = False
        
        response = client.post("/logout", headers={"Authorization": "Bearer valid-jwt-token"})
    
    assert response.status_code == 204
    mock_auth_repository.revoke_token.assert_called_once_with("valid-jwt-token")

def test_protected_route_no_token(override_dependencies):
    response = client.get("/decks")
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}

def test_protected_route_invalid_token(override_dependencies):
    # Real AuthService will raise error on invalid token
    response = client.get("/decks", headers={"Authorization": "Bearer invalid-token"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}

def test_protected_route_revoked_token(override_dependencies, mock_auth_repository):
    mock_auth_repository.is_token_revoked.return_value = True
    response = client.get("/decks", headers={"Authorization": "Bearer revoked-token"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Token has been revoked"}

def test_public_route_health(override_dependencies):
    response = client.get("/health")
    assert response.status_code == 200
