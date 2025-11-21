import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from google.oauth2 import id_token
from google.auth.transport import requests
from jose import jwt

from app import config
from app.services.user_service import UserService
from app.repositories import AuthRepository
from app.schemas import UserOut

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, user_service: UserService, auth_repository: AuthRepository):
        self.user_service = user_service
        self.auth_repository = auth_repository

    def verify_google_token(self, token: str) -> dict:
        try:
            id_info = id_token.verify_oauth2_token(
                token, requests.Request(), config.GOOGLE_CLIENT_ID
            )
            return id_info
        except ValueError as e:
            logger.error(f"Invalid Google token: {e}")
            raise ValueError("Invalid Google token")

    def get_or_create_user(self, user_info: dict) -> UserOut:
        return self.user_service.get_or_create_user(user_info)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=config.JWT_EXPIRES_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM
        )
        return encoded_jwt

    def revoke_token(self, token: str) -> None:
        self.auth_repository.revoke_token(token)

    def is_token_revoked(self, token: str) -> bool:
        return self.auth_repository.is_token_revoked(token)

    def verify_jwt(self, token: str) -> dict:
        try:
            payload = jwt.decode(
                token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.JWTError:
            raise ValueError("Invalid token")
