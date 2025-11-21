from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.repositories import UserRepository
from app.schemas import UserOut


class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def get_or_create_user(self, user_info: dict) -> UserOut:
        email = user_info.get("email")
        google_id = user_info.get("sub")
        name = user_info.get("name")

        # Check if user exists by google_id
        user = self.user_repository.find_by_google_id(google_id)

        if not user:
            # Check if user exists by email (legacy or manual signup)
            user = self.user_repository.find_by_email(email)
            
            if user:
                # Link google_id to existing user
                user = self.user_repository.update_google_id(user["id"], google_id)

        if not user:
            # Create new user
            public_id = str(uuid4())
            created_at = datetime.now(timezone.utc).isoformat()
            user = self.user_repository.create(
                public_id=public_id,
                email=email,
                name=name,
                google_id=google_id,
                created_at=created_at,
            )

        return UserOut(
            public_id=user["public_id"],
            email=user["email"],
            name=user["name"],
            google_id=user["google_id"],
            created_at=user["created_at"],
        )

    def get_user(self, public_id: str) -> Optional[UserOut]:
        user = self.user_repository.find_by_public_id(public_id)
        if user:
            return UserOut(
                public_id=user["public_id"],
                email=user["email"],
                name=user["name"],
                google_id=user["google_id"],
                created_at=user["created_at"],
            )
        return None
