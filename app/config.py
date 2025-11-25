"""Application configuration constants."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

from datetime import timezone
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(BASE_DIR / os.getenv("DB_NAME"))

MIN_EF: float = 1.3
LEARNING_STEPS_MIN: List[int] = [1, 10]
GRADUATE_GOOD_DAYS: int = 1
GRADUATE_EASY_DAYS: int = 4
LEARN_AHEAD_MIN: int = 20
LEARN_AHEAD_IF_EMPTY: bool = True
NORMALIZE_TO_DAY_START: bool = True
TIMEZONE = timezone.utc

# Database
DB_NAME = os.getenv("DB_NAME")

# Google Auth & JWT
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# gRPC
GRPC_ENDPOINT = os.getenv("GRPC_ENDPOINT")

# API Server
API_HOST = os.getenv("API_HOST")
API_PORT = int(os.getenv("API_PORT"))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() in ("true", "1", "yes")

# OpenAI Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
TTS_MODEL = os.getenv("TTS_MODEL")
TTS_VOICE = os.getenv("TTS_VOICE")

# File Storage
TEMP_FILES_DIR = os.getenv("TEMP_FILES_DIR")
