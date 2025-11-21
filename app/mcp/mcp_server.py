# mcp_server.py
from pathlib import Path
from typing import Tuple, Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.services.text_to_speech import TextToSpeach
from app.utils.b64 import mp3_to_base64
from app.schemas import CardCreate
from app.services.other_services import CardService, DeckService
from app.repositories import CardRepository, DeckRepository
from app.db import db_manager
from app.time_utils import utc_now

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Cria o servidor MCP
mcp = FastMCP("AddServer")

# Instantiate dependencies
text_speech_service: TextToSpeach = TextToSpeach()
card_repo = CardRepository(db_manager)
deck_repo = DeckRepository(db_manager)
deck_service = DeckService(deck_repo)
card_service = CardService(card_repo, deck_service, text_speech_service, utc_now)


@mcp.tool()
def generate_tts_audio(text: str) -> Dict[str, str]:
    """
    Gera um áudio a partir de texto.
    Retorna o caminho do arquivo gerado.
    """
    # supondo que isso retorna um Path
    file_name, audio_path = text_speech_service.generate_tts_audio(text)

    #

    return {
        "file_path": str(file_name),
    }

@mcp.tool()
def pronunciation_practice(text: str) -> Dict[str, str]:
    return {
        "practice": str(text),
    }
    # return "START_PRACTICE: " + text

@mcp.tool()
def create_new_card(
    content: Dict[str, Any] = Field(..., description="Structured card content. Must contain 'phrase' key."),
    deck_id: str = Field(..., description="Deck ID")
) -> Dict[str, Any]:
    """Create a new card in the specified deck."""
    payload = CardCreate(content=content, deck_id=deck_id)
    card = card_service.create(payload)
    return card.model_dump()


if __name__ == "__main__":
    # Transport "stdio" = comunicação via stdin/stdout
    # (é o que o MCP client em Python vai usar)
    mcp.run(transport="stdio")
