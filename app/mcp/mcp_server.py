# mcp_server.py
from pathlib import Path
from typing import Tuple, Dict, Any, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.services.text_to_speech import TextToSpeach
from app.utils.b64 import mp3_to_base64
from app.schemas import CardCreate, DeckOut, DeckCreate
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

from fastapi import HTTPException

@mcp.tool()
def create_new_card(
    content: Dict[str, Any] = Field(..., description="Structured card content. Must contain 'phrase' key."),
    deck_id: str = Field(..., description="Deck ID or Deck Name")
) -> Dict[str, Any]:
    """Create a new card in the specified deck."""
    
    # Attempt to resolve deck_id if it's actually a name
    target_deck_id = deck_id
    
    # First check if it works as an ID
    try:
        deck_service.get_by_public_id(target_deck_id)
    except HTTPException:
        # Not found as ID, try as name
        try:
            deck = deck_service.get_by_name(target_deck_id)
            target_deck_id = deck["public_id"]
        except HTTPException:
            # If failing both, we leave it as is and let card_service.create fail
            pass

    payload = CardCreate(content=content, deck_id=target_deck_id)
    card = card_service.create(payload)
 
    return {
        "public_id": card.public_id,
        "deck_id": card.deck_id,
        "deck_name": card.deck_name,
        "due": card.due,
    }



@mcp.tool()
def create_new_deck(
    name: str = Field(..., description="Name of the new deck"),
    type: str = Field("speech", description="Type of the deck: 'speech' or 'shadowing'")
) -> Dict[str, Any]:
    """Create a new deck."""
    payload = DeckCreate(name=name, type=type)
    deck = deck_service.create(payload)
    return deck.model_dump()


@mcp.tool()
def get_all_decks() -> List[DeckOut]:
    return deck_service.list()


if __name__ == "__main__":
    # Transport "stdio" = comunicação via stdin/stdout
    # (é o que o MCP client em Python vai usar)
    mcp.run(transport="stdio")
