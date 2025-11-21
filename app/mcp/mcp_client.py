import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

from mcp import StdioServerParameters, stdio_client, ClientSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]

server_params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "app.mcp.mcp_server"],
    cwd=str(PROJECT_ROOT),  # garante que a raiz do projeto é o diretório de trabalho
)

async def mcp_call_generate_tts_audio(text: str):
    print(f"[MCP CLIENT] Chamando MCP generate_tts_audio({text!r})")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "generate_tts_audio",
                arguments={"text": text},
            )

            # ✔️ usa structuredContent (camelCase)
            structured = result.structuredContent

            # se o servidor devolver {"result": {...}}
            if isinstance(structured, dict) and "result" in structured:
                structured = structured["result"]

            return structured



def call_generate_tts_audio_sync(text: str) -> Dict[str, Any]:
    """Wrapper síncrono pra usar em código não-async (ex: FastAPI em threadpool)."""
    return asyncio.run(mcp_call_generate_tts_audio(text))


async def mcp_call_pronunciation_practice(text: str):
    print(f"[MCP CLIENT] Chamando MCP pronunciation_practice({text!r})")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "pronunciation_practice",
                arguments={"text": text},
            )

            # ✔️ usa structuredContent (camelCase)
            structured = result.structuredContent

            # se o servidor devolver {"result": {...}}
            if isinstance(structured, dict) and "result" in structured:
                structured = structured["result"]

            return structured

def call_pronunciation_practice_sync(text: str) -> Dict[str, Any]:
    """Wrapper síncrono pra usar em código não-async (ex: FastAPI em threadpool)."""
    return asyncio.run(mcp_call_pronunciation_practice(text))


async def mcp_call_create_new_card(content: Dict[str, Any], deck_id: str):
    print(f"[MCP CLIENT] Chamando MCP create_new_card(deck_id={deck_id!r})")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "create_new_card",
                arguments={"content": content, "deck_id": deck_id},
            )

            # ✔️ usa structuredContent (camelCase)
            structured = result.structuredContent

            # Check for errors
            if result.isError:
                error_msg = "Unknown error"
                if result.content and hasattr(result.content[0], 'text'):
                    error_msg = result.content[0].text
                return {"error": error_msg}

            # se o servidor devolver {"result": {...}}
            if isinstance(structured, dict) and "result" in structured:
                structured = structured["result"]

            return structured


def call_create_new_card_sync(content: Dict[str, Any], deck_id: str) -> Dict[str, Any]:
    """Wrapper síncrono pra usar em código não-async (ex: FastAPI em threadpool)."""
    return asyncio.run(mcp_call_create_new_card(content, deck_id))


async def mcp_call_get_all_decks():
    print(f"[MCP CLIENT] Chamando MCP get_all_decks()")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "get_all_decks",
                arguments={},
            )

            # ✔️ usa structuredContent (camelCase)
            structured = result.structuredContent

            # Check for errors
            if result.isError:
                error_msg = "Unknown error"
                if result.content and hasattr(result.content[0], 'text'):
                    error_msg = result.content[0].text
                return {"error": error_msg}

            # se o servidor devolver {"result": {...}}
            if isinstance(structured, dict) and "result" in structured:
                structured = structured["result"]

            return structured


def call_get_all_decks_sync() -> Dict[str, Any]:
    """Wrapper síncrono pra usar em código não-async (ex: FastAPI em threadpool)."""
    return asyncio.run(mcp_call_get_all_decks())


async def mcp_call_create_new_deck(name: str, type: str = "speech"):
    print(f"[MCP CLIENT] Chamando MCP create_new_deck(name={name!r}, type={type!r})")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "create_new_deck",
                arguments={"name": name, "type": type},
            )

            # ✔️ usa structuredContent (camelCase)
            structured = result.structuredContent

            # Check for errors
            if result.isError:
                error_msg = "Unknown error"
                if result.content and hasattr(result.content[0], 'text'):
                    error_msg = result.content[0].text
                return {"error": error_msg}

            # se o servidor devolver {"result": {...}}
            if isinstance(structured, dict) and "result" in structured:
                structured = structured["result"]

            return structured


def call_create_new_deck_sync(name: str, type: str = "speech") -> Dict[str, Any]:
    """Wrapper síncrono pra usar em código não-async (ex: FastAPI em threadpool)."""
    return asyncio.run(mcp_call_create_new_deck(name, type))