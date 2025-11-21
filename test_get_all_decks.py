import asyncio
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para importar os módulos
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.mcp.mcp_client import call_get_all_decks_sync

def test_get_all_decks():
    print("Testing get_all_decks...")
    try:
        result = call_get_all_decks_sync()
        print("Result:", result)
        if isinstance(result, list):
            print("SUCCESS: Received a list of decks.")
        else:
            print("FAILURE: Did not receive a list.")
    except Exception as e:
        print(f"FAILURE: Exception occurred: {e}")

if __name__ == "__main__":
    test_get_all_decks()
