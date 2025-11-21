import sys
from pathlib import Path
import json

# Adiciona o diretório raiz ao path para importar os módulos
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.services.chat_service import ChatService

# Mock OpenAI client to avoid API calls and just test the logic if possible, 
# but since ChatService initializes OpenAI in __init__, we might need a valid key or mock it.
# For this test, we will try to instantiate ChatService and check if the method exists and imports are correct.
# We can't easily test the full flow without a real API key or complex mocking.
# So we will verify the imports and syntax by instantiating the class (if key exists) or just checking imports.

def test_chat_service_integration():
    print("Testing ChatService integration...")
    try:
        # Check if we can import the service and if the new method is imported
        from app.services.chat_service import call_get_all_decks_sync
        print("SUCCESS: call_get_all_decks_sync imported in chat_service.")
        
        # Check if TOOLS has the new tool
        from app.mcp.tools import TOOLS
        has_tool = any(t['function']['name'] == 'get_all_decks' for t in TOOLS)
        if has_tool:
            print("SUCCESS: get_all_decks found in TOOLS.")
        else:
            print("FAILURE: get_all_decks NOT found in TOOLS.")

    except ImportError as e:
        print(f"FAILURE: ImportError: {e}")
    except Exception as e:
        print(f"FAILURE: Exception: {e}")

if __name__ == "__main__":
    test_chat_service_integration()
