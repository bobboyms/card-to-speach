import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

try:
    from app.services import evaluate
    print("Successfully imported app.services.evaluate")
except Exception as e:
    print(f"Failed to import app.services.evaluate: {e}")
    sys.exit(1)
