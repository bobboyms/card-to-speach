import pytest
from fastapi import HTTPException
from pathlib import Path
import os
from app.utils.b64 import mp3_to_base64

def test_mp3_to_base64_path_traversal():
    """Test that mp3_to_base64 prevents path traversal."""
    # Create a dummy file outside allowed dirs (e.g., in root)
    dummy_file = Path("dummy_outside.mp3")
    dummy_file.touch()
    
    try:
        # Try to access it via traversal or direct path
        with pytest.raises(HTTPException) as excinfo:
            mp3_to_base64("dummy_outside.mp3")
        assert excinfo.value.status_code == 403
        
        with pytest.raises(HTTPException) as excinfo:
            mp3_to_base64("../mini_elsa/dummy_outside.mp3")
        assert excinfo.value.status_code == 403
        
    finally:
        if dummy_file.exists():
            dummy_file.unlink()

def test_mp3_to_base64_allowed_path():
    """Test that mp3_to_base64 allows access to valid files in temp_files."""
    # Create a dummy file in temp_files
    temp_dir = Path("temp_files")
    temp_dir.mkdir(exist_ok=True)
    dummy_file = temp_dir / "test_valid.mp3"
    dummy_file.touch()
    
    try:
        # Should not raise 403 (might raise ValueError if empty/invalid mp3 content, but not 403)
        # Since we just touched it, it's empty, so base64 encoding might succeed or fail on read, 
        # but we want to ensure it passes the security check.
        # Actually, the function reads the file. Empty file read -> empty bytes -> b64 encode empty -> empty string.
        # But wait, the function checks suffix.
        
        # Let's write some dummy content
        with open(dummy_file, "wb") as f:
            f.write(b"fake mp3 content")
            
        result = mp3_to_base64(str(dummy_file))
        assert isinstance(result, str)
        
    finally:
        if dummy_file.exists():
            dummy_file.unlink()
