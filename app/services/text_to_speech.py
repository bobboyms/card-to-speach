import os
import uuid
from pathlib import Path
from typing import Tuple
from openai import OpenAI

from app.utils.files import ensure_directory_exists
from app import config


class TextToSpeach:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")

        self.client = OpenAI(api_key=openai_api_key)

    def generate_tts_audio(self, text: str, output_dir: Path = None) -> Tuple[str, Path]:
        """
        Generate a TTS audio file from text using OpenAI's TTS model.

        Args:
            text: Input string to convert to speech.
            output_dir: Directory where output file will be saved.

        Returns:
            A tuple containing the filename and full Path to the saved audio.
        """
        if output_dir is None:
            output_dir = Path(config.TEMP_FILES_DIR)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{uuid.uuid4()}.mp3"
        output_path = output_dir / unique_name


        try:
            tts_response = self.client.audio.speech.create(
                model=config.TTS_MODEL,
                voice=config.TTS_VOICE,
                input=text
            )
            tts_response.stream_to_file(output_path)
            return unique_name, output_path
        except Exception as exc:
            raise