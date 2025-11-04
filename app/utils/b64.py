import base64
import os
import tempfile
from http.client import HTTPException

def guess_audio_extension(header: bytes) -> str:
    if header.startswith(b"RIFF"):  # WAV
        return ".wav"
    if header.startswith(b"ID3") or header[:2] == b"\xff\xfb":  # MP3
        return ".mp3"
    if header.startswith(b"fLaC"):  # FLAC
        return ".flac"
    if header.startswith(b"OggS"):  # OGG
        return ".ogg"
    # WebM/Matroska (EBML) header 0x1A45DFA3
    if len(header) >= 4 and header[:4] == b"\x1aE\xdf\xa3":  # EBML (Matroska/WebM)
        return ".webm"
    # ISO-BMFF (MP4/M4A)
    if header[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"  # padrão seguro


def b64_to_temp_audio_file(b64_str: str) -> str:
    try:
        raw = base64.b64decode(b64_str, validate=True)
    except Exception:
        # pode ser data URL; tentar extrair após vírgula
        try:
            b64_part = b64_str.split(",", 1)[-1]
            raw = base64.b64decode(b64_part, validate=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 inválido: {e}")
    ext = guess_audio_extension(raw[:16])
    fd, path = tempfile.mkstemp(prefix="pron_", suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path

import base64
import os
from typing import Union
from pathlib import Path

def mp3_to_base64(path: Union[str, os.PathLike]) -> str:
    """
    Converte um arquivo .mp3 em uma string Base64.

    Parâmetros
    ----------
    path : str | PathLike
        Caminho para o arquivo .mp3.
    as_data_uri : bool
        Se True, retorna no formato Data URI ("data:audio/mpeg;base64,...").

    Retorna
    -------
    str
        String Base64 (ou Data URI se as_data_uri=True).

    Levanta
    -------
    FileNotFoundError: se o arquivo não existir
    ValueError: se a extensão não for .mp3
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    if p.suffix.lower() != ".mp3":
        raise ValueError("O arquivo precisa ter extensão .mp3")

    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    return b64
