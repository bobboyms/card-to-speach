# -*- coding: utf-8 -*-
"""
FastAPI – API REST de avaliação de pronúncia

Entrada (POST /evaluate): JSON com
{
  "target_text": "She look guilty after lying to her friend.",
  "audio_b64": "<base64 do áudio>",
  "phoneme_fmt": "ipa|ascii|arpabet" (opcional, padrão: "ipa"),
  "model_repo": "mlx-community/whisper-small-mlx" (opcional)
}

Saída: JSON com os mesmos campos produzidos pelo núcleo `evaluate_pronunciation(...)`.

Execução local:
    pip install fastapi uvicorn numpy jiwer g2p_en phonecodes soundfile
    pip install mlx mlx-whisper   # (Apple Silicon)
    uvicorn pron_api:app --reload --port 8000

Teste rápido:
    curl -X POST http://localhost:8000/evaluate \
      -H "Content-Type: application/json" \
      -d '{"target_text":"she looks happy","audio_b64":"<BASE64>"}'

Observações:
- O Whisper (MLX) roda localmente; se não houver suporte MLX, a API retornará erro informando.
- O áudio deve ser mono (ou será convertido) e taxa de amostragem será tratada pelo backend do Whisper quando possível.
"""

import os
import io
import re
import json
import time
import base64
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import jiwer
import soundfile as sf
from difflib import SequenceMatcher
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from g2p_en import G2p
from phonecodes import phonecodes as pc

# ===== Whisper (MLX) =====
try:
    import mlx_whisper
    HAS_MLX = True
except Exception:
    mlx_whisper = None
    HAS_MLX = False

# ========================= Configs =========================
DEFAULT_MODEL = "mlx-community/whisper-small-mlx"
SAMPLE_RATE_TARGET = 16000  # ASR alvo
CHANNELS = 1

g2p = G2p()
MORPH_SUFFIXES = ("ed", "ing", "s", "es")

# ========================= Normalização & Tokenização =========================
def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_words(s: str):
    s = normalize_text(s)
    return [t for t in s.split() if t]

# ========================= IPA -> ASCII legível =========================
_IPA_TO_ASCII = {
    "ɑ": "ah", "æ": "a", "ʌ": "uh", "ɔ": "aw", "aʊ": "ow", "aɪ": "eye",
    "ɛ": "eh", "ɝ": "er", "ɚ": "er", "eɪ": "ay", "ɪ": "ih", "i": "ee",
    "oʊ": "oh", "ɔɪ": "oy", "ʊ": "uh", "u": "oo",
    "θ": "th", "ð": "dh", "ʃ": "sh", "ʒ": "zh", "ŋ": "ng",
    "tʃ": "ch", "dʒ": "j", "ɡ": "g", "ɹ": "r", "j": "y",
}

def ipa_to_readable_ascii(ipa_str: str) -> str:
    order = sorted(_IPA_TO_ASCII.keys(), key=len, reverse=True)
    out = ipa_str
    for k in order:
        out = out.replace(k, _IPA_TO_ASCII[k])
    out = out.replace("ˈ", "'").replace("ˌ", "")
    return out

# ========================= ARPAbet (g2p_en) =========================
def g2p_arpabet_tokens(word: str) -> List[str]:
    toks = [t for t in g2p(word) if t.strip()]
    return toks

# ========================= Conversão com phonecodes =========================
def arpabet_seq_to_ipa(tokens: List[str]) -> str:
    if not tokens:
        return ""
    ipa_spaced = pc.convert(" ".join(tokens), "arpabet", "ipa", "eng")
    ipa_spaced = re.sub(r"\s+", " ", ipa_spaced).strip()
    ipa = ipa_spaced.replace(" ", "")
    return ipa


def get_phonemes_word(word: str, fmt: str = "ipa") -> str:
    if not word:
        return ""
    toks = g2p_arpabet_tokens(word)
    if fmt == "arpabet":
        return " ".join(toks)
    ipa = arpabet_seq_to_ipa(toks)
    if fmt == "ascii":
        return ipa_to_readable_ascii(ipa)
    return ipa

# ========================= Alinhamento & Confiança =========================
def align_token_sequences(ref_tokens, hyp_tokens):
    sm = SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    return sm.get_opcodes()


def map_avg_logprob_to_conf(x):
    if x is None:
        return None
    lo, hi = -2.0, -0.1
    x = max(lo, min(hi, float(x)))
    return (x - lo) / (hi - lo)


def combine_confidence(wer_value, avg_logprob=None, word_probs=None):
    wer_value = min(1.0, max(0.0, float(wer_value)))
    conf_from_wer = max(0.0, 1.0 - (wer_value / 0.3))
    candidates = []
    if avg_logprob is not None:
        c = map_avg_logprob_to_conf(avg_logprob)
        if c is not None:
            candidates.append(c)
    if word_probs:
        clean = [p for p in word_probs if p is not None and 0.0 <= p <= 1.0]
        if clean:
            candidates.append(float(np.mean(clean)))
    if candidates:
        conf_asr = float(np.mean(candidates))
        return float(0.5 * conf_from_wer + 0.5 * conf_asr)
    else:
        return float(conf_from_wer)

# ========================= Atenuação morfológica =========================
def is_minor_morph_variation(target_word: str, hyp_word: str) -> bool:
    t = target_word.lower()
    h = hyp_word.lower()
    if t == h:
        return False
    if any(t + suf == h for suf in MORPH_SUFFIXES):
        return True
    if t.endswith("y") and h == t[:-1] + "ies":
        return True
    if t.endswith("e") and h == t[:-1] + "ing":
        return True
    if t.endswith("ie") and h == t[:-2] + "ying":
        return True
    return False

# ========================= Utilitários de áudio/base64 =========================
def guess_audio_extension(header: bytes) -> str:
    if header.startswith(b"RIFF"):  # WAV
        return ".wav"
    if header.startswith(b"ID3") or header[:2] == b"\xff\xfb":  # MP3
        return ".mp3"
    if header.startswith(b"fLaC"):  # FLAC
        return ".flac"
    if header.startswith(b"OggS"):  # OGG
        return ".ogg"
    if header[4:8] == b"ftyp":  # MP4/M4A container
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


# ========================= Núcleo de avaliação =========================
def evaluate_pronunciation(audio_path: str, target_text: str, phoneme_fmt: str = "ipa",
                           model_repo: str = DEFAULT_MODEL) -> Dict:
    if not HAS_MLX:
        raise RuntimeError("mlx-whisper não está disponível neste ambiente. Instale `mlx` e `mlx-whisper` em Apple Silicon, ou adapte para outro ASR.")

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        word_timestamps=True,
    )
    hypothesis_text = (result.get("text") or "").strip()

    # word-level
    word_scores = []
    word_probs = []
    if "segments" in result:
        for seg in result["segments"]:
            for w in seg.get("words") or []:
                item = {
                    "word": (w.get("word") or "").strip(),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "confidence": w.get("probability", None),
                }
                word_scores.append(item)
                if item["confidence"] is not None:
                    word_probs.append(item["confidence"])

    ref_norm = normalize_text(target_text)
    hyp_norm = normalize_text(hypothesis_text)
    wer = jiwer.wer(ref_norm, hyp_norm)
    wer = min(1.0, max(0.0, float(wer)))

    seg_avg_logprobs = []
    if "segments" in result:
        for seg in result["segments"]:
            if "avg_logprob" in seg and seg["avg_logprob"] is not None:
                seg_avg_logprobs.append(float(seg["avg_logprob"]))
    avg_logprob = float(np.mean(seg_avg_logprobs)) if seg_avg_logprobs else None

    confidence = combine_confidence(wer, avg_logprob=avg_logprob, word_probs=word_probs or None)
    intelligibility = (wer <= 0.30)

    results = {
        "intelligibility": {
            "is_intelligible": bool(intelligibility),
            "wer": float(wer),
            "confidence": float(confidence),
            "transcription": hypothesis_text
        }
    }

    target_tokens = tokenize_words(target_text)
    hyp_tokens = tokenize_words(hypothesis_text)
    opcodes = align_token_sequences(target_tokens, hyp_tokens)

    target_phonemes = {w: get_phonemes_word(w, fmt=phoneme_fmt) for w in target_tokens}
    hypothesis_phonemes = {w: get_phonemes_word(w, fmt=phoneme_fmt) for w in hyp_tokens}

    results["phonetic_analysis"] = {
        "format": phoneme_fmt,
        "target_phonemes": target_phonemes,
        "hypothesis_phonemes": hypothesis_phonemes,
        "word_scores": word_scores
    }

    phone_feedback = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "replace":
            for ti, hi in zip(range(i1, i2), range(j1, j2)):
                if ti >= len(target_tokens) or hi >= len(hyp_tokens):
                    continue
                tw, hw = target_tokens[ti], hyp_tokens[hi]
                if tw != hw:
                    minor = is_minor_morph_variation(tw, hw)
                    phone_feedback.append({
                        "word": tw,
                        "issue": f"soou como '{hw}'" + (" (variação morfológica leve)" if minor else ""),
                        "expected_phonemes": get_phonemes_word(tw, fmt=phoneme_fmt),
                        "produced_phonemes": get_phonemes_word(hw, fmt=phoneme_fmt),
                        "severity": "minor" if minor else "major",
                        "tip": "Atenção ao sufixo (-ed/-s/-ing)." if minor else "Verifique a pronúncia desta palavra."
                    })
        elif tag == "delete":
            for ti in range(i1, i2):
                if ti >= len(target_tokens):
                    continue
                tw = target_tokens[ti]
                phone_feedback.append({
                    "word": tw,
                    "issue": "omissão",
                    "expected_phonemes": get_phonemes_word(tw, fmt=phoneme_fmt),
                    "produced_phonemes": "",
                    "severity": "major",
                    "tip": "A palavra parece ter sido omitida."
                })
    results["phone_feedback"] = phone_feedback
    return results

