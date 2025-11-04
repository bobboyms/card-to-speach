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
- Suporta **audio/webm (Opus)**: o backend converte automaticamente para WAV 16 kHz mono via `ffmpeg` se necessário.
- O Whisper (MLX) roda localmente; se não houver suporte MLX, a API retornará erro informando.
- O áudio deve ser mono (ou será convertido) e taxa de amostragem será tratada pelo backend do Whisper quando possível.

- O Whisper (MLX) roda localmente; se não houver suporte MLX, a API retornará erro informando.
- O áudio deve ser mono (ou será convertido) e taxa de amostragem será tratada pelo backend do Whisper quando possível.
"""

import re
import base64
from threading import Lock
from typing import List, Dict, Optional

import numpy as np
import jiwer
from difflib import SequenceMatcher
from g2p_en import G2p
from phonecodes import phonecodes as pc

from app.models.word_segment import WordSegment
from app.services.alignment_service import ForcedAlignmentService
from app.services.audio_service import AudioService, AudioServiceError
from app.services.transcription_service import TranscriptionService

g2p = G2p()
MORPH_SUFFIXES = ("ed", "ing", "s", "es")


# ========================= Serviços em cache =========================
_KNOWN_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}
_AUDIO_SERVICE = AudioService(target_sr=16000, mono=True, apply_fade_ms=3)
_ALIGNER = ForcedAlignmentService(min_word_len_s=0.015)
_TRANSCRIBER_SERVICE = TranscriptionService(whisper_model="small")
_TRANSCRIBER_LOCK = Lock()

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


# ========================= Núcleo de avaliação =========================
def evaluate_pronunciation(
        audio_path: str,
        target_text: str,
        phoneme_fmt: str = "ipa",
        *,
        audio_service: AudioService | None = None,
        aligner: ForcedAlignmentService | None = None,
        transcriber: TranscriptionService | None = None,
) -> Dict:
    audio = audio_service or _AUDIO_SERVICE
    aligner = aligner or _ALIGNER
    transcriber = transcriber or _TRANSCRIBER_SERVICE

    wav_path = audio.convert_to_wav(audio_path)
    hypothesis_text, word_scores, word_probs, avg_logprob = transcriber.transcribe_text(wav_path)

    # ===== novo: normalização + flag global de mismatch =====
    ref_norm = normalize_text(target_text)
    hyp_norm = normalize_text(hypothesis_text)
    target_text_mismatch = (ref_norm != hyp_norm)
    # ========================================================

    wer = jiwer.wer(ref_norm, hyp_norm)
    wer = min(1.0, max(0.0, float(wer)))

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

    # ===== novo: flags por palavra do target =====
    # True = pronunciada corretamente (alinhada como 'equal'); False = trocada/omitida
    correct_flags = [False] * len(target_tokens)
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i1, i2):
                correct_flags[k] = True
        elif tag in ("replace", "delete"):
            for k in range(i1, i2):
                correct_flags[k] = False
        # 'insert' não afeta palavras do target
    # =============================================
    # 3) alinhar (CTC)
    words: list[WordSegment] = aligner.align(wav_path, hypothesis_text)

    phrases_audio_b64: List[Optional[str]] = []
    for w in words:
        try:
            audio_bytes = audio.cut_precise_to_bytes(
                wav_path,
                w.start,
                w.end,
                fmt="mp3",
            )
            if audio_bytes:
                phrases_audio_b64.append(base64.b64encode(audio_bytes).decode("ascii"))
            else:
                phrases_audio_b64.append(None)
        except AudioServiceError:
            phrases_audio_b64.append(None)

    # target_phonemes agora é word -> {"phonemes": "...", "is_correct": bool}
    target_phonemes = {}
    for idx, w in enumerate(target_tokens):
        target_phonemes[w] = {
            "phonemes": get_phonemes_word(w, fmt=phoneme_fmt),
            "is_correct": bool(correct_flags[idx]),
            "audio_b64": phrases_audio_b64[idx] if idx < len(phrases_audio_b64) else None,
        }

    # hipótese continua como mapa simples (por palavra)
    hypothesis_phonemes = {w: get_phonemes_word(w, fmt=phoneme_fmt) for w in hyp_tokens}

    results["phonetic_analysis"] = {
        "format": phoneme_fmt,
        "target_phonemes": target_phonemes,              # <-- agora com is_correct por palavra
        "hypothesis_phonemes": hypothesis_phonemes,
        "word_scores": word_scores,
        "target_text_mismatch": bool(target_text_mismatch)  # <-- flag global solicitada antes
    }

    # feedback fonético (inalterado)
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
