# -*- coding: utf-8 -*-
"""
Streamlit – GUI de avaliação de pronúncia (com gravação no navegador via WebRTC)

Recursos:
- Frase-alvo editável
- Formato de fonemas (IPA/ASCII/ARPAbet)
- Gravar áudio no navegador (streamlit-webrtc) **ou** enviar arquivo
- Pipeline: ASR (mlx-whisper), WER/Confiança combinada, alinhamento léxico,
  fonemas por palavra usando g2p_en + phonecodes
- Resultados em JSON + tabelas

Instalação (ex.: macOS/Apple Silicon):
    pip install streamlit numpy jiwer g2p_en phonecodes soundfile
    pip install mlx mlx-whisper

Execução:
    streamlit run streamlit_pronuncia_app.py

Observações:
- A gravação usa WebRTC; conceda permissão ao microfone no navegador.
- O modelo MLX Whisper roda localmente; ajuste o repositório no sidebar se preferir.
- Se não estiver em Apple Silicon, você pode trocar para o whisper padrão (TODO no código, ver try/except).
"""

import os
import io
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import streamlit as st
import jiwer
import soundfile as sf
from difflib import SequenceMatcher
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

# ========================= Utilitários de áudio (WebRTC) =========================
# WebRTC normalmente entrega áudio a 48kHz. Fazemos reamostragem simples para 16kHz.
def resample_to_16k(x: np.ndarray, src_rate: int = 48000) -> np.ndarray:
    if src_rate == SAMPLE_RATE_TARGET:
        return x
    # Evita dependências extras: reamostra por interpolação linear
    duration = len(x) / float(src_rate)
    t_src = np.linspace(0, duration, num=len(x), endpoint=False)
    t_dst = np.linspace(0, duration, num=int(duration * SAMPLE_RATE_TARGET), endpoint=False)
    y = np.interp(t_dst, t_src, x).astype(np.float32)
    return y


def save_wav_float32(data: np.ndarray, path: str, samplerate: int = SAMPLE_RATE_TARGET):
    # Normaliza para -1..1 e salva float32 (SoundFile lida bem)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    m = np.max(np.abs(data)) or 1.0
    data = (data / m).astype(np.float32)
    sf.write(path, data, samplerate)

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

# ========================= UI =========================
st.set_page_config(page_title="Avaliação de Pronúncia", page_icon="🗣️", layout="wide")
st.title("🗣️ Avaliação de Pronúncia (Streamlit)")

with st.sidebar:
    st.header("Configurações")
    target_text = st.text_area("Frase-alvo (inglês)", value="She look guilty after lying to her friend.")
    phoneme_fmt = st.selectbox("Formato de fonemas", ["ipa", "ascii", "arpabet"], index=0)
    model_repo = st.text_input("Modelo MLX Whisper", value=DEFAULT_MODEL)
    st.caption("Dica: 'small' é rápido; troque para 'medium' ou 'large' se quiser mais qualidade (desde que compatível MLX).")

st.markdown("### 1) Grave ou envie seu áudio")

# ===== Gravador nativo do Streamlit (st.audio_input) =====
rec = st.audio_input("Gravar do microfone", sample_rate=16000)
rec_path = None
if rec is not None:
    os.makedirs("gravacoes", exist_ok=True)
    rec_path = os.path.abspath(os.path.join("gravacoes", f"gravacao_{int(time.time())}.wav"))
    with open(rec_path, "wb") as f:
        f.write(rec.getvalue())
    st.audio(rec)
    st.caption("Se o navegador perguntar, permita o acesso ao microfone.")
    st.session_state["audio_path"] = rec_path  # <- guarda para o próximo rerun

# ===== Upload de arquivo =====
up = st.file_uploader("Ou envie um arquivo (wav/mp3/flac/ogg/m4a)", type=["wav", "mp3", "flac", "ogg", "m4a"])
if up is not None:
    os.makedirs("uploads", exist_ok=True)
    uploaded_path = os.path.abspath(os.path.join("uploads", up.name))
    with open(uploaded_path, "wb") as f:
        f.write(up.read())
    st.audio(uploaded_path)
    st.session_state["audio_path"] = uploaded_path  # <- guarda

st.markdown("### 2) Executar avaliação")

# Caminho final do áudio: preferir o que está na sessão
audio_path = st.session_state.get("audio_path")
# Se acabou de gravar nesta execução e ainda não salvou na sessão, use rec_path
if audio_path is None:
    audio_path = rec_path
    if audio_path is not None:
        st.session_state["audio_path"] = audio_path

can_run = bool(audio_path)
run_btn = st.button("🚀 Avaliar", disabled=not can_run)

if run_btn:
    if (not audio_path) or (not os.path.exists(audio_path)):
        st.error("Áudio não encontrado. Grave ou envie novamente.")
    elif not target_text.strip():
        st.warning("Informe a frase-alvo.")
    elif not HAS_MLX:
        st.error("mlx-whisper não está disponível. Instale `mlx` e `mlx-whisper` (Apple Silicon) ou adapte para outro ASR.")
    else:
        with st.spinner("Executando ASR + análise..."):
            try:
                results = evaluate_pronunciation(
                    audio_path=audio_path,
                    target_text=target_text,
                    phoneme_fmt=phoneme_fmt,
                    model_repo=model_repo,
                )
            except Exception as e:
                st.exception(e)
                results = None
        if results:
            st.success("Concluído!")
            st.subheader("Resumo")
            col1, col2, col3 = st.columns(3)
            col1.metric("WER", f"{results['intelligibility']['wer']:.2f}")
            col2.metric("Confiança", f"{results['intelligibility']['confidence']:.2f}")
            col3.metric("Inteligível?", "Sim" if results['intelligibility']['is_intelligible'] else "Não")

            st.subheader("Transcrição")
            st.write(results["intelligibility"]["transcription"])

            # st.subheader("JSON completo")
            # st.json(results)

            ws = results["phonetic_analysis"].get("word_scores", [])
            if ws:
                import pandas as pd
                st.subheader("Palavras reconhecidas (com timestamps e confiança)")
                st.dataframe(pd.DataFrame(ws))

            fb = results.get("phone_feedback", [])
            if fb:
                import pandas as pd
                st.subheader("Feedback lexical / fonético")
                st.dataframe(pd.DataFrame(fb))

# Indicador de prontidão
if st.session_state.get("audio_path"):
    st.info("🎙️ Áudio pronto para avaliar: " + st.session_state["audio_path"])
