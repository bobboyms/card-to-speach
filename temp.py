# -*- coding: utf-8 -*-
"""
Pipeline de avaliação de pronúncia (versão enxuta + phonecodes)
- ASR (mlx-whisper) com timestamps por palavra
- WER + confiança combinada
- Alinhamento difflib p/ feedback lexical (com atenuação morfológica)
- Fonemas em formato legível (IPA ou ASCII) com marcas de acento

Config:
    PHONEME_OUTPUT: "ipa" | "ascii" | "arpabet"
"""

import re
import json
from difflib import SequenceMatcher

import numpy as np
import jiwer
import mlx_whisper
from g2p_en import G2p

# >>> NOVO: phonecodes <<<
from phonecodes import phonecodes as pc

# ========================= Configs =========================
PHONEME_OUTPUT = "ipa"  # "ipa" (padrão), "ascii" ou "arpabet"
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
# Mantemos uma aproximação simples para "ascii".
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
def g2p_arpabet_tokens(word: str) -> list[str]:
    """Retorna tokens ARPAbet do g2p_en (filtrados)."""
    toks = [t for t in g2p(word) if t.strip()]
    return toks

# ========================= Conversão com phonecodes =========================
def arpabet_seq_to_ipa(tokens: list[str]) -> str:
    """
    Converte uma sequência ARPAbet (1 palavra) para IPA usando `phonecodes`.
    O `phonecodes` já trata ER1/ER0, AH0→ə e insere acentos.
    Retornamos sem espaços para ficar legível como no seu formato anterior.
    """
    if not tokens:
        return ""
    # phonecodes espera string com tokens separados por espaço
    ipa_spaced = pc.convert(" ".join(tokens), "arpabet", "ipa", "eng")
    ipa_spaced = re.sub(r"\s+", " ", ipa_spaced).strip()
    ipa = ipa_spaced.replace(" ", "")

    # (Opcional) Se você quiser mover ˈ/ˌ para antes do onset consonantal,
    # implemente aqui um pós-processamento que recua o acento até a borda
    # silábica; por padrão, mantemos o acento do phonecodes (antes do núcleo).
    return ipa

def get_phonemes_word(word: str, fmt: str = PHONEME_OUTPUT) -> str:
    """
    fmt: "ipa" (padrão), "ascii" ou "arpabet"
    """
    if not word:
        return ""
    toks = g2p_arpabet_tokens(word)

    if fmt == "arpabet":
        return " ".join(toks)

    ipa = arpabet_seq_to_ipa(toks)

    if fmt == "ascii":
        return ipa_to_readable_ascii(ipa)

    return ipa  # IPA

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

# ========================= Main =========================
if __name__ == "__main__":
    audio_path = "audio.mp3"
    target_text = "She look guilty after lying to her friend."

    # 1) ASR (Whisper / MLX)
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-small-mlx",
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

    # 2) WER + confiança global
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

    # 3) Alinhamento & fonemas por palavra (formato configurável)
    target_tokens = tokenize_words(target_text)
    hyp_tokens = tokenize_words(hypothesis_text)
    opcodes = align_token_sequences(target_tokens, hyp_tokens)

    target_phonemes = {w: get_phonemes_word(w, fmt=PHONEME_OUTPUT) for w in target_tokens}
    hypothesis_phonemes = {w: get_phonemes_word(w, fmt=PHONEME_OUTPUT) for w in hyp_tokens}

    results["phonetic_analysis"] = {
        "format": PHONEME_OUTPUT,
        "target_phonemes": target_phonemes,
        "hypothesis_phonemes": hypothesis_phonemes,
        "word_scores": word_scores
    }

    # 4) Feedback lexical (com atenuação morfológica) + fonemas legíveis
    phone_feedback = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "replace":
            for ti, hi in zip(range(i1, i2), range(j1, j2)):
                tw, hw = target_tokens[ti], hyp_tokens[hi]
                if tw != hw:
                    minor = is_minor_morph_variation(tw, hw)
                    phone_feedback.append({
                        "word": tw,
                        "issue": f"soou como '{hw}'" + (" (variação morfológica leve)" if minor else ""),
                        "expected_phonemes": get_phonemes_word(tw, fmt=PHONEME_OUTPUT),
                        "produced_phonemes": get_phonemes_word(hw, fmt=PHONEME_OUTPUT),
                        "severity": "minor" if minor else "major",
                        "tip": "Atenção ao sufixo (-ed/-s/-ing)." if minor else "Verifique a pronúncia desta palavra."
                    })
        elif tag == "delete":
            for ti in range(i1, i2):
                tw = target_tokens[ti]
                phone_feedback.append({
                    "word": tw,
                    "issue": "omissão",
                    "expected_phonemes": get_phonemes_word(tw, fmt=PHONEME_OUTPUT),
                    "produced_phonemes": "",
                    "severity": "major",
                    "tip": "A palavra parece ter sido omitida."
                })
    results["phone_feedback"] = phone_feedback

    print(json.dumps(results, ensure_ascii=False, indent=4))

    # ========================= (Opcional) Testes rápidos =========================
    # ATENÇÃO: com phonecodes o acento vem junto da vogal (p.ex. 'ˈɛ').
    # Ajuste os "esperados" se antes você marcava o acento antes do onset.
    def _print_result(name, got, exp):
        status = "OK " if got == exp else "ERR"
        mark = "✅" if got == exp else "❌"
        print(f"{mark} {status:3s} {name:14s} -> {got}" + ("" if got == exp else f" | esperado: {exp}"))

    def run_accent_tests():
        tests = {
            # Estes "esperados" assumem o acento do phonecodes (vinculado ao núcleo).
            "record (N)": (["R","EH1","K","ER0","D"],            "ɹˈɛkɚd"),
            "record (V)": (["R","IH0","K","AO1","R","D"],        "ɹɪˈkɔɹd"),
            "content (N)": (["K","AA1","N","T","EH0","N","T"],   "kˈɑntɛnt"),
            "content (V)": (["K","AH0","N","T","EH1","N","T"],   "kənˈtɛnt"),
            "increase (N)": (["IH1","N","K","R","IY2","S"],      "ˈɪnˌkɹis"),
            "increase (V)": (["IH0","N","K","R","IY1","S"],      "ɪnˈkɹis"),
            "photograph":   (["F","OW1","T","AH0","G","R","AE2","F"],        "ˈfoʊtəˌɡɹæf"),
            "photographer": (["F","AH0","T","AA1","G","R","AH0","F","ER0"],  "fəˈtɑɡɹəfɚ"),
            "photography":  (["F","AH0","T","AA1","G","R","AH0","F","IY0"],  "fəˈtɑɡɹəfi"),
            "look":         (["L","UH1","K"],                    "lʊk"),
            "start":        (["S","T","AA1","R","T"],            "stɑɹt"),
        }
        print("\n=== Testes de acento (ˈ/ˌ) ===")
        fails = 0
        for name, (arp_seq, expected) in tests.items():
            got = arpabet_seq_to_ipa(arp_seq)
            _print_result(name, got, expected)
            if got != expected:
                fails += 1
        total = len(tests)
        print(f"\nResumo: {total - fails}/{total} passaram.")
        if fails == 0:
            print("🎉 Todos os testes passaram!")
        else:
            print("⚠️  Diferença provável: convenção de posicionamento do acento (núcleo vs. onset).")

    # Descomente para rodar localmente os testes:
    # run_accent_tests()
