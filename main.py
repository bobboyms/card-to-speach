# -*- coding: utf-8 -*-
"""
Pipeline de avaliação de pronúncia — versão com CTC fonêmico:
- ASR (mlx-whisper) com timestamps por palavra
- WER + confiança combinada
- Alinhamento difflib p/ feedback lexical (com atenuação morfológica)
- Fluência (WPM/pausas/estabilidade)
- Prosódia (F0 global) + stress lexical heurístico
- Score por palavra SEM MFA via CTC fonêmico:
  * vocabulário reconstruído a partir do tokenizer (label2id consistente)
  * posterior renormalizado excluindo BLANK
  * composição para ditongos/africadas (média geométrica)
  * sub-janela mínima de 60 ms por fonema
  * score híbrido 50% CTC + 50% ASR  ⟵ (ajuste 3)
  * mapeamento ER com prioridade por stress  ⟵ (ajuste 2)
"""

import re
import json
import tempfile
from difflib import SequenceMatcher

import numpy as np
import jiwer
import mlx_whisper
from pydub import AudioSegment
import parselmouth
import cmudict
from g2p_en import G2p

import torch
from transformers import AutoProcessor, AutoModelForCTC

#######
import json

def simple_feedback_from_json(json_str: str, threshold: float = 60.0, include_feedback: bool = True) -> str:
    """
    Lê o JSON (string) do pipeline e devolve um texto simples contendo:
      - Palavras possivelmente erradas (score < threshold)
      - Fluency score (0–100), WPM e nº de pausas

    Parâmetros:
      json_str: string com o JSON completo do resultado do pipeline
      threshold: limiar de corte para marcar palavra como "errada"
      include_feedback: se True, também considera 'phone_feedback' (erros major)

    Retorna:
      Uma string formatada com o resumo.
    """
    data = json.loads(json_str)

    # 1) Palavras “erradas” pelo score
    wrong_by_score = []
    for item in (data.get("word_pron_scores") or []):
        word = item.get("word")
        score = item.get("score")
        if word and isinstance(score, (int, float)) and score < threshold:
            wrong_by_score.append((word, float(score)))

    # 2) Palavras marcadas pelo feedback lexical (major)
    wrong_by_feedback = []
    if include_feedback:
        for fb in (data.get("phone_feedback") or []):
            word = fb.get("word")
            severity = (fb.get("severity") or "").lower()
            if word and severity != "minor":
                wrong_by_feedback.append(word)

    # Mesclar (sem duplicar), priorizando o score quando existir
    wrong_words = {}
    for w, s in wrong_by_score:
        wrong_words[w] = s
    for w in wrong_by_feedback:
        wrong_words.setdefault(w, None)  # se não tinha score, mantém None

    # 3) Fluência (mesma lógica do pipeline)
    fluency = data.get("fluency", {}) or {}
    wpm = fluency.get("wpm", None)
    num_pauses = fluency.get("num_pauses", 0)
    rate_stability = float(fluency.get("rate_stability", 0) or 0)
    wpm_penalty = float(fluency.get("wpm_penalty", 0) or 0)
    fluency_score = max(0.0, min(100.0, 100.0 - (float(num_pauses) * 5.0) - (rate_stability * 20.0) - wpm_penalty))

    # 4) Montar texto
    lines = []
    lines.append("=== Palavras com possível erro de pronúncia ===")
    if not wrong_words:
        lines.append("Nenhuma palavra marcada como problemática pelo limiar atual.")
    else:
        # ordenar por score crescente (None vai por último)
        def _key(item):
            word, score = item
            return (1, 0) if score is None else (0, score)

        for word, score in sorted(wrong_words.items(), key=_key):
            if score is None:
                lines.append(f"- {word} (marcada por feedback lexical)")
            else:
                lines.append(f"- {word}: score {score:.1f} (< {threshold})")

    lines.append("\n=== Fluência ===")
    if isinstance(wpm, (int, float)):
        lines.append(f"WPM: {wpm:.0f}")
    lines.append(f"Pausas longas: {int(num_pauses or 0)}")
    lines.append(f"Fluency score (0–100): {fluency_score:.1f}")

    return "\n".join(lines)

#######

# ========================= Configs =========================
PHONEME_CTC_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
MIN_SUBWINDOW_MS = 60  # mantém 60 ms (ajuste 1 NÃO solicitado)

g2p = G2p()
cmu_dict = cmudict.dict()

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
    "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}
MORPH_SUFFIXES = ("ed", "ing", "s", "es")

# ========================= Texto / Fonemas =========================
def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_words(s: str):
    s = normalize_text(s)
    return [t for t in s.split() if t]

def get_phonemes_word(word: str) -> str:
    if not word:
        return ""
    phones = g2p(word)
    phones = [p for p in phones if p.strip()]
    return " ".join(phones)

def cmu_pronunciations(word: str):
    return cmu_dict.get(word.lower(), [])

def syllable_phone_indices(pron):
    idxs = []
    for i, p in enumerate(pron):
        base = re.sub(r"\d", "", p)
        if base in VOWELS and re.search(r"\d", p):
            idxs.append(i)
    return idxs

def primary_stress_phone_index(pron):
    for i, p in enumerate(pron):
        if p.endswith("1"):
            return i
    for i, p in enumerate(pron):
        if p.endswith("2"):
            return i
    return None

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

# ========================= Prosódia =========================
def analyze_prosody_global(audio_path: str):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    intensity = snd.to_intensity()
    f0_values = pitch.selected_array['frequency']
    intensity_values = intensity.values[0] if intensity.values.size else np.array([])
    return np.array(f0_values), np.array(intensity_values)

def _mean_f0_for_segment(seg: AudioSegment) -> float:
    if len(seg) <= 0:
        return 0.0
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        seg.export(tmp.name, format="wav")
        snd = parselmouth.Sound(tmp.name)
        f0 = snd.to_pitch().selected_array['frequency']
        voiced = f0[f0 > 0]
        return float(np.mean(voiced)) if voiced.size > 0 else 0.0

def _zscore(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mu = np.mean(arr)
    sd = np.std(arr)
    if sd <= 1e-8:
        return np.zeros_like(arr)
    return (arr - mu) / sd

def detect_lexical_stress(word: str, word_audio_segment: AudioSegment):
    prons = cmu_pronunciations(word)
    if not prons:
        return None, "Sem pronúncia no CMUdict."
    pron = prons[0]
    syl_idxs = syllable_phone_indices(pron)
    if len(syl_idxs) < 2:
        return True, "Monossilábica ou sem contraste de stress."
    stressed_ph_idx = primary_stress_phone_index(pron)
    if stressed_ph_idx is None:
        return None, "Stress não marcado no CMUdict."
    stressed_syl = sum(1 for k in syl_idxs if k <= stressed_ph_idx) - 1
    n = len(syl_idxs)
    total_ms = int(len(word_audio_segment))
    if total_ms <= 0:
        return None, "Segmento de áudio vazio."
    seg_ms = max(MIN_SUBWINDOW_MS, total_ms // n)
    energies, f0_means = [], []
    for i in range(n):
        start = int(i * seg_ms)
        end = int((i + 1) * seg_ms) if i < n - 1 else total_ms
        blk = word_audio_segment[start:end]
        energies.append(blk.rms)
        f0_means.append(_mean_f0_for_segment(blk))
    z_en = _zscore(energies)
    z_f0 = _zscore(f0_means)
    scores = z_en + z_f0
    is_correct = (int(np.argmax(scores)) == int(stressed_syl))
    return is_correct, "Combinação RMS+F0 por sílaba (heurística)."

# ========================= Alinhamento & Fluência =========================
def align_token_sequences(ref_tokens, hyp_tokens):
    sm = SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    return sm.get_opcodes()

def compute_fluency_from_word_timestamps(word_scores):
    if not word_scores:
        return {"wpm": 0.0, "num_pauses": 0, "avg_pause": 0.0,
                "rate_stability": 0.0, "total_silence_inside": 0.0}
    t0 = float(word_scores[0]["start"])
    t1 = float(word_scores[-1]["end"])
    spoken_dur = max(1e-3, t1 - t0)
    spoken_words = [w for w in word_scores if re.search(r"\w", w["word"])]
    wpm = (len(spoken_words) / spoken_dur) * 60.0
    gaps = []
    for a, b in zip(word_scores, word_scores[1:]):
        gap = max(0.0, float(b["start"]) - float(a["end"]))
        if gap >= 0.2:
            gaps.append(gap)
    num_pauses = len(gaps)
    avg_pause = float(np.mean(gaps)) if gaps else 0.0
    total_silence_inside = float(np.sum(gaps)) if gaps else 0.0
    bursts_rates = []
    cur_count = 0
    burst_start = word_scores[0]["start"]
    last_end = word_scores[0]["end"]
    for w in word_scores[1:]:
        gap = float(w["start"]) - float(last_end)
        if gap >= 0.2:
            burst_dur = float(last_end) - float(burst_start)
            if burst_dur > 0 and cur_count > 0:
                bursts_rates.append(cur_count / burst_dur)
            burst_start = w["start"]
            cur_count = 1
        else:
            cur_count += 1
        last_end = w["end"]
    final_burst_dur = float(last_end) - float(burst_start)
    if final_burst_dur > 0 and cur_count > 0:
        bursts_rates.append(cur_count / final_burst_dur)
    rate_stability = float(np.std(bursts_rates)) if len(bursts_rates) > 1 else 0.0
    return {"wpm": float(wpm), "num_pauses": int(num_pauses), "avg_pause": float(avg_pause),
            "rate_stability": float(rate_stability), "total_silence_inside": float(total_silence_inside)}

# ========================= Confiança =========================
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

# ========================= Snippet de erro =========================
def extract_error_snippet(audio: AudioSegment, word_scores, opcodes):
    if not word_scores or not opcodes:
        return None
    def clamp_ms(x): return int(max(0, min(len(audio), x)))
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("replace", "insert", "delete"):
            if tag in ("replace", "insert") and j1 < j2 and j1 < len(word_scores):
                start = float(word_scores[max(0, j1)]["start"])
                end = float(word_scores[min(len(word_scores) - 1, j2 - 1)]["end"])
            else:
                if j1 < len(word_scores):
                    left_end = float(word_scores[max(0, j1 - 1)]["end"]) if j1 > 0 else float(word_scores[0]["start"])
                    right_start = float(word_scores[j1]["start"])
                    start, end = left_end, right_start
                    if end <= start:
                        start = float(word_scores[max(0, j1 - 1)]["start"])
                        end = float(word_scores[max(0, j1 - 1)]["end"])
                else:
                    start = float(word_scores[-1]["end"]) - 0.6
                    end = float(word_scores[-1]["end"])
            start_ms = clamp_ms(int((start - 0.5) * 1000))
            end_ms = clamp_ms(int((end + 0.5) * 1000))
            if end_ms <= start_ms:
                continue
            snippet = audio[start_ms:end_ms]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                snippet.export(tmp.name, format="wav")
                return tmp.name
    return None

# ========================= CTC: loader + mapeamento =========================
def load_phoneme_ctc_model(model_id=PHONEME_CTC_MODEL_ID, device=None):
    """
    Retorna: (processor, model, (id2label, label2id), device, vocab_tokens, has_multichar)
    Vocabulário é reconstruído a partir do tokenizer (fallback para config.id2label).
    """
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id)
        model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        vocab_tokens = set()
        id2label, label2id = {}, {}

        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "get_vocab"):
            vocab = tok.get_vocab()  # token -> id
            for token, idx in vocab.items():
                id2label[int(idx)] = token
            id2label = {i: id2label[i] for i in range(len(id2label))}
            label2id = {v: k for k, v in id2label.items()}
            vocab_tokens = set(label2id.keys())

        if len(vocab_tokens) < 5:
            if isinstance(model.config.id2label, dict):
                id2label_cfg = {int(k): v for k, v in model.config.id2label.items()}
            else:
                id2label_cfg = dict(enumerate(model.config.id2label))
            if len(id2label_cfg) >= 5:
                id2label = id2label_cfg
                label2id = {v: k for k, v in id2label.items()}
                vocab_tokens = set(label2id.keys())

        has_multichar = any(len(tok) > 1 for tok in vocab_tokens)
        print(f"[CTC] Carregado '{model_id}' | tokens={len(vocab_tokens)} | multi-char={has_multichar}")

        if len(vocab_tokens) < 5:
            print("[AVISO] Vocabulário do CTC muito pequeno. Considere outro checkpoint fonêmico.")

        return processor, model, (id2label, label2id), device, vocab_tokens, has_multichar

    except Exception as e:
        print(f"[AVISO] Falha ao carregar modelo CTC '{model_id}': {e}")
        return None, None, (None, None), None, set(), False

def is_vowel_arpabet(p): return re.sub(r"\d", "", p) in VOWELS
def strip_stress(p): return re.sub(r"\d", "", p)

ARPABET_TO_IPA_CANDIDATES = {
    # Vogais monoftongos
    "AA": ["ɑ"], "AE": ["æ"], "AH": ["ʌ", "ə"], "AO": ["ɔ"], "EH": ["ɛ"],
    "ER": ["ɝ", "ɚ"],  # prioridade ajustada por stress na função abaixo
    "IH": ["ɪ"], "IY": ["i"], "UH": ["ʊ"], "UW": ["u"],
    # Ditongos
    "AY": ["aɪ", "a"], "EY": ["eɪ", "e"], "OW": ["oʊ", "o"], "OY": ["ɔɪ", "ɔ"], "AW": ["aʊ", "a"],
    # Consoantes
    "P": ["p"], "B": ["b"], "T": ["t"], "D": ["d"], "K": ["k"], "G": ["ɡ"],
    "F": ["f"], "V": ["v"], "TH": ["θ"], "DH": ["ð"],
    "S": ["s"], "Z": ["z"], "SH": ["ʃ"], "ZH": ["ʒ"],
    "HH": ["h"], "M": ["m"], "N": ["n"], "NG": ["ŋ"],
    "L": ["l"], "R": ["ɹ", "r"], "Y": ["j"], "W": ["w"],
    # Africadas
    "CH": ["tʃ", "t͡ʃ", "ʧ", "ʃ", "t"],
    "JH": ["dʒ", "d͡ʒ", "ʤ", "ʒ", "d"],
}

def get_token_candidates_for_arpabet(arpabet_phone: str):
    base = strip_stress(arpabet_phone)
    stress_digit = "".join(ch for ch in arpabet_phone if ch.isdigit()) or None
    cands = list(ARPABET_TO_IPA_CANDIDATES.get(base, []))
    # --- AJUSTE (2): prioridade por stress para ER ---
    if base == "ER":
        if stress_digit == "1":
            cands = ["ɝ", "ɚ"] + [c for c in cands if c not in ("ɝ", "ɚ")]
        elif stress_digit == "0":
            cands = ["ɚ", "ɝ"] + [c for c in cands if c not in ("ɚ", "ɝ")]
    return cands

def choose_vocab_token_for_phone(arpabet_phone: str, vocab_tokens: set):
    cands = get_token_candidates_for_arpabet(arpabet_phone)
    # diretos
    for tok in cands:
        if tok in vocab_tokens:
            return tok
    # ditongos → núcleo
    SIMPLE_NUCLEUS = {"aɪ": "a", "eɪ": "e", "oʊ": "o", "ɔɪ": "ɔ", "aʊ": "a"}
    for tok in cands:
        if tok in SIMPLE_NUCLEUS and SIMPLE_NUCLEUS[tok] in vocab_tokens:
            return SIMPLE_NUCLEUS[tok]
    # africadas → fricativo
    AFRICATE_FRIC = {"tʃ": "ʃ", "t͡ʃ": "ʃ", "ʧ": "ʃ", "dʒ": "ʒ", "d͡ʒ": "ʒ", "ʤ": "ʒ"}
    for tok in cands:
        if tok in AFRICATE_FRIC and AFRICATE_FRIC[tok] in vocab_tokens:
            return AFRICATE_FRIC[tok]
    # ER → ɚ/ɝ/r como fallback
    if strip_stress(arpabet_phone) == "ER":
        for alt in ("ɚ", "ɝ", "r"):
            if alt in vocab_tokens:
                return alt
    # vogal → schwa
    if is_vowel_arpabet(arpabet_phone) and "ə" in vocab_tokens:
        return "ə"
    return None

def audiosegment_to_float_tensor(seg: AudioSegment, target_sr: int, device="cpu"):
    if seg.channels > 1:
        seg = seg.set_channels(1)
    seg = seg.set_frame_rate(target_sr)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    max_val = float(1 << (8 * seg.sample_width - 1))
    samples = samples / max_val
    wav = torch.from_numpy(samples).to(device)
    return wav

def composite_prob_if_better(token: str, probs_rowwise: torch.Tensor, label2id: dict):
    """
    Tenta melhorar probabilidade de tokens compostos usando componentes.
    Retorna float ou None.
    """
    decomps = {
        "aɪ": ["a", "ɪ"],
        "eɪ": ["e", "ɪ"],
        "oʊ": ["o", "ʊ"],
        "ɔɪ": ["ɔ", "ɪ"],
        "aʊ": ["a", "ʊ"],
        "tʃ": ["t", "ʃ"],
        "dʒ": ["d", "ʒ"],
    }
    if token not in decomps:
        return None
    idxs = [label2id.get(tk) for tk in decomps[token]]
    if any(i is None or i >= probs_rowwise.shape[-1] for i in idxs):
        return None
    comps = probs_rowwise[:, idxs]  # [T, 2]
    gmean = torch.clamp(comps, min=1e-8).log().mean(-1).exp()  # [T]
    return float(gmean.mean().cpu().item())

def mean_posterior_for_label_token(wav, processor, model, token: str, label2id: dict):
    """
    Probabilidade média do 'token' alvo ao longo dos frames (CTC),
    renormalizada SEM BLANK + composição para tokens compostos.
    """
    idx = label2id.get(token, None)
    if idx is None:
        return None
    with torch.no_grad():
        inputs = processor(
            wav.cpu().numpy(),
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=False
        )
        logits = model(inputs.input_values.to(model.device)).logits  # [B,T,V]
        probs = torch.softmax(logits, dim=-1)[0]                    # [T,V]

        # --- renormaliza sem blank ---
        blank_id = getattr(model.config, "pad_token_id", None)
        if blank_id is None and hasattr(processor, "tokenizer"):
            blank_id = getattr(processor.tokenizer, "pad_token_id", None)
        if blank_id is not None and 0 <= blank_id < probs.shape[-1]:
            p_blank = probs[:, blank_id:blank_id+1]
            denom = torch.clamp(1.0 - p_blank, min=1e-6)
            probs = probs / denom
            probs[:, blank_id] = 0.0
        # -----------------------------

        if idx >= probs.shape[-1]:
            return None

        p_tok = float(probs[:, idx].mean().cpu().item())

        # tenta composição para ditongos/africadas e pega o melhor
        p_comp = composite_prob_if_better(token, probs, label2id)
        if p_comp is not None:
            return max(p_tok, p_comp)
        return p_tok

def weighted_aggregate(scores, weights):
    s = 0.0; w = 0.0
    for sc, wt in zip(scores, weights):
        if sc is None:
            continue
        s += sc * wt; w += wt
    return (s / w) if w > 0 else None

def iter_equal_pairs_with_times(opcodes, target_tokens, hyp_tokens, word_scores):
    k = 0
    n_ws = len(word_scores)
    def clean_token(x): return re.sub(r"[^\w']", "", x.lower())
    while k < n_ws and not re.search(r"\w", word_scores[k]["word"]):
        k += 1
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            continue
        ti = i1; hj = j1
        while ti < i2 and hj < j2:
            t_tok = target_tokens[ti]; h_tok = hyp_tokens[hj]
            h_clean = clean_token(h_tok)
            while k < n_ws:
                ws_clean = clean_token(word_scores[k]["word"])
                if ws_clean == h_clean:
                    start = float(word_scores[k]["start"])
                    end = float(word_scores[k]["end"])
                    yield (t_tok, h_tok, start, end, k)
                    k += 1
                    break
                k += 1
            ti += 1; hj += 1

def word_pron_score_without_mfa(
    audio: AudioSegment,
    target_word: str,
    start_sec: float,
    end_sec: float,
    processor,
    model,
    id_maps,
    device="cpu",
    vocab_tokens: set = None
):
    if processor is None or model is None or not vocab_tokens:
        return None, []
    start_ms = int(max(0, start_sec * 1000))
    end_ms = int(min(len(audio), end_sec * 1000))
    if end_ms <= start_ms:
        return None, []
    seg = audio[start_ms:end_ms]

    prons = cmu_pronunciations(target_word)
    if prons:
        pron = prons[0]
        stress_idx = primary_stress_phone_index(pron)
    else:
        pron_str = get_phonemes_word(target_word)
        pron = [p for p in pron_str.split() if p]
        stress_idx = None
    if not pron:
        return None, []

    n = len(pron)
    seg_ms = max(MIN_SUBWINDOW_MS, (end_ms - start_ms) // n)

    weights = []
    for i, p in enumerate(pron):
        base = strip_stress(p)
        wt = 1.0
        if base in VOWELS:
            wt = 1.5
            if stress_idx is not None and i == stress_idx:
                wt = 2.0
        weights.append(wt)

    try:
        sr = processor.feature_extractor.sampling_rate
    except Exception:
        sr = 16000

    id2label, label2id = id_maps if id_maps and id_maps[0] is not None else ({}, {})
    details, block_scores = [], []
    for i, p in enumerate(pron):
        blk_start = start_ms + i * seg_ms
        blk_end = start_ms + (i + 1) * seg_ms if i < n - 1 else end_ms
        blk = audio[blk_start:blk_end]
        wav = audiosegment_to_float_tensor(blk, target_sr=sr, device=device)

        token = choose_vocab_token_for_phone(p, vocab_tokens)
        prob = None
        if token is not None:
            prob = mean_posterior_for_label_token(wav, processor, model, token, label2id)

        sc = None if prob is None else float(100.0 * prob)
        block_scores.append(sc)
        details.append({
            "arpabet": p,
            "token": token or "",
            "weight": weights[i],
            "posterior_mean": None if prob is None else float(prob),
            "score": sc
        })

    agg = weighted_aggregate(block_scores, weights)
    return (None if agg is None else float(agg)), details

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

    results = {}
    results["intelligibility"] = {
        "is_intelligible": bool(intelligibility),
        "wer": float(wer),
        "confidence": float(confidence),
        "transcription": hypothesis_text
    }

    # 3) Alinhamento & fonemas por palavra (relato)
    target_tokens = tokenize_words(target_text)
    hyp_tokens = tokenize_words(hypothesis_text)
    opcodes = align_token_sequences(target_tokens, hyp_tokens)
    target_phonemes = {w: get_phonemes_word(w) for w in target_tokens}
    hypothesis_phonemes = {w: get_phonemes_word(w) for w in hyp_tokens}
    results["phonetic_analysis"] = {
        "target_phonemes": target_phonemes,
        "hypothesis_phonemes": hypothesis_phonemes,
        "word_scores": word_scores
    }

    # 4) Feedback lexical (com atenuação morfológica)
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
                        "expected_phonemes": get_phonemes_word(tw),
                        "produced_phonemes": get_phonemes_word(hw),
                        "severity": "minor" if minor else "major",
                        "tip": "Atenção ao sufixo (-ed/-s/-ing)." if minor else "Verifique a pronúncia desta palavra."
                    })
        elif tag == "delete":
            for ti in range(i1, i2):
                tw = target_tokens[ti]
                phone_feedback.append({
                    "word": tw,
                    "issue": "omissão",
                    "expected_phonemes": get_phonemes_word(tw),
                    "produced_phonemes": "",
                    "severity": "major",
                    "tip": "A palavra parece ter sido omitida."
                })
    results["phone_feedback"] = phone_feedback

    # 5) Fluência
    fluency = compute_fluency_from_word_timestamps(word_scores)
    wpm_penalty = 0.0
    if fluency["wpm"] < 60.0:
        wpm_penalty = min(20.0, (60.0 - fluency["wpm"]) * 0.3)
    elif fluency["wpm"] > 180.0:
        wpm_penalty = min(20.0, (fluency["wpm"] - 180.0) * 0.1)
    results["fluency"] = {**fluency, "wpm_penalty": float(wpm_penalty)}

    # 6) Prosódia
    f0_values, intensity_values = analyze_prosody_global(audio_path)
    voiced = f0_values[f0_values > 0]
    f0_variance = float(np.var(voiced)) if voiced.size > 0 else 0.0
    audio = AudioSegment.from_file(audio_path)
    stress_ok_count = 0
    total_polysyllabic_words = 0
    stress_details = []
    for w in word_scores:
        word = re.sub(r"[^\w\s]", "", (w["word"] or "")).lower()
        if not word:
            continue
        prons = cmu_pronunciations(word)
        if not prons:
            continue
        syl_idxs = syllable_phone_indices(prons[0])
        if len(syl_idxs) < 2:
            continue
        total_polysyllabic_words += 1
        start_ms = int(max(0.0, float(w["start"]) * 1000))
        end_ms = int(min(len(audio), float(w["end"]) * 1000))
        if end_ms <= start_ms:
            continue
        word_audio_segment = audio[start_ms:end_ms]
        is_correct, reason = detect_lexical_stress(word, word_audio_segment)
        stress_details.append({"word": word, "correct_stress": is_correct, "reason": reason})
        if is_correct:
            stress_ok_count += 1
    stress_ok_percentage = float(100.0 if total_polysyllabic_words == 0
                                 else (100.0 * stress_ok_count / total_polysyllabic_words))
    rate_stability = fluency.get("rate_stability", 0.0)
    rhythm_note = "Prosódia adequada."
    if f0_variance < 500.0:
        rhythm_note = "Entonação monótona detectada."
    elif rate_stability > 0.5:
        rhythm_note = "Ritmo inconsistente."
    results["prosody"] = {
        "f0_variance": f0_variance,
        "stress_ok_percentage": stress_ok_percentage,
        "stress_details": stress_details,
        "rhythm_note": rhythm_note
    }

    # 7) CTC — carregar e calcular score por palavra
    processor, phone_model, id_maps, device, vocab_tokens, has_multichar = load_phoneme_ctc_model(PHONEME_CTC_MODEL_ID)
    per_word_scores = []
    ctc_ready = (processor is not None and phone_model is not None and vocab_tokens)

    results["ctc_status"] = {
        "loaded": bool(ctc_ready),
        "vocab_size": int(len(vocab_tokens or [])),
        "has_multichar_tokens": bool(has_multichar),
        "model_id": PHONEME_CTC_MODEL_ID
    }

    # pares equal com tempo + cálculo de score
    for (t_tok, h_tok, st, en, idx_ws) in iter_equal_pairs_with_times(opcodes, target_tokens, hyp_tokens, word_scores):
        sc_ctc, det = (None, [])
        if ctc_ready:
            sc_ctc, det = word_pron_score_without_mfa(
                audio=audio,
                target_word=t_tok,
                start_sec=st,
                end_sec=en,
                processor=processor,
                model=phone_model,
                id_maps=id_maps,
                device=device or "cpu",
                vocab_tokens=vocab_tokens
            )

        # Fallback ASR
        ws_conf = word_scores[idx_ws].get("confidence", None)
        sc_asr = None if ws_conf is None else float(ws_conf * 100.0)

        # --- AJUSTE (3): Score híbrido 50% CTC + 50% ASR ---
        if sc_ctc is not None and sc_asr is not None:
            sc = 0.5 * sc_ctc + 0.5 * sc_asr
        elif sc_ctc is not None:
            sc = sc_ctc
        else:
            sc = sc_asr  # pode ser None se ASR sem confiança

        per_word_scores.append({
            "word": t_tok, "start": st, "end": en, "score": sc, "details": det
        })

    results["word_pron_scores"] = per_word_scores

    # 8) Target proximity
    valid_word_scores = [w["score"] for w in per_word_scores if w["score"] is not None]
    if valid_word_scores:
        phonetic_score = float(np.mean(valid_word_scores))
    else:
        phonetic_score = (1.0 - wer) * 100.0
    fluency_score = max(0.0, 100.0 - (fluency["num_pauses"] * 5.0) - (rate_stability * 20.0) - wpm_penalty)
    prosody_score = stress_ok_percentage
    target_proximity = (0.7 * phonetic_score) + (0.15 * fluency_score) + (0.15 * prosody_score)
    target_proximity = float(max(0.0, min(100.0, target_proximity)))
    results["target_proximity"] = target_proximity

    # 9) Snippet de erro
    error_audio_snippet_path = extract_error_snippet(audio, word_scores, opcodes)
    results["error_audio_snippet_path"] = error_audio_snippet_path

    print(json.dumps(results, ensure_ascii=False, indent=4))

    # suponha que 'json_str' seja a string com o JSON do pipeline
    # resumo = simple_feedback_from_json(json.dumps(results, ensure_ascii=False, indent=4), threshold=45.0, include_feedback=True)
    # print(resumo)


