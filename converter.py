# -*- coding: utf-8 -*-
"""
Força de alinhamento CTC (Wav2Vec2/torchaudio) para cortes palavra a palavra.
- Usa Whisper apenas para obter o texto.
- Usa CTC (trellis + backtracking) para achar os frames de cada token/palavra.
- Converte frames em segundos e corta com FFmpeg (atrim + asetpts).
"""
import os
import re
import math
import wave
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
import whisper_timestamped as whisper_ts

from app.models.word_segment import WordSegment
from app.services.alignment_service import ForcedAlignmentService
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.utils.b64 import mp3_to_base64

# ------------------------
# Configurações
# ------------------------
DEFAULT_MODEL = "small"        # Whisper (tiny/base/small/medium/large)
TARGET_SR = 16000              # 16 kHz
APPLY_FADE = True
FADE_MS = 3                    # micro-fade p/ evitar clique
MIN_WORD_LEN_S = 0.015         # descartar palavras ultracurtas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Utilidades gerais
# ------------------------
def sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-_\.]", "_", s, flags=re.UNICODE)
    return s[:40] or "word"

def webm_to_wav_16k(input_path: str,
                    output_path: Optional[str] = None,
                    mono: bool = True,
                    overwrite: bool = True) -> str:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg não encontrado no PATH.")
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".wav"
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Arquivo de saída já existe: {output_path}")

    cmd = ["ffmpeg", "-y" if overwrite else "-n", "-i", input_path, "-vn",
           "-acodec", "pcm_s16le", "-ar", str(TARGET_SR)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(output_path)

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # valida
    with wave.open(output_path, "rb") as wf:
        if wf.getframerate() != TARGET_SR: raise RuntimeError("SR diferente de 16 kHz.")
        if mono and wf.getnchannels() != 1: raise RuntimeError("Não é mono.")
        if wf.getsampwidth() != 2: raise RuntimeError("Não é PCM 16-bit.")
    return output_path

def export_word_with_ffmpeg(input_file: str, start: float, end: float, out_file: str) -> None:
    dur = max(0.0, end - start)
    if dur <= 0: return
    if APPLY_FADE and dur > (2 * FADE_MS / 1000.0):
        fi = FADE_MS / 1000.0
        fo = FADE_MS / 1000.0
        filter_expr = (
            f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fi:.6f},afade=t=out:st={dur-fo:.6f}:d={fo:.6f}"
        )
    else:
        filter_expr = f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS"

    cmd = ["ffmpeg", "-y", "-i", input_file, "-filter_complex", filter_expr,
           "-c:a", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "1", out_file]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg erro: {e.stderr.decode(errors='ignore')}")

# ------------------------
# Whisper: obter transcrição bruta (texto)
# ------------------------
def transcribe_text_with_whisper(wav_path: str, language: str = "en", vad: bool = True) -> str:
    print("Carregando Whisper...")
    model = whisper_ts.load_model(DEFAULT_MODEL)
    audio = whisper_ts.load_audio(wav_path)
    print("Transcrevendo (só texto)...")
    result = whisper_ts.transcribe(model, audio, language=language, vad=vad)
    text = "".join(seg.get("text", "") for seg in result.get("segments", []))
    return text.strip()

# ------------------------
# Torchaudio / CTC Forced Alignment
# ------------------------
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float  # prob (não log) do passo escolhido

@dataclass
class Segment:
    label: str
    start: int  # frame idx (trellis)
    end: int    # frame idx (trellis, exclusivo)
    score: float
    @property
    def length(self) -> int:
        return self.end - self.start

def normalize_transcript_for_bundle(text: str) -> str:
    """
    Normaliza o texto para o alfabeto do bundle ASR (A..Z, ' e separador de palavras '|').
    - Uppercase
    - Remove acentos/caracteres fora do set
    - Troca espaços por '|'
    - Envolve com '|' (SOS/EOS), como no tutorial
    """
    # remove acentos de forma simples
    text_ascii = text.upper()
    text_ascii = text_ascii.encode("ascii", "ignore").decode("ascii")
    # keep A-Z, apostrophe e espaço
    text_ascii = re.sub(r"[^A-Z'\s]+", " ", text_ascii)
    text_ascii = re.sub(r"\s+", " ", text_ascii).strip()
    text_bar = "|" + text_ascii.replace(" ", "|") + "|"
    return text_bar

def ctc_emissions_and_labels(waveform: torch.Tensor, sr: int):
    """
    Emissões (log-softmax) e rótulos do bundle wav2vec2 ASR Base (inglês).
    """
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(DEVICE).eval()
    labels = bundle.get_labels()
    with torch.inference_mode():
        emissions, _ = model(waveform.to(DEVICE))
        emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu()  # [T, C]
    return emission, labels, bundle.sample_rate

def text_to_tokens(transcript_bar: str, labels: Tuple[str, ...]) -> List[int]:
    dictionary = {c: i for i, c in enumerate(labels)}
    return [dictionary[c] for c in transcript_bar]

def get_trellis(emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> torch.Tensor:
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")
    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],            # stay
            trellis[t, :-1] + emission[t, tokens[1:]]          # change
        )
    return trellis

def backtrack(trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> List[Point]:
    t, j = trellis.size(0) - 1, trellis.size(1) - 1
    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        assert t > 0
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change
        t -= 1
        if changed > stayed:
            j -= 1
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))
    # preencher até o início (só visual/consistência)
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(0, t - 1, prob))
        t -= 1
    return path[::-1]

def merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    i1, i2 = 0, 0
    segments: List[Segment] = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / max(1, (i2 - i1))
        segments.append(Segment(
            label=transcript[path[i1].token_index],
            start=path[i1].time_index,
            end=path[i2 - 1].time_index + 1,
            score=score
        ))
        i1 = i2
    return segments

def merge_words(segments: List[Segment], separator: str = "|") -> List[Segment]:
    words: List[Segment] = []
    i1 = i2 = 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join(s.label for s in segs)
                score = sum(s.score * s.length for s in segs) / max(1, sum(s.length for s in segs))
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def frames_to_seconds(word_segments: List[Segment], n_frames: int, n_samples: int, sr_wave: int) -> List[Dict]:
    # razão: (n_amostras / sr) / n_frames
    ratio = (n_samples / sr_wave) / float(n_frames)
    out = []
    for w in word_segments:
        start_s = w.start * ratio
        end_s = w.end * ratio
        out.append({"word": w.label, "start": start_s, "end": end_s, "score": w.score})
    return out

# ------------------------
# Pipeline: alinhar e cortar
# ------------------------
def forced_align_words(wav_path: str, transcript_text: str) -> List[Dict]:
    """
    Retorna lista de palavras {word, start, end, score} (em segundos) via CTC.
    """
    # carregar waveform como tensor (torchaudio.load mantém SR original)
    waveform, sr = torchaudio.load(wav_path)  # [1, N]
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # mono
    # Reamostrar para SR do bundle, se necessário
    emission, labels, bundle_sr = ctc_emissions_and_labels(waveform, sr)

    # normalizar transcrição para o alfabeto do bundle
    transcript_bar = normalize_transcript_for_bundle(transcript_text)
    tokens = text_to_tokens(transcript_bar, labels)

    trellis = get_trellis(emission, tokens, blank_id=0)
    path = backtrack(trellis, emission, tokens, blank_id=0)
    segments = merge_repeats(path, transcript_bar)
    word_segments = merge_words(segments, separator="|")

    # converter frames -> segundos no domínio do WAV original
    n_frames = emission.size(0)
    n_samples = waveform.size(1)
    words_sec = frames_to_seconds(word_segments, n_frames, n_samples, bundle_sr)  # usa sr do bundle para relação tempo

    # limpar palavras (remover tokens inválidos, nome legível)
    for w in words_sec:
        w["word"] = w["word"].replace("|", "").strip()
    words_sec = [w for w in words_sec if w["word"] and (w["end"] - w["start"]) >= MIN_WORD_LEN_S]
    return words_sec

def cut_words_with_alignment(input_wav: str, words: List[Dict], out_dir: str = "words_ctc") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, w in enumerate(words):
        name = sanitize_filename(w["word"])
        out_path = os.path.join(out_dir, f"{i+1:04d}_{name}_{w['start']:.3f}-{w['end']:.3f}.wav")
        export_word_with_ffmpeg(input_wav, w["start"], w["end"], out_path)
        print("Escreveu:", out_path)

# ------------------------
# main
# ------------------------



if __name__ == "__main__":
    audio = AudioService(target_sr=16000, mono=True, apply_fade_ms=3)
    transcriber = TranscriptionService(whisper_model="small")
    aligner = ForcedAlignmentService(min_word_len_s=0.015)

    # 1) converter
    wav_path = audio.convert_to_wav("meu_audio.webm")
    # 2) transcrever texto (ou use seu próprio texto limpo)
    text = transcriber.transcribe_text(wav_path, language="en", vad=True)
    # 3) alinhar (CTC)
    words: list[WordSegment] = aligner.align(wav_path, text[0])

    # 4) cortar
    phrases_cut = []
    os.makedirs("words_by_ctc", exist_ok=True)
    for i, w in enumerate(words):
        name = AudioService.sanitize_filename(w.word)
        out_path = os.path.join("words_by_ctc", f"{i + 1:04d}_{name}_{w.start:.3f}-{w.end:.3f}.mp3")
        audio.cut_precise(wav_path, w.start, w.end, out_path)
        phrases_cut.append({
            "word": w.word,
            "out_path": out_path,
        })

    print("Phrases cut:", phrases_cut)

    b64 = mp3_to_base64(phrases_cut[0]["out_path"])

    # audio_original = "meu_audio.webm"  # ajuste aqui
    #
    # print("Convertendo para WAV 16 kHz mono...")
    # wav_path = webm_to_wav_16k(audio_original)
    # print("OK:", wav_path)
    #
    # # 1) Texto com Whisper (você pode também fornecer seu próprio texto limpo)
    # text = transcribe_text_with_whisper(wav_path, language="en", vad=True)
    # print("Texto (Whisper):", text)
    #
    # # 2) Alinhamento forçado (CTC) e tempos por palavra
    # print("Alinhando com CTC (trellis/backtracking)...")
    # words_aligned = forced_align_words(wav_path, text)
    # print(f"{len(words_aligned)} palavras alinhadas.")
    #
    # # 3) Corte preciso com FFmpeg
    # cut_words_with_alignment(wav_path, words_aligned, out_dir="words_by_ctc")
    #
    # print("Concluído.")
