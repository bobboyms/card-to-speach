import re
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from app.models.word_segment import WordSegment

logger = logging.getLogger(__name__)

class AlignmentFailedError(RuntimeError):
    """Raised when CTC backtracking cannot build a valid alignment path."""

@dataclass
class _Point:
    token_index: int
    time_index: int
    score: float  # prob (não log)

@dataclass
class _Segment:
    label: str
    start: int
    end: int
    score: float
    @property
    def length(self) -> int:
        return self.end - self.start

class ForcedAlignmentService:
    """
    Alinhamento CTC (trellis + backtracking) com Wav2Vec2 (torchaudio).
    """
    def __init__(self, device: str | None = None, min_word_len_s: float = 0.015):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_word_len_s = min_word_len_s
        self.processor = None
        self.model = None
        self._bundle_sr: int | None = None

    def _ensure_model(self):
        if self.model is None:
            logger.info("Carregando wav2vec2 model e processor para alinhamento...")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device).eval()
            self._bundle_sr = 16000

    @staticmethod
    def _normalize_transcript(text: str) -> str:
        text_ascii = text.upper().encode("ascii", "ignore").decode("ascii")
        text_ascii = re.sub(r"[^A-Z'\s]+", " ", text_ascii)
        text_ascii = re.sub(r"\s+", " ", text_ascii).strip()
        return "|" + text_ascii.replace(" ", "|") + "|"

    def _emission(self, waveform: torch.Tensor) -> torch.Tensor:
        self._ensure_model()
        assert self.model is not None
        assert self.processor is not None
        with torch.inference_mode():
            # processor expects input values. If tensor, we pass it directly.
            # We assume waveform is [1, T] or [T].
            wav_input = waveform.squeeze()
            if wav_input.ndim > 1:
                 wav_input = wav_input.mean(dim=0)
            
            inputs = self.processor(wav_input, sampling_rate=16000, return_tensors="pt")
            logits = self.model(inputs.input_values.to(self.device)).logits
            emissions = torch.log_softmax(logits, dim=-1)
        return emissions[0].cpu()  # [T, C]

    def _text_to_tokens(self, transcript_bar: str) -> List[int]:
        if self.processor is None:
            self._ensure_model()
        assert self.processor is not None
        vocab = self.processor.tokenizer.get_vocab()
        return [vocab[c] for c in transcript_bar]

    @staticmethod
    def _trellis(emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> torch.Tensor:
        T = emission.size(0)
        J = len(tokens)
        trellis = torch.full((T, J), -float("inf"))

        # Initial frame (t=0)
        # Start at the first blank
        trellis[0, 0] = emission[0, tokens[0]]
        # Or the first token if it exists (optional, but standard CTC allows starting at blank or first symbol)
        if J > 1:
            trellis[0, 1] = emission[0, tokens[1]]

        for t in range(1, T):
            # Emission scores for all states at time t
            emission_t = emission[t, tokens]
            
            # Previous scores
            prev = trellis[t - 1]
            
            # Shifted previous scores for transition from j-1
            # We prepend -inf to align indices: prev_shifted[j] corresponds to prev[j-1]
            prev_shifted = torch.cat([torch.tensor([-float("inf")]).to(prev.device), prev[:-1]])
            
            # Update trellis: max(stay, change) + emission
            trellis[t] = emission_t + torch.maximum(prev, prev_shifted)

        return trellis

    @staticmethod
    def _backtrack(trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> List[_Point]:
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        
        # Check if the final state is valid
        if trellis[t, j] == -float("inf"):
             raise AlignmentFailedError("No valid path found in trellis.")

        prob = emission[t, tokens[j]].exp().item()
        path = [_Point(j, t, prob)]

        while t > 0:
            # Determine if we stayed (j) or changed (j-1)
            step_stay = trellis[t - 1, j]
            step_change = -float("inf")
            if j > 0:
                step_change = trellis[t - 1, j - 1]
            
            # Greedy choice: if change is possible and has better or equal score, we might take it.
            # Standard Viterbi takes max.
            if j > 0 and step_change >= step_stay:
                j -= 1
            
            t -= 1
            prob = emission[t, tokens[j]].exp().item()
            path.append(_Point(j, t, prob))

        return path[::-1]

    @staticmethod
    def _merge_repeats(path: List[_Point], transcript: str) -> List[_Segment]:
        i1 = i2 = 0
        segs: List[_Segment] = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / max(1, (i2 - i1))
            segs.append(_Segment(transcript[path[i1].token_index],
                                 path[i1].time_index, path[i2 - 1].time_index + 1, score))
            i1 = i2
        return segs

    @staticmethod
    def _merge_words(segs: List[_Segment], sep: str = "|") -> List[_Segment]:
        out: List[_Segment] = []
        i1 = i2 = 0
        while i1 < len(segs):
            if i2 >= len(segs) or segs[i2].label == sep:
                if i1 != i2:
                    group = segs[i1:i2]
                    label = "".join(s.label for s in group)
                    score = sum(s.score * s.length for s in group) / max(1, sum(s.length for s in group))
                    out.append(_Segment(label, group[0].start, group[-1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return out

    def align(self, wav_path: str, transcript_text: str) -> List[WordSegment]:
        self._ensure_model()
        # carrega waveform e resample se necessário
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        emission = self._emission(waveform)
        transcript_bar = self._normalize_transcript(transcript_text)
        tokens = self._text_to_tokens(transcript_bar)

        # Trellis needs more frames than tokens; otherwise short or silent audios have no path.
        if emission.size(0) <= len(tokens):
            logger.warning(
                "Alinhamento ignorado: audio com %d frames e transcript com %d tokens.",
                emission.size(0),
                len(tokens),
            )
            return []

        trellis = self._trellis(emission, tokens, blank_id=0)

        if not torch.isfinite(trellis[-1, -1]):
            logger.warning("Alinhamento inválido: caminho final inatingível (trellis[-1, -1] = %s).", trellis[-1, -1])
            return []

        try:
            path = self._backtrack(trellis, emission, tokens, blank_id=0)
        except AlignmentFailedError as exc:
            logger.warning("Alinhamento inválido ao reconstruir caminho: %s", exc)
            return []
        segs = self._merge_repeats(path, transcript_bar)
        word_segs = self._merge_words(segs, sep="|")

        # frames -> segundos
        n_frames = emission.size(0)
        n_samples = waveform.size(1)
        assert self._bundle_sr is not None
        ratio = (n_samples / self._bundle_sr) / float(n_frames)  # (duração em s) / n_frames

        words: List[WordSegment] = []
        for w in word_segs:
            label = w.label.replace("|", "").strip()
            if not label:
                continue
            start_s = w.start * ratio
            end_s = w.end * ratio
            if (end_s - start_s) >= self.min_word_len_s:
                words.append(WordSegment(label, start_s, end_s, w.score))
        logger.info("Alinhadas %d palavras.", len(words))
        return words
