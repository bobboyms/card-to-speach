import logging
from typing import Optional, Any

import numpy as np
import whisper_timestamped as whisper_ts


class TranscriptionService:
    def __init__(self, whisper_model: str = "small"):
        self.whisper_model = whisper_model
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = whisper_ts.load_model(self.whisper_model)

    def transcribe_text(self, wav_path: str, language: Optional[str] = "en", vad: bool = True) -> tuple[
        Any, list[Any], list[Any], float | None]:
        self._ensure_model()
        audio = whisper_ts.load_audio(wav_path)
        result = whisper_ts.transcribe(self._model, audio, language=language, vad=vad)
        # text = "".join(seg.get("text", "") )

        word_scores = []
        word_probs = []
        for seg in result.get("segments", []):
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

        seg_avg_logprobs = []
        if "segments" in result:
            for seg in result["segments"]:
                if "avg_logprob" in seg and seg["avg_logprob"] is not None:
                    seg_avg_logprobs.append(float(seg["avg_logprob"]))
        avg_logprob = float(np.mean(seg_avg_logprobs)) if seg_avg_logprobs else None

        return result["text"], word_scores, word_probs, avg_logprob
