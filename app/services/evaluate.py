# -*- coding: utf-8 -*-
"""Pronunciation evaluation utilities backed by :class:`PronEvaluator`."""

from __future__ import annotations

import base64
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from app.models.word_segment import WordSegment
from app.services.pron_evaluator import PronEvaluator
from app.services.alignment_service import ForcedAlignmentService
from app.services.audio_service import AudioService, AudioServiceError


class PronunciationEvaluationService:
    """Evaluate pronunciation quality for recorded speech segments."""

    QUALITY_WEIGHTS: ClassVar[Dict[str, float]] = {
        "perfect": 1.0,
        "high": 0.80,
        "medium": 0.5,
        "low": 0.0,
    }
    QUALITY_THRESHOLDS: ClassVar[Sequence[float]] = (0.90, 0.80, 0.70)
    OPS_PENALTY: ClassVar[float] = 0.05
    MAX_OPS_PENALTY: ClassVar[float] = 0.15

    def __init__(
        self,
        *,
        audio_service: AudioService | None = None,
        aligner: ForcedAlignmentService | None = None,
        pron_evaluator: PronEvaluator | None = None,
        weight_per: float = 0.6,
        weight_gop: float = 0.4,
        segment_format: str = "mp3",
    ) -> None:
        self._audio_service = audio_service or AudioService(
            target_sr=16000,
            mono=True,
            apply_fade_ms=3,
        )
        self._aligner = aligner or ForcedAlignmentService(min_word_len_s=0.015)
        self._pron_evaluator = pron_evaluator or PronEvaluator(
            global_gop_threshold=0.4,
            gop_rival_scope="same_class",
            word_ok_threshold=0.8,
            war_tolerant_alpha=0.8,
            war_hybrid_alpha=0.8,
            war_hybrid_beta=0.2,
        )
        self._weight_per = weight_per
        self._weight_gop = weight_gop
        self._segment_format = segment_format

    def evaluate(
        self,
        audio_path: str,
        target_text: str,
        phoneme_format: str = "ipa",
    ) -> Dict[str, Any]:
        """Run the pronunciation pipeline and return aggregated metrics."""
        del phoneme_format  # placeholder for future phoneme format handling

        wav_path = self._audio_service.convert_to_wav(audio_path)
        word_audio_segments = self._segment_word_audio(wav_path, target_text)
        evaluation = self._pron_evaluator.evaluate(wav_path, target_text)

        return self._format_output(evaluation["sentence_metrics"], word_audio_segments)

    def _segment_word_audio(
        self,
        wav_path: str,
        target_text: str,
    ) -> List[Optional[str]]:
        aligned_words: List[WordSegment] = self._aligner.align(wav_path, target_text)
        segments: List[Optional[str]] = []
        for segment in aligned_words:
            try:
                audio_bytes = self._audio_service.cut_precise_to_bytes(
                    wav_path,
                    segment.start,
                    segment.end,
                    fmt=self._segment_format,
                )
            except AudioServiceError:
                segments.append(None)
                continue

            if audio_bytes:
                segments.append(base64.b64encode(audio_bytes).decode("ascii"))
            else:
                segments.append(None)
        return segments

    def _format_output(
        self,
        metrics: Dict[str, Any],
        segments: Sequence[Optional[str]],
    ) -> Dict[str, Any]:
        thresholds = self.QUALITY_THRESHOLDS
        words: List[Dict[str, Any]] = []

        for index, word_metrics in enumerate(metrics.get("words", [])):
            per_score = float(word_metrics.get("word_score_per", 0.0))
            gop_score = float(word_metrics.get("word_score_gop", 0.0))
            operations = [
                op
                for op in word_metrics.get("per_ops", [])
                if isinstance(op, str) and op != "match"
            ]

            if not operations and per_score == 1.0:
                gop_score = 1.0

            score = (
                self._weight_per * per_score
                + self._weight_gop * gop_score
                - self._penalize_ops(operations)
            )
            score = max(0.0, min(1.0, score))
            quality = self._classify_score(score, thresholds)

            word_payload = {
                "target_word": word_metrics.get("target_word"),
                "target_phones": word_metrics.get("target_phones", []),
                "produced_phone": word_metrics.get("produced_phone", []),
                "per": per_score,
                "gop": gop_score,
                "ops": operations,
                "pronunciation_quality": quality,
                "audio_b64": segments[index] if index < len(segments) else None,
            }
            words.append(word_payload)

        return {
            "intelligibility": float(metrics.get("intelligibility", 0.0)),
            "word_accuracy_rate": self._weighted_word_accuracy(words),
            "fluency_level": metrics.get("fluency", {}).get("level"),
            "words": words,
        }

    def _penalize_ops(self, operations: Sequence[str]) -> float:
        penalized_ops = sum(
            1
            for op in operations
            if op.startswith("del(") or op.startswith("ins(")
        )
        penalty = penalized_ops * self.OPS_PENALTY
        return min(penalty, self.MAX_OPS_PENALTY)

    def _classify_score(
        self,
        score: float,
        thresholds: Sequence[float],
    ) -> str:
        perfect, high, medium = thresholds
        if score >= perfect:
            return "perfect"
        if score >= high:
            return "high"
        if score >= medium:
            return "medium"
        return "low"

    def _weighted_word_accuracy(self, words: Sequence[Dict[str, Any]]) -> float:
        if not words:
            return 0.0
        total = 0.0
        for word in words:
            quality = str(word.get("pronunciation_quality", "")).lower()
            total += self.QUALITY_WEIGHTS.get(quality, 0.0)
        return total / len(words)


_DEFAULT_SERVICE = PronunciationEvaluationService()


def evaluate_pronunciation(
    audio_path: str,
    target_text: str,
    phoneme_fmt: str = "ipa",
    *,
    service: PronunciationEvaluationService | None = None,
) -> Dict[str, Any]:
    """Convenience wrapper to evaluate pronunciation using the default service."""
    evaluator = service or _DEFAULT_SERVICE
    return evaluator.evaluate(audio_path, target_text, phoneme_fmt)

