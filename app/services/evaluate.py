# -*- coding: utf-8 -*-
"""Pronunciation evaluation utilities backed by gRPC service."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict

import grpc
from app.services import service_pb2, service_pb2_grpc
from app.schemas import EvalResponse
from app import config


def _file_to_b64(path: Path | str) -> str:
    p = Path(path)
    with p.expanduser().resolve().open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def call_evaluate(
    *,
    endpoint: str,
    audio_path: Path | str,
    target_text: str,
    phoneme_format: str = "ipa",
    timeout: float | None = 30.0,
) -> service_pb2.EvaluateResponse:
    channel = grpc.insecure_channel(endpoint)
    stub = service_pb2_grpc.PronunciationServiceStub(channel)
    request = service_pb2.EvaluateRequest(
        audio_b64=_file_to_b64(audio_path),
        target_text=target_text,
        phoneme_format=phoneme_format,
    )
    return stub.Evaluate(request, timeout=timeout)


def response_to_dict(resp: service_pb2.EvaluateResponse) -> dict:
    return {
        "intelligibility": resp.intelligibility.score,
        "word_accuracy_rate": resp.intelligibility.word_accuracy_rate,
        "fluency_level": resp.phonetic_analysis.fluency_level,
        "fluency_score": resp.phonetic_analysis.fluency_score,
        "words": [
            {
                "target_word": w.target_word,
                "target_phones": list(w.target_phones),
                "produced_phone": list(w.produced_phone),
                "per": w.per,
                "gop": w.gop,
                "ops": list(w.ops),
                "pronunciation_quality": w.pronunciation_quality,
                "audio_b64": w.audio_b64,
            }
            for w in resp.phonetic_analysis.words
        ],
        "meta": dict(resp.meta),
    }


def evaluate_pronunciation(
    audio_path: str,
    target_text: str,
    phoneme_fmt: str = "ipa",
) -> Dict[str, Any]:
    """Convenience wrapper to evaluate pronunciation using the gRPC service."""
    endpoint = config.GRPC_ENDPOINT
    resp = call_evaluate(
        endpoint=endpoint,
        audio_path=audio_path,
        target_text=target_text,
        phoneme_format=phoneme_fmt,
        timeout=30.0,
    )
    return response_to_dict(resp)


def format_eval_response(results: Dict[str, Any], phoneme_fmt: str) -> EvalResponse:
    """Build the EvalResponse payload expected by the public API."""
    words: list[Dict[str, Any]] = list(results.get("words") or [])
    intelligibility_score = float(results.get("intelligibility", 0.0))
    word_accuracy_rate = float(results.get("word_accuracy_rate", 0.0))
    fluency_level = results.get("fluency_level")
    fluency_score = results.get("fluency_score")

    return EvalResponse(
        intelligibility={
            "score": intelligibility_score,
            "word_accuracy_rate": word_accuracy_rate,
        },
        phonetic_analysis={
            "fluency_level": fluency_level,
            "fluency_score": fluency_score,
            "words": words,
        },
        meta={
            "phoneme_fmt": phoneme_fmt,
        },
    )
