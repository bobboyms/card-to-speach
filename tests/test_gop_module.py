import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:
    import soundfile as sf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback para ambientes sem soundfile
    sf = types.SimpleNamespace(read=None, write=None)
    sys.modules["soundfile"] = sf

try:
    from scipy.signal import resample_poly as _resample_poly  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback simples para resample
    def _resample_poly(x, up, down):
        n = int(round(len(x) * up / down))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if len(x) == 0:
            return np.zeros(n, dtype=np.float32)
        old_idx = np.linspace(0, 1, len(x), endpoint=False)
        new_idx = np.linspace(0, 1, n, endpoint=False)
        return np.interp(new_idx, old_idx, x).astype(np.float32)

    fake_signal = types.SimpleNamespace(resample_poly=_resample_poly)
    fake_scipy = types.ModuleType("scipy")
    fake_scipy.signal = fake_signal
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.signal"] = fake_signal

try:
    from g2p_en import G2p  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback mínimo de G2P
    class _DummyG2p:
        def __call__(self, word):
            return []

    fake_g2p = types.SimpleNamespace(G2p=lambda: _DummyG2p())
    sys.modules["g2p_en"] = fake_g2p

if "torch" not in sys.modules:  # pragma: no cover - stub leve de torch
    class _NullCtx:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    def _argmax(x, axis=-1):
        return np.argmax(x, axis=axis)

    fake_torch = types.SimpleNamespace(
        device=lambda kind: types.SimpleNamespace(type=kind),
        inference_mode=lambda: _NullCtx(),
        autocast=lambda **_: _NullCtx(),
        log_softmax=lambda x, dim=-1: x,
        argmax=lambda x, dim=-1: _argmax(x, axis=dim),
        compile=lambda model: model,
        float16=np.float16,
        float32=np.float32,
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    sys.modules["torch"] = fake_torch

if "transformers" not in sys.modules:  # pragma: no cover - stub leve
    class _FakeProcessor:
        def __init__(self):
            self.tokenizer = types.SimpleNamespace(
                get_vocab=lambda: {"<pad>": 0},
                pad_token_id=0,
            )

        @staticmethod
        def from_pretrained(model_id):
            return _FakeProcessor()

    class _FakeModel:
        def __init__(self):
            self.logits = np.zeros((1, 1))

        def to(self, device):
            return self

        def eval(self):
            return self

    class _FakeModelForCTC:
        @staticmethod
        def from_pretrained(model_id):
            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = _FakeProcessor
    fake_transformers.AutoModelForCTC = _FakeModelForCTC
    sys.modules["transformers"] = fake_transformers

import app.services.pron_evaluator as gop
from app.services.pron_evaluator import (
    PhoneSegment,
    PronEvaluator,
    _approx_frame_energy_and_zcr,
    _enforce_min_frames,
    _pause_stats,
    _trim_leading_trailing,
    _vad_mask_from_energy,
    articulatory_distance_weighted_lists,
    bootstrap_per_ci,
    build_arpabet_to_model_vocab_map,
    build_ctc_targets,
    compute_fluency,
    confusion_list,
    ctc_collapse,
    ctc_force_align,
    expand_diphthongs_if_missing,
    gop_lr_excluding_self,
    is_vowel,
    levenshtein_backtrace,
    levenshtein_ops,
    load_audio_16k_mono,
    map_ref_arpabet_to_model_symbols,
    normalize_phone_sequence,
    normalize_text,
    per_by_position_from_ops,
    stratified_per,
    summarize_ops,
    tokenize_words,
    wordwise_sdin,
)


# ========================= Text utilities =========================


def test_normalize_text_cases():
    assert normalize_text("  She   look ") == "She look"
    assert normalize_text("") == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("She, LOOK'd!", ["she", "look'd"]),
        ("   ", []),
        ("it's ok", ["it's", "ok"]),
    ],
)
def test_tokenize_words_cases(text, expected):
    assert tokenize_words(text) == expected


# ========================= Audio loader =========================


def test_load_audio_16k_mono_handles_mono(monkeypatch):
    sr = 16000
    data = np.linspace(-0.5, 0.5, sr, dtype=np.float32)
    monkeypatch.setattr(gop.sf, "read", lambda *args, **kwargs: (data.copy(), sr), raising=False)

    out = load_audio_16k_mono("dummy.wav")
    assert out.dtype == np.float32
    assert len(out) == sr
    np.testing.assert_allclose(out, data, atol=1e-6)


def test_load_audio_16k_mono_downmixes_stereo(monkeypatch):
    sr = 16000
    left = np.linspace(-1.0, 1.0, sr, dtype=np.float32)
    right = np.linspace(1.0, -1.0, sr, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    monkeypatch.setattr(gop.sf, "read", lambda *args, **kwargs: (stereo.copy(), sr), raising=False)

    out = load_audio_16k_mono("dummy.wav")
    assert out.dtype == np.float32
    assert len(out) == sr
    # média dos canais → vetor próximo de zero
    np.testing.assert_allclose(out, np.zeros_like(left), atol=1e-6)


# def test_load_audio_16k_mono_resamples(monkeypatch):
#     sr_in = 22050
#     n_samples = sr_in  # 1 segundo aproximado
#     t = np.arange(n_samples, dtype=np.float32) / sr_in
#     data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
#     monkeypatch.setattr(gop.sf, "read", lambda *args, **kwargs: (data.copy(), sr_in), raising=False)
#
#     out = load_audio_16k_mono("dummy.wav")
#     expected_len = int(round(len(data) * 16000 / sr_in))
#     assert out.dtype == np.float32
#     assert len(out) == expected_len


# ========================= Mapping helpers =========================


def test_strip_stress_cases():
    assert gop.strip_stress("AE1") == "AE"
    assert gop.strip_stress("T") == "T"


def test_build_arpabet_to_model_vocab_map_basic():
    model_vocab = {"æ": 10, "i": 11, "tʃ": 12, "sil": 0, "ɹ": 13}
    out = build_arpabet_to_model_vocab_map(model_vocab)
    assert out == {"AE": "æ", "IY": "i", "CH": "tʃ", "SIL": "sil", "R": "ɹ"}


def test_map_ref_arpabet_to_model_symbols_success():
    arpa2sym = {"AE": "æ", "F": "f", "T": "t", "ER": "ɝ"}
    ref = ["AE1", "F", "T", "ER0", "SIL"]
    mapped, missing = map_ref_arpabet_to_model_symbols(ref, arpa2sym)
    assert mapped == ["æ", "f", "t", "ɝ"]
    assert missing == []


def test_map_ref_arpabet_to_model_symbols_with_missing():
    arpa2sym = {"AE": "æ", "T": "t"}
    ref = ["AE0", "F", "T", "ER1"]
    mapped, missing = map_ref_arpabet_to_model_symbols(ref, arpa2sym)
    assert mapped == ["æ", "t"]
    assert missing == ["ER", "F"]


def test_expand_diphthongs_if_missing_cases():
    vocab1 = {"k": 0, "e": 1, "ɪ": 2, "t": 3}
    assert expand_diphthongs_if_missing(["k", "eɪ", "t"], vocab1) == ["k", "e", "ɪ", "t"]

    vocab2 = {"k": 0, "eɪ": 1, "t": 2}
    assert expand_diphthongs_if_missing(["k", "eɪ", "t"], vocab2) == ["k", "eɪ", "t"]

    vocab3 = {"ɪ": 1}
    assert expand_diphthongs_if_missing(["eɪ"], vocab3) == ["eɪ"]


def test_normalize_phone_sequence_equivalents():
    seq = ["ʤ", "r", "ɚ", "ɡ"]
    assert normalize_phone_sequence(seq) == ["dʒ", "ɹ", "ɝ", "g"]


# ========================= Levenshtein & sequence alignment =========================


@pytest.mark.parametrize(
    "ref, hyp, expected",
    [
        (["a", "b", "c"], ["a", "x", "c", "d"], (1, 0, 1, 3)),
        (["p", "t"], ["p"], (0, 1, 0, 2)),
        ([], ["k", "a"], (0, 0, 2, 0)),
    ],
)
def test_levenshtein_ops_cases(ref, hyp, expected):
    assert levenshtein_ops(ref, hyp) == expected


def test_levenshtein_backtrace_contains_deletion():
    ref = ["k", "e", "ɪ", "t"]
    hyp = ["k", "e", "t"]
    ops = levenshtein_backtrace(ref, hyp)
    assert ("del", 2, None) in ops
    matches = [op for op in ops if op[0] == "match"]
    assert len(matches) >= 3


def test_confusion_and_summarize_ops():
    ref = ["k", "e", "ɪ", "t"]
    hyp = ["k", "e", "t"]
    conf = confusion_list(ref, hyp)
    assert {"op": "del", "ref": "ɪ", "hyp": None} in conf

    per_ops = summarize_ops(ref, hyp)
    assert ["match", "match", "del(ɪ)", "match"] == per_ops


# ========================= CTC helpers =========================


def test_ctc_collapse_basic():
    ids = [0, 1, 1, 0, 2, 2, 0]
    assert ctc_collapse(ids, blank_id=0) == [1, 2]


def test_build_ctc_targets_basic():
    assert build_ctc_targets([12, 7, 33], blank_id=0) == [0, 12, 0, 7, 0, 33, 0]


def test_ctc_force_align_minimal_case():
    blank = 0
    ref_ids = [1, 2]
    probs = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )
    log_post = np.log(probs)

    segs = ctc_force_align(log_post, ref_ids, blank)
    assert len(segs) == 2

    seg1, seg2 = segs
    assert seg1.t1 <= seg1.t2 <= 4
    assert seg2.t1 <= seg2.t2 <= 4
    assert seg1.start_s == pytest.approx(-1.0)
    assert seg2.end_s == pytest.approx(-1.0)


# ========================= GOP helpers =========================


@pytest.mark.parametrize(
    "args, expected",
    [
        ((10, 10, 100, 3), (9, 11)),
        ((0, 0, 2, 4), (0, 1)),
        ((-1, -1, 100, 3), (-1, -1)),
    ],
)
def test_enforce_min_frames_cases(args, expected):
    assert _enforce_min_frames(*args) == expected


def test_gop_lr_excluding_self_excludes_target():
    seg_log_post = np.array(
        [
            [-0.1, -2.0, -3.0, -4.0],
            [-0.2, -1.5, -3.5, -4.0],
            [-0.2, -2.0, -3.0, -4.0],
        ],
        dtype=np.float32,
    )
    li = 0
    rival_mask = np.array([False, True, True, True])

    gop_val = gop_lr_excluding_self(seg_log_post, li, rival_mask)
    assert gop_val > 0.0

    # Sem excluir o alvo: GOP cai
    rival_mask_including = rival_mask.copy()
    rival_mask_including[li] = True
    rival = np.max(seg_log_post[:, rival_mask_including], axis=1)
    no_exclusion = float(np.mean(seg_log_post[:, li] - rival))
    assert gop_val > no_exclusion


# ========================= AER helpers =========================


@pytest.mark.parametrize(
    "phone, expected",
    [
        ("i", True),
        ("t", False),
        ("dʒ", False),
        ("ɝ", True),
        ("xyz", None),
    ],
)
def test_is_vowel_cases(phone, expected):
    assert is_vowel(phone) == expected


def test_articulatory_distance_weighted_lists_mismatch():
    mism, total, per_feat, legacy = articulatory_distance_weighted_lists(["t", "ʃ"], ["t", "s"])
    assert total > 0.0
    assert 0.0 <= mism <= total
    assert any(per_feat[k][0] > 0.0 for k in ("man", "pla") if k in per_feat)
    assert legacy == (0.0, 0.0)


def test_articulatory_distance_weighted_lists_deletion():
    mism, total, per_feat, _ = articulatory_distance_weighted_lists(["t", "s"], ["t"])
    assert total > 0.0
    assert mism >= per_feat.get("pla", (0.0, 0.0))[0]


# ========================= Fluency helpers =========================


def test_approx_frame_energy_and_zcr_basic():
    audio = np.concatenate([np.zeros(4000, dtype=np.float32), np.full(4000, 0.1, dtype=np.float32)])
    am, zcr = _approx_frame_energy_and_zcr(audio, T=8)

    assert am.shape == (8,)
    assert zcr.shape == (8,)
    assert np.all(np.isfinite(am)) and np.all(np.isfinite(zcr))
    assert am[:4].mean() < am[4:].mean()
    assert np.allclose(zcr[:4], 0.0, atol=1e-6)
    assert np.allclose(zcr[4:], 0.0, atol=1e-6)


def test_vad_mask_from_energy_balanced():
    am = np.array([0.1] * 5 + [1.0] * 5, dtype=np.float32)
    zcr = np.zeros_like(am)  # impede que o ramo de ZCR ligue tudo
    mask = _vad_mask_from_energy(am, zcr)
    # 60º percentil de am será 1.0 => fala nos 5 últimos frames
    assert mask.dtype == bool
    assert mask.sum() == 5


def test_trim_leading_trailing_basic():
    mask = np.array([False, False, True, True, False, True, True, False], dtype=bool)
    trimmed, i1, i2 = _trim_leading_trailing(mask)
    expected = np.array([False, False, True, True, False, True, True, False], dtype=bool)
    assert trimmed.tolist() == expected.tolist()
    assert (i1, i2) == (2, 6)


def test_pause_stats_expected_values():
    is_pause = np.array([False] * 20 + [True] * 8 + [False] * 40 + [True] * 7 + [False] * 25, dtype=bool)
    stats = _pause_stats(is_pause, sr_frames_hz=25.0, words_out=[{"syllables": 12}], word_spans=[])
    assert stats["num_pauses_250ms"] == 2
    assert stats["mean_pause_ms"] == pytest.approx(300.0, abs=10.0)
    assert stats["mlr_syllables"] > 0.0
    assert stats["std_inter_pause_ms"] >= 0.0


def test_compute_fluency_minimal_scenario():
    audio = np.concatenate([np.zeros(4000, dtype=np.float32), np.full(4000, 0.1, dtype=np.float32)])
    T = 50
    V = 5
    log_post = np.zeros((T, V), dtype=np.float32)
    tok2id = {"sil": 0}
    words_out = [{"syllables": 2, "target_word": "test"}]
    ref_phones = ["t", "ɛ", "s", "t"]
    argmax = [0] * 25 + [1] * 25
    blank_id = 0

    basic, details = compute_fluency(audio, log_post, tok2id, words_out, ref_phones, argmax, blank_id)

    assert 0.4 <= basic["pause_ratio"] <= 0.6
    assert basic["duration_s"] == pytest.approx(0.5, abs=0.05)
    assert basic["words_per_sec"] >= 0.0
    assert details["articulation_rate_syll_per_sec"] > 0.0
    assert details["wpm"] >= 0.0


# ========================= PER metrics =========================


def test_stratified_per_counts_by_class():
    ref = ["i", "t", "s"]
    hyp = ["i", "s"]
    per_cls = stratified_per(ref, hyp)
    assert per_cls["cons"]["per"] > 0.0
    assert per_cls["vowel"]["per"] == pytest.approx(0.0)
    for bucket in per_cls.values():
        assert 0.0 <= bucket["per"] <= 1.0


def test_per_by_position_from_ops_medial_deletion():
    ref = ["t", "i", "s", "t"]
    hyp = ["t", "i", "t"]
    ops = levenshtein_backtrace(ref, hyp)
    spans = [(0, 3)]
    result = per_by_position_from_ops(ref, ops, spans)
    assert result["medial"]["D"] == 1
    assert result["initial"]["D"] == 0
    assert result["final"]["D"] == 0


def test_wordwise_sdin_basic():
    ref_words = [["t", "ɛ", "s", "t"], ["k", "æ", "t"]]
    hyp_words = [["t", "ɛ", "s", "t"], ["k", "e", "t"]]
    out = wordwise_sdin(ref_words, hyp_words)
    assert out == [(0, 0, 0, 4), (1, 0, 0, 3)]


def test_bootstrap_per_ci_expected_limits():
    data = [(0, 0, 0, 4), (1, 0, 0, 3)]
    ci = bootstrap_per_ci(data, n_boot=1000, seed=1337, alpha=0.05)
    per_true = 1 / 7
    assert ci["level"] == pytest.approx(0.95)
    assert ci["n_boot"] == 1000
    assert ci["low"] <= per_true <= ci["high"]


# ========================= PronEvaluator helpers =========================


def _make_stub_evaluator():
    stub = PronEvaluator.__new__(PronEvaluator)
    stub.gop_rival_scope = "same_class"
    stub.global_gop_threshold = 0.0
    stub.gop_thresholds = {}
    stub.min_frames_per_phone = 2
    stub.blank_id = 0

    stub.id2tok = {0: "<pad>", 1: "sil", 2: "t", 3: "ɛ", 4: "s", 5: "a"}
    stub.tok2id = {tok: idx for idx, tok in stub.id2tok.items()}
    V = len(stub.id2tok)

    rival_mask_all = np.zeros(V, dtype=bool)
    rival_mask_all[[2, 3, 4, 5]] = True
    stub.rival_mask_all = rival_mask_all

    rival_mask_vowel = np.zeros(V, dtype=bool)
    rival_mask_vowel[[3, 5]] = True  # ɛ e a
    stub.rival_mask_vowel = rival_mask_vowel

    rival_mask_cons = np.zeros(V, dtype=bool)
    rival_mask_cons[[2, 4]] = True  # t, s
    stub.rival_mask_cons = rival_mask_cons

    stub.ignore_tokens = {"|", "<s>", "</s>", " ", "sil", "spn", "nsn"}
    stub.arpa2sym = {"T": "t", "EH": "ɛ", "S": "s"}
    stub.word_ok_threshold = 0.8
    stub.war_tolerant_alpha = 0.8
    stub.war_hybrid_alpha = 0.8
    stub.war_hybrid_beta = 0.0
    return stub


def test_rival_mask_for_phone_same_class():
    stub = _make_stub_evaluator()
    mask = stub._rival_mask_for_phone("ɛ")
    assert mask.dtype == bool
    assert mask[stub.tok2id["ɛ"]]
    assert mask.sum() >= 2  # inclui o alvo e ao menos outro rival da mesma classe

    seg_log_post = np.array(
        [
            [-0.2, -2.0, -3.0, -0.5, -4.0, -1.5],
            [-0.2, -2.0, -3.5, -0.4, -4.2, -1.6],
        ],
        dtype=np.float32,
    )
    li = stub.tok2id["ɛ"]
    gop_val = gop_lr_excluding_self(seg_log_post, li, mask)
    assert gop_val > 0.0

    # Comparação com versão que não exclui o alvo
    rival = np.max(seg_log_post[:, mask], axis=1)
    no_excl = float(np.mean(seg_log_post[:, li] - rival))
    assert gop_val > no_excl


# ========================= PronEvaluator.evaluate (stubbed) =========================


def _make_evaluator_with_stubs(monkeypatch, tmp_path):
    ev = _make_stub_evaluator()
    ev.processor = None
    ev.model = None

    audio = np.concatenate([np.zeros(4000, dtype=np.float32), np.full(4000, 0.1, dtype=np.float32)])
    wav_path = tmp_path / "stub.wav"

    log_post = np.log(
        np.array(
            [
                [0.55, 0.10, 0.20, 0.05, 0.05, 0.05],
                [0.05, 0.10, 0.70, 0.05, 0.05, 0.05],
                [0.05, 0.10, 0.70, 0.05, 0.05, 0.05],
                [0.05, 0.10, 0.05, 0.70, 0.05, 0.05],
                [0.05, 0.10, 0.05, 0.70, 0.05, 0.05],
                [0.05, 0.10, 0.05, 0.05, 0.70, 0.05],
                [0.05, 0.10, 0.05, 0.05, 0.70, 0.05],
                [0.05, 0.10, 0.70, 0.05, 0.05, 0.05],
            ],
            dtype=np.float32,
        )
    )
    pred_ids_full = [0, 2, 0, 3, 0, 4, 0, 2]

    monkeypatch.setattr(gop, "load_audio_16k_mono", lambda _: audio, raising=True)
    monkeypatch.setattr(
        gop.PronEvaluator,
        "_audio_to_logits",
        lambda self, _: (log_post, pred_ids_full),
        raising=True,
    )
    monkeypatch.setattr(gop, "g2p_arpabet_tokens", lambda word: ["T", "EH", "S", "T"], raising=True)

    segments = [
        PhoneSegment(ev.tok2id["t"], 0, 1, -1.0, -1.0),
        PhoneSegment(ev.tok2id["ɛ"], 2, 3, -1.0, -1.0),
        PhoneSegment(ev.tok2id["s"], 4, 5, -1.0, -1.0),
        PhoneSegment(ev.tok2id["t"], 6, 7, -1.0, -1.0),
    ]
    monkeypatch.setattr(gop, "ctc_force_align", lambda *args, **kwargs: segments, raising=True)

    return ev, str(wav_path)


def test_pron_evaluator_evaluate_stub(monkeypatch, tmp_path):
    ev, wav_path = _make_evaluator_with_stubs(monkeypatch, tmp_path)

    result = gop.PronEvaluator.evaluate(ev, wav_path, "test")
    sent = result["sentence_metrics"]

    assert sent["words"][0]["target_word"] == "test"
    assert 0.0 <= sent["intelligibility"] <= 1.0
    assert sent["war_variants"]["exact"] in (0.0, 1.0)

    word = sent["words"][0]
    for phone in word["phones"]:
        thr = ev._thresh_for(phone["phone"])
        assert phone["threshold"] == pytest.approx(thr)
        assert phone["pass"] == (phone["gop"] >= phone["threshold"])
        assert phone["start_s"] >= 0.0
        assert phone["end_s"] >= phone["start_s"]


# ========================= Golden regression =========================


# def test_pron_evaluator_golden_snapshot(monkeypatch, tmp_path):
#     ev, wav_path = _make_evaluator_with_stubs(monkeypatch, tmp_path)
#     result = gop.PronEvaluator.evaluate(ev, wav_path, "test")
#
#     gold_path = Path(__file__).with_name("data").joinpath("golden_small.json")
#     with gold_path.open("r", encoding="utf-8") as fh:
#         golden = json.load(fh)
#
#     sent = result["sentence_metrics"]
#     gold_sent = golden["sentence_metrics"]
#
#     assert sent["intelligibility"] == pytest.approx(gold_sent["intelligibility"], abs=1e-6)
#     assert sent["war_variants"]["exact"] == pytest.approx(gold_sent["war_variants"]["exact"], abs=1e-6)
#     assert sent["fluency"]["pause_ratio"] == pytest.approx(gold_sent["fluency"]["pause_ratio"], abs=1e-3)
#
#     word = sent["words"][0]
#     gold_word = gold_sent["words"][0]
#     assert word["word_score_per"] == pytest.approx(gold_word["word_score_per"], abs=1e-6)
#     assert word["word_score_gop"] == pytest.approx(gold_word["word_score_gop"], abs=1e-6)
#     assert word["phones"][0]["pass"] == gold_word["phones"][0]["pass"]
