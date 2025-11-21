
import pytest
import torch
from unittest.mock import MagicMock, patch
from app.services.alignment_service import ForcedAlignmentService, _Point, _Segment, AlignmentFailedError
from app.models.word_segment import WordSegment

@pytest.fixture
def alignment_service():
    # Initialize without loading model to speed up tests
    service = ForcedAlignmentService()
    # Mock model and processor to avoid actual loading in _ensure_model
    service.model = MagicMock()
    service.processor = MagicMock()
    service._bundle_sr = 16000
    service.device = torch.device("cpu")
    return service

def test_normalize_transcript(alignment_service):
    text = "Hello World!"
    normalized = alignment_service._normalize_transcript(text)
    assert normalized == "|HELLO|WORLD|"

    text_with_special_chars = "It's a   test."
    normalized = alignment_service._normalize_transcript(text_with_special_chars)
    assert normalized == "|IT'S|A|TEST|"

def test_emission_mock(alignment_service):
    # Mock waveform
    waveform = torch.randn(1, 16000) # 1 second of audio
    
    # Mock processor output
    mock_inputs = MagicMock()
    mock_inputs.input_values = torch.randn(1, 16000)
    alignment_service.processor.return_value = mock_inputs
    
    # Mock model output
    mock_logits = torch.randn(1, 50, 32) # Batch=1, Time=50, Classes=32
    alignment_service.model.return_value.logits = mock_logits
    
    emissions = alignment_service._emission(waveform)
    
    assert emissions.shape == (50, 32)
    # Check if log_softmax was applied (values should be negative)
    assert (emissions <= 0).all()

def test_text_to_tokens(alignment_service):
    # Mock tokenizer vocab
    mock_vocab = {"|": 0, "A": 1, "B": 2}
    alignment_service.processor.tokenizer.get_vocab.return_value = mock_vocab
    
    tokens = alignment_service._text_to_tokens("|AB|")
    assert tokens == [0, 1, 2, 0]

def test_trellis_simple():
    # Simple emission: 3 frames, 2 classes (0=blank, 1=token)
    # We want to match token 1.
    # Frame 0: high prob for blank
    # Frame 1: high prob for token 1
    # Frame 2: high prob for blank
    
    # Log probabilities
    emission = torch.tensor([
        [-0.1, -10.0], # Frame 0: Blank
        [-10.0, -0.1], # Frame 1: Token
        [-0.1, -10.0]  # Frame 2: Blank
    ])
    
    tokens = [0, 1, 0] # | Token | (assuming 0 is blank/separator)
    
    trellis = ForcedAlignmentService._trellis(emission, tokens, blank_id=0)
    
    # Check dimensions: Frames x Tokens
    assert trellis.shape == (3, 3)
    
    # The last cell should have a high probability (close to 0 in log scale, or at least > -inf)
    assert trellis[-1, -1] > -10.0

def test_backtrack_simple():
    # Construct a fake trellis and emission for a clear path
    # Path: (0,0) -> (1,1) -> (2,2)
    # Tokens: [0, 1, 0]
    
    emission = torch.zeros((3, 2)) # Dummy emission
    tokens = [0, 1, 0]
    
    # Manually construct a trellis that forces a specific path
    # This is hard to mock perfectly without reimplementing the logic, 
    # so we'll test the structure of the output given a valid trellis/emission from the previous test.
    
    emission = torch.tensor([
        [0.0, -10.0], 
        [-10.0, 0.0], 
        [0.0, -10.0]
    ])
    trellis = ForcedAlignmentService._trellis(emission, tokens, blank_id=0)
    
    path = ForcedAlignmentService._backtrack(trellis, emission, tokens, blank_id=0)
    
    assert len(path) == 3
    # Expected path indices (token_index, time_index)
    # We expect it to hit token 0 at t=0, token 1 at t=1, token 2 at t=2
    assert path[0].token_index == 0 and path[0].time_index == 0
    assert path[1].token_index == 1 and path[1].time_index == 1
    assert path[2].token_index == 2 and path[2].time_index == 2

def test_merge_repeats():
    # Path with repeats: t=0(tok=0), t=1(tok=0), t=2(tok=1)
    path = [
        _Point(token_index=0, time_index=0, score=1.0),
        _Point(token_index=0, time_index=1, score=1.0),
        _Point(token_index=1, time_index=2, score=1.0),
    ]
    transcript = "|A" # 0=|, 1=A
    
    segs = ForcedAlignmentService._merge_repeats(path, transcript)
    
    assert len(segs) == 2
    assert segs[0].label == "|"
    assert segs[0].start == 0
    assert segs[0].end == 2 # Exclusive end
    
    assert segs[1].label == "A"
    assert segs[1].start == 2
    assert segs[1].end == 3

def test_merge_words():
    # Segments: "|", "H", "E", "L", "L", "O", "|"
    # We expect one word "HELLO"
    segs = [
        _Segment("|", 0, 1, 1.0),
        _Segment("H", 1, 2, 1.0),
        _Segment("E", 2, 3, 1.0),
        _Segment("L", 3, 4, 1.0),
        _Segment("L", 4, 5, 1.0),
        _Segment("O", 5, 6, 1.0),
        _Segment("|", 6, 7, 1.0),
    ]
    
    words = ForcedAlignmentService._merge_words(segs, sep="|")
    
    assert len(words) == 1
    assert words[0].label == "HELLO"
    assert words[0].start == 1
    assert words[0].end == 6

@patch("torchaudio.load")
@patch("torchaudio.functional.resample")
def test_align_integration(mock_resample, mock_load, alignment_service):
    # Setup mocks
    mock_waveform = torch.randn(1, 32000) # 2 seconds
    mock_load.return_value = (mock_waveform, 16000)
    
    # Mock emission to return a predictable sequence
    # Let's say we have 10 frames.
    # Transcript: "HI" -> "|HI|" -> tokens [0, 1, 2, 0]
    # We need emission [10, V]
    
    # Mock vocab
    alignment_service.processor.tokenizer.get_vocab.return_value = {"|": 0, "H": 1, "I": 2}
    
    # Mock emission output directly to bypass model forward pass logic in this integration test
    # or we can mock _emission method. Mocking _emission is safer for unit testing 'align' logic.
    with patch.object(alignment_service, '_emission') as mock_emission_method:
        # Create emission that favors the path 0, 1, 2, 0
        # 10 frames. 
        # 0-2: |
        # 3-5: H
        # 6-7: I
        # 8-9: |
        T = 10
        C = 3 # vocab size
        emission = torch.full((T, C), -10.0)
        
        # Set high probs
        emission[0:3, 0] = 0.0
        emission[3:6, 1] = 0.0
        emission[6:8, 2] = 0.0
        emission[8:10, 0] = 0.0
        
        mock_emission_method.return_value = emission
        
        words = alignment_service.align("dummy.wav", "Hi")
        
        assert len(words) == 1
        assert words[0].word == "HI"
        # Check timings
        # Ratio calculation: (32000 samples / 16000 sr) / 10 frames = 2.0 / 10 = 0.2 sec/frame
        # Word "HI" corresponds to segments H and I.
        # H: frames 3-6 (start=3, end=6) -> 0.6s to 1.2s
        # I: frames 6-8 (start=6, end=8) -> 1.2s to 1.6s
        # Merged HI: start=3, end=8 -> 0.6s to 1.6s
        
        assert words[0].start == pytest.approx(0.6)
        assert words[0].end == pytest.approx(1.6)

def test_align_short_audio(alignment_service):
    # Test case where audio is too short for the transcript
    with patch("torchaudio.load") as mock_load:
        mock_load.return_value = (torch.randn(1, 100), 16000) # Very short audio
        
        with patch.object(alignment_service, '_emission') as mock_emission:
            # Emission very short
            mock_emission.return_value = torch.randn(2, 10) 
            
            # Transcript long
            transcript = "Long transcript"
            
            # Mock vocab to return dummy tokens
            alignment_service.processor.tokenizer.get_vocab.return_value = {c: 1 for c in "|LONGTRANSCRIPT"}
            
            words = alignment_service.align("dummy.wav", transcript)
            
            # Should return empty list and log warning (not checking log here, but result)
            assert words == []
