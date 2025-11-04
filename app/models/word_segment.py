from dataclasses import dataclass
from typing import List

@dataclass
class WordSegment:
    word: str
    start: float
    end: float
    score: float