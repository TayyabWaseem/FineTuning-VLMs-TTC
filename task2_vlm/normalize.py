"""One deterministic normalizer for selection, targets, and evaluation.

The manual permits a consistent normalization rule; this implementation is
not the complete official VQA evaluator's contractions table. Record this
choice in run metadata and do not describe it as exact official scoring.
"""
from __future__ import annotations

import re
import unicodedata


NUMBERS = {word: str(number) for number, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
NUMBERS["none"] = "0"
ARTICLES = {"a", "an", "the"}
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldnt": "couldn't",
                "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                "hasnt": "hasn't", "havent": "haven't", "isnt": "isn't", "shouldnt": "shouldn't",
                "thats": "that's", "theres": "there's", "theyre": "they're", "wasnt": "wasn't",
                "werent": "weren't", "whats": "what's", "wont": "won't", "wouldnt": "wouldn't",
                "youre": "you're", "youve": "you've"}


def normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s'.:]", " ", text)
    words = [NUMBERS.get(word, word) for word in text.split()]
    return " ".join(CONTRACTIONS.get(word, word) for word in words if word not in ARTICLES)
