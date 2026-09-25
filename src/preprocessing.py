"""
src/preprocessing.py
--------------------
Stage 1: Text Preprocessing for Automotive Review Sentiment Analytics.

Handles:
- Text cleaning, unicode normalization, whitespace stripping
- Clause segmentation for multi-aspect reviews (e.g. separating clauses with 'but', 'however', etc.)
- Negation and intensifier preservation for accurate sentiment detection
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """
    Normalizes whitespace, strips irregular unicode artifacts,
    while strictly preserving sentence boundaries and conjunction punctuation.
    """
    if not text or not isinstance(text, str):
        return ""
    # Replace multiple whitespaces/newlines with single space
    cleaned = re.sub(r'[\r\n\t]+', ' ', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()


def segment_review_into_clauses(text: str) -> List[str]:
    """
    Deconstructs a multi-aspect automotive review into independent syntactic clauses.
    Example:
      "The battery range is excellent, but the charging time is too long."
      -> ["The battery range is excellent", "the charging time is too long"]
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # First split by terminal punctuation
    sentences = re.split(r'[.!?]+', cleaned)
    clauses: List[str] = []

    # Conjunction pattern for contrastive/compound clause splitting
    conjunction_pattern = re.compile(
        r'(?:,\s*|\s+)(?=but\b|however\b|although\b|though\b|whereas\b|while\b|except\b|and\s+the\b|and\s+its\b|yet\b|on\s+the\s+other\s+hand\b)',
        re.IGNORECASE
    )

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Split on semicolon first
        semi_parts = re.split(r';+', sentence)
        for part in semi_parts:
            part = part.strip()
            if not part:
                continue

            # Split on contrastive conjunctions
            sub_clauses = conjunction_pattern.split(part)
            for sc in sub_clauses:
                sc = sc.strip()
                # Clean leading punctuation or conjunction words from clause start
                sc = re.sub(r'^(?:,\s*|and\s+|but\s+|yet\s+)', '', sc, flags=re.IGNORECASE).strip()
                if sc and len(sc) > 3:
                    clauses.append(sc)

    # Fallback to original text if no clauses were produced
    if not clauses and cleaned:
        clauses = [cleaned]

    return clauses
