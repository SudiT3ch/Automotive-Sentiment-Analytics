"""
AI-Based Automotive Review and Customer Sentiment Analytics
Package source code.
"""

from .pipeline import AutomotiveSentimentPipeline, get_pipeline
from .preprocessing import clean_text, segment_review_into_clauses
from .tokenizer import AutomotiveTokenizer
from .sentiment_classifier import AutomotiveSentimentClassifier
from .aspect_extraction import extract_aspect_terms, AUTOMOTIVE_ASPECTS
