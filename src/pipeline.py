"""
src/pipeline.py
---------------
Unified 7-Stage End-to-End Orchestrator Pipeline:
1. Automotive Reviews
2. Text Preprocessing
3. Tokenizer
4. BERT / XLM-R
5. Sentiment Classification
6. Aspect Extraction
7. Aspect-Level Sentiment
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from .preprocessing import clean_text, segment_review_into_clauses
from .sentiment_classifier import AutomotiveSentimentClassifier
from .aspect_extraction import (
    extract_aspect_terms,
    get_clause_lexicon_polarity,
    AUTOMOTIVE_ASPECTS
)


class AutomotiveSentimentPipeline:
    """
    End-to-End Pipeline implementing the exact specified architecture.
    """
    def __init__(self, model_type: str = "BERT"):
        self.model_type = model_type.upper()
        self.classifier = AutomotiveSentimentClassifier(model_type=self.model_type)

    def set_model(self, model_type: str):
        """Allows switching between BERT and XLM-R dynamically."""
        if model_type.upper() != self.model_type:
            self.model_type = model_type.upper()
            self.classifier = AutomotiveSentimentClassifier(model_type=self.model_type)

    def analyze_review(self, review_text: str) -> Dict[str, Any]:
        """
        Executes the full 7-stage NLP pipeline on an automotive review.
        """
        # Step 1: Text Preprocessing
        cleaned_review = clean_text(review_text)
        if not cleaned_review:
            return {
                "input_review": review_text,
                "overall_sentiment": "Neutral",
                "overall_confidence": 0.5,
                "aspects": [],
                "model_used": self.model_type
            }

        # Step 2, 3, 4: Overall Sentiment Classification via BERT / XLM-R
        overall_pred = self.classifier.predict_sentiment(cleaned_review)
        overall_sentiment = overall_pred["label"]
        overall_confidence = overall_pred["confidence"]

        # Step 5: Syntactic Clause Segmentation
        clauses = segment_review_into_clauses(cleaned_review)

        # Step 6 & 7: Aspect Extraction & Aspect-Level Sentiment
        extracted_aspects: List[Dict[str, Any]] = []
        seen_aspect_terms = set()

        for clause in clauses:
            aspect_matches = extract_aspect_terms(clause)
            if not aspect_matches:
                continue

            # Evaluate clause sentiment using Transformer + Lexicon calibration
            clause_pred = self.classifier.predict_sentiment(clause)
            clause_sent = clause_pred["label"]
            clause_conf = clause_pred["confidence"]

            # Validate against polarity cues for contrastive clauses
            lex_sent, lex_conf = get_clause_lexicon_polarity(clause)
            if lex_sent is not None:
                final_sent = lex_sent
                final_conf = max(clause_conf, lex_conf)
            else:
                final_sent = clause_sent
                final_conf = clause_conf

            for match in aspect_matches:
                term = match["term"]
                domain = match["domain"]
                if term not in seen_aspect_terms:
                    seen_aspect_terms.add(term)
                    extracted_aspects.append({
                        "aspect": term,
                        "domain": domain,
                        "sentiment": final_sent,
                        "confidence": final_conf,
                        "clause": clause
                    })

        # If no specific aspect terms were captured but review exists, map to general Vehicle aspect
        if not extracted_aspects and cleaned_review:
            extracted_aspects.append({
                "aspect": "Vehicle",
                "domain": "Vehicle",
                "sentiment": overall_sentiment,
                "confidence": overall_confidence,
                "clause": cleaned_review
            })

        return {
            "input_review": review_text,
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "model_used": overall_pred["model_name"],
            "aspects": extracted_aspects
        }

    def format_aspect_table(self, analysis_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Formats extracted aspects into a clean tabular DataFrame:
        Aspect | Sentiment | Domain | Confidence | Clause
        """
        rows = []
        for item in analysis_result["aspects"]:
            rows.append({
                "Aspect": item["aspect"],
                "Sentiment": item["sentiment"],
                "Domain": item["domain"],
                "Confidence": f"{item['confidence'] * 100:.1f}%",
                "Clause": item["clause"]
            })
        return pd.DataFrame(rows)


# Singleton instance helper
_PIPELINE_INSTANCES: Dict[str, AutomotiveSentimentPipeline] = {}

def get_pipeline(model_type: str = "BERT") -> AutomotiveSentimentPipeline:
    m = model_type.upper()
    if m not in _PIPELINE_INSTANCES:
        _PIPELINE_INSTANCES[m] = AutomotiveSentimentPipeline(model_type=m)
    return _PIPELINE_INSTANCES[m]
