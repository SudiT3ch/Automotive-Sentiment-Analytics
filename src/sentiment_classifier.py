"""
src/sentiment_classifier.py
---------------------------
Stages 3 & 4: BERT / XLM-R Transformer Model & Sentiment Classification.

Performs:
- Sequence classification into Negative (0), Neutral (1), Positive (2)
- Returns predicted sentiment label, probabilities, and confidence score
- Supports BERT (bert-base-uncased) and XLM-R (xlm-roberta-base)
- Graceful fallback to TF-IDF Baseline if transformer weights are not yet generated
"""

import os
import joblib
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForSequenceClassification

from .tokenizer import AutomotiveTokenizer
from .preprocessing import clean_text


ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}


class AutomotiveSentimentClassifier:
    """
    Unified Sentiment Classification Engine using BERT / XLM-R.
    """
    def __init__(
        self,
        model_type: str = "BERT",
        custom_model_path: Optional[str] = None
    ):
        self.model_type = model_type.upper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer_wrapper = None
        self.baseline_model = None
        self.baseline_vec = None
        self.is_transformer_loaded = False
        
        # Log prior shift to compensate for class imbalance
        # Inverse class weights: [Negative: 2.871, Neutral: 1.927, Positive: 0.469]
        self.log_prior_shift = torch.tensor([2.8707483, 1.92694064, 0.46888889]).log().to(self.device)

        self._initialize_model(custom_model_path)

    def _initialize_model(self, custom_path: Optional[str]):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # 1. Initialize Tokenizer
        try:
            self.tokenizer_wrapper = AutomotiveTokenizer(self.model_type)
        except Exception as e:
            print(f"[Warning] Tokenizer init for {self.model_type} failed: {e}")

        # 2. Load Transformer Model
        if self.model_type == "BERT":
            model_dir = custom_path or os.path.join(project_root, "models", "bert_sentiment")
            if os.path.exists(model_dir) and (os.path.exists(os.path.join(model_dir, "model.safetensors")) or os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir,
                        num_labels=3,
                        id2label=ID2LABEL,
                        label2id=LABEL2ID
                    ).to(self.device)
                    self.model.eval()
                    self.is_transformer_loaded = True
                    print(f"[Classifier] BERT loaded successfully from local checkpoint on {self.device}.")
                except Exception as e:
                    print(f"[Classifier] Error loading BERT local model: {e}")
            else:
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        "bert-base-uncased",
                        num_labels=3,
                        id2label=ID2LABEL,
                        label2id=LABEL2ID
                    ).to(self.device)
                    self.model.eval()
                    self.is_transformer_loaded = True
                    print(f"[Classifier] BERT loaded from HuggingFace Hub on {self.device}.")
                except Exception as e:
                    print(f"[Classifier] Error loading BERT from Hub: {e}")

        elif self.model_type in ["XLM-R", "XLMR"]:
            model_dir = custom_path or os.path.join(project_root, "models", "xlmr_sentiment")
            if os.path.exists(model_dir) and (os.path.exists(os.path.join(model_dir, "model.safetensors")) or os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir,
                        num_labels=3,
                        id2label=ID2LABEL,
                        label2id=LABEL2ID
                    ).to(self.device)
                    self.model.eval()
                    self.is_transformer_loaded = True
                    print(f"[Classifier] XLM-R loaded successfully from local checkpoint on {self.device}.")
                except Exception as e:
                    print(f"[Classifier] Error loading local XLM-R: {e}")
            else:
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        "xlm-roberta-base",
                        num_labels=3,
                        id2label=ID2LABEL,
                        label2id=LABEL2ID
                    ).to(self.device)
                    self.model.eval()
                    self.is_transformer_loaded = True
                    print(f"[Classifier] XLM-R loaded from HuggingFace Hub on {self.device}.")
                except Exception as e:
                    print(f"[Classifier] Error loading XLM-R from Hub: {e}")

        # 3. Load ML Baseline fallback if transformer is not available
        if not self.is_transformer_loaded:
            baseline_path = os.path.join(project_root, "models", "baseline_model.joblib")
            vec_path = os.path.join(project_root, "models", "tfidf_vectorizer.joblib")
            if os.path.exists(baseline_path) and os.path.exists(vec_path):
                try:
                    self.baseline_model = joblib.load(baseline_path)
                    self.baseline_vec = joblib.load(vec_path)
                    print(f"[Classifier] Baseline TF-IDF model loaded as standby.")
                except Exception as e:
                    print(f"[Classifier] Error loading baseline: {e}")

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predicts sentiment for a single review or clause.
        Returns:
            label: "Positive" | "Neutral" | "Negative"
            confidence: float (0.0 to 1.0)
            probabilities: Dict[str, float]
            model_name: str
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "label": "Neutral",
                "confidence": 0.5,
                "probabilities": {"Negative": 0.25, "Neutral": 0.50, "Positive": 0.25},
                "model_name": "Default"
            }

        # Use Transformer if loaded
        if self.is_transformer_loaded and self.model is not None and self.tokenizer_wrapper is not None:
            enc = self.tokenizer_wrapper.encode(cleaned, max_length=128)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(0)

                # Calibrate probabilities against class imbalance
                calibrated_logits = logits - self.log_prior_shift
                probs = F.softmax(calibrated_logits, dim=-1).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_label = ID2LABEL[pred_idx]
            confidence = float(probs[pred_idx])

            return {
                "label": pred_label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "Negative": round(float(probs[0]), 4),
                    "Neutral": round(float(probs[1]), 4),
                    "Positive": round(float(probs[2]), 4)
                },
                "model_name": f"{self.model_type} Transformer"
            }

        # Fallback to Baseline TF-IDF + Logistic Regression
        elif self.baseline_model is not None and self.baseline_vec is not None:
            features = self.baseline_vec.transform([cleaned])
            pred_idx = int(self.baseline_model.predict(features)[0])
            pred_label = ID2LABEL.get(pred_idx, "Neutral")
            
            probs = self.baseline_model.predict_proba(features)[0]
            confidence = float(np.max(probs))

            return {
                "label": pred_label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "Negative": round(float(probs[0]), 4),
                    "Neutral": round(float(probs[1]), 4),
                    "Positive": round(float(probs[2]), 4)
                },
                "model_name": "TF-IDF + Logistic Regression"
            }

        else:
            return {
                "label": "Neutral",
                "confidence": 0.5,
                "probabilities": {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
                "model_name": "Fallback Rule"
            }
