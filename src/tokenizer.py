"""
src/tokenizer.py
----------------
Stage 2: Tokenizer for BERT and XLM-R models.

Loads and manages tokenization for:
1. BERT (WordPiece tokenization via bert-base-uncased)
2. XLM-R (SentencePiece BPE tokenization via xlm-roberta-base)
"""

import os
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer


class AutomotiveTokenizer:
    """
    Unified tokenizer wrapper supporting both BERT and XLM-R.
    Loads locally saved tokenizer assets for fast, offline execution.
    """
    def __init__(self, model_type: str = "BERT", model_path: Optional[str] = None):
        self.model_type = model_type.upper()
        self.tokenizer = None
        self._load_tokenizer(model_path)

    def _load_tokenizer(self, custom_path: Optional[str]):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        if self.model_type == "BERT":
            path = custom_path or os.path.join(project_root, "models", "bert_sentiment")
            if not os.path.exists(path):
                path = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        elif self.model_type in ["XLM-R", "XLMR", "XLM_ROBERTA"]:
            path = custom_path or os.path.join(project_root, "models", "xlmr_tokenizer")
            if not os.path.exists(path):
                path = "xlm-roberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose 'BERT' or 'XLM-R'.")

    def encode(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Tokenizes text and returns PyTorch tensor encodings with padding & truncation.
        """
        if not text:
            text = ""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"]
        }

    def decode(self, token_ids) -> str:
        """Decodes token IDs back to human-readable string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
