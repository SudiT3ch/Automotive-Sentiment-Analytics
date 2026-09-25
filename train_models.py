"""
train_models.py
---------------
Training & Fine-Tuning Script for BERT, XLM-R, and ML Baseline on Automotive Reviews.

Features:
- Stratified 80/10/10 Train/Validation/Test splitting
- Class-weighted CrossEntropyLoss to resolve class imbalance (71% Positive)
- Supports training both BERT (bert-base-uncased) and XLM-R (xlm-roberta-base)
- Automatic CUDA GPU detection with seamless CPU fallback
- Saves model checkpoints and evaluation reports
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def prepare_data(data_path="data/car_reviews_.csv"):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["review_text", "rating"]).drop_duplicates(subset=["review_text"])
    
    # Map 1-5 ratings to 3 sentiment classes
    def rating_to_sentiment(r):
        if r <= 2:
            return 0  # Negative
        elif r == 3:
            return 1  # Neutral
        else:
            return 2  # Positive

    df["sentiment_label"] = df["rating"].apply(rating_to_sentiment)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["sentiment_label"]
    )
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, random_state=42, stratify=test_df["sentiment_label"]
    )
    return train_df, val_df, test_df


def train_transformer(model_name="bert-base-uncased", model_type="BERT", epochs=3, batch_size=16, lr=2e-5):
    print(f"\n[Training] Starting Fine-Tuning for {model_type} ({model_name})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    train_df, val_df, test_df = prepare_data()
    print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Class weights for CrossEntropyLoss
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["sentiment_label"]),
        y=train_df["sentiment_label"].values
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"[Weights] Class weights: {class_weights}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    train_dataset = ReviewDataset(train_df["review_text"], train_df["sentiment_label"], tokenizer)
    val_dataset = ReviewDataset(val_df["review_text"], val_df["sentiment_label"], tokenizer)
    test_dataset = ReviewDataset(test_df["review_text"], test_df["sentiment_label"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(batch["label"].numpy())

        val_acc = accuracy_score(val_trues, val_preds)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Evaluate on held-out test set
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_trues.extend(batch["label"].numpy())

    test_acc = accuracy_score(test_trues, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_trues, test_preds, average="macro")
    print(f"\n[Test Results for {model_type}]")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n" + classification_report(test_trues, test_preds, target_names=["Negative", "Neutral", "Positive"]))

    # Save model and tokenizer
    out_dir = f"models/{model_type.lower()}_sentiment"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[Saved] Model and tokenizer saved to {out_dir}/")


def train_baseline():
    print("\n[Baseline] Training TF-IDF + Logistic Regression baseline...")
    train_df, val_df, test_df = prepare_data()
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train = vec.fit_transform(train_df["review_text"])
    X_test = vec.transform(test_df["review_text"])

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, train_df["sentiment_label"])

    preds = clf.predict(X_test)
    acc = accuracy_score(test_df["sentiment_label"], preds)
    print(f"[Baseline] Test Accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/baseline_model.joblib")
    joblib.dump(vec, "models/tfidf_vectorizer.joblib")
    print("[Baseline] Saved models/baseline_model.joblib and models/tfidf_vectorizer.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["bert", "xlmr", "baseline", "all"])
    args = parser.parse_args()

    if args.model in ["baseline", "all"]:
        train_baseline()
    if args.model in ["bert", "all"]:
        train_transformer("bert-base-uncased", "BERT", epochs=2)
    if args.model in ["xlmr", "all"]:
        train_transformer("xlm-roberta-base", "XLM-R", epochs=2)
