"""
run_pipeline.py
---------------
CLI tool to test any automotive review through the 7-stage NLP pipeline:
Automotive Reviews -> Text Preprocessing -> Tokenizer -> BERT / XLM-R -> Sentiment Classification -> Aspect Extraction -> Aspect-Level Sentiment
"""

import sys
import argparse
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import get_pipeline


def main():
    parser = argparse.ArgumentParser(description="Analyze automotive review sentiment & aspects.")
    parser.add_argument(
        "--review",
        type=str,
        default="The battery range is excellent, but the charging time is too long.",
        help="Input vehicle review text"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BERT",
        choices=["BERT", "XLM-R"],
        help="Transformer architecture to use (BERT or XLM-R)"
    )
    args = parser.parse_args()

    pipeline = get_pipeline(model_type=args.model)
    result = pipeline.analyze_review(args.review)

    print("=" * 75)
    print("AI-BASED AUTOMOTIVE REVIEW & CUSTOMER SENTIMENT ANALYTICS")
    print(f"Architecture: Reviews -> Preprocessing -> Tokenizer -> {args.model} -> Sentiment -> Aspect Extraction -> Aspect Sentiment")
    print("=" * 75)
    print(f"Review: \"{result['input_review']}\"")
    print(f"Overall Sentiment: {result['overall_sentiment']} (Confidence: {result['overall_confidence']:.2f})")
    print(f"Model: {result['model_used']}")
    print("-" * 75)
    print(f"{'Aspect':<24} | {'Domain':<14} | {'Sentiment':<10} | {'Confidence':<10} | Clause")
    print("-" * 75)
    for a in result["aspects"]:
        print(f"{a['aspect']:<24} | {a['domain']:<14} | {a['sentiment']:<10} | {a['confidence']:<10.2f} | \"{a['clause']}\"")
    print("=" * 75)

    # Clean horizontal summary table matching spec
    if result["aspects"]:
        aspect_names = [a["aspect"] for a in result["aspects"]]
        sentiments = [a["sentiment"] for a in result["aspects"]]
        print("\nSummary (Specification Format):")
        hdr_aspect = "Aspect"
        hdr_sentiment = "Sentiment"
        print(f"{hdr_aspect:<16}" + "".join(f"{name:<22}" for name in aspect_names))
        print(f"{hdr_sentiment:<16}" + "".join(f"{s:<22}" for s in sentiments))
        print()


if __name__ == "__main__":
    main()
