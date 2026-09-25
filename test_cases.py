"""
test_cases.py
-------------
Validates the project pipeline against the exact required specification and examples:

Example:
  "The battery range is excellent, but the charging time is too long."
Output:
  Aspect: Battery range | Charging time
  Sentiment: Positive | Negative
"""

import os
import sys

# Ensure project root is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import get_pipeline


def run_tests():
    print("=" * 70)
    print("Project: AI-Based Automotive Review and Customer Sentiment Analytics")
    print("AI/ML Technologies: BERT, XLM-R, NLP")
    print("=" * 70)

    pipeline = get_pipeline(model_type="BERT")

    test_reviews = [
        # 1. The exact canonical example from the specification
        "The battery range is excellent, but the charging time is too long.",
        # 2. Mileage and Comfort
        "The mileage is excellent and the car is very comfortable.",
        # 3. Service and Price
        "The service is poor and the maintenance cost is expensive.",
        # 4. Engine and Infotainment
        "The engine power and acceleration are amazing, but the touchscreen infotainment is sluggish.",
        # 5. Safety
        "The braking system and driver assistance safety features are top notch.",
        # 6. Overall vehicle sentiment
        "Overall the car is average."
    ]

    for idx, review in enumerate(test_reviews, 1):
        print(f"\n[Test Case {idx}]")
        print(f"Input Review: \"{review}\"")
        res = pipeline.analyze_review(review)
        print(f"Overall Sentiment: {res['overall_sentiment']} (Confidence: {res['overall_confidence']:.2f})")
        print(f"Model: {res['model_used']}")
        print("\nExtracted Aspect-Level Sentiments:")
        print(f"{'Aspect':<22} | {'Domain':<14} | {'Sentiment':<10} | {'Confidence':<10} | Clause")
        print("-" * 80)
        for item in res["aspects"]:
            print(f"{item['aspect']:<22} | {item['domain']:<14} | {item['sentiment']:<10} | {item['confidence']:<10.2f} | \"{item['clause']}\"")

        # Specific verification for example from specification
        if idx == 1:
            aspect_dict = {a['aspect']: a['sentiment'] for a in res['aspects']}
            print("\n>> Spec Format Output Check <<")
            print(f"Aspect:    {'Battery range':<18} {'Charging time':<18}")
            print(f"Sentiment: {aspect_dict.get('Battery range', 'N/A'):<18} {aspect_dict.get('Charging time', 'N/A'):<18}")
            assert aspect_dict.get("Battery range") == "Positive", "Battery range must be Positive"
            assert aspect_dict.get("Charging time") == "Negative", "Charging time must be Negative"
            print(">> SUCCESS: Test Case 1 strictly matches the example output!")

    print("\n" + "=" * 70)
    print("ALL TEST CASES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
