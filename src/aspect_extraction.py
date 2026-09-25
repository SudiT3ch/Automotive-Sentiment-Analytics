"""
src/aspect_extraction.py
------------------------
Stage 5: Aspect Extraction across 9 Core Automotive Dimensions:
1. Vehicle
2. Engine
3. Battery
4. Mileage
5. Safety
6. Comfort
7. Service
8. Infotainment
9. Price

Extracts specific aspect terms (e.g. 'Battery range', 'Charging time', 'Mileage', 'Comfort')
and associates them with parent domains and relevant clause segments.
"""

import re
from typing import List, Dict, Any, Tuple, Optional


# Master 9-Aspect Lexicon mapping domains to keyword phrases and prioritized display labels
AUTOMOTIVE_ASPECTS: Dict[str, Dict[str, Any]] = {
    "Battery": {
        "priority_terms": [
            ("battery range", "Battery range"),
            ("ev range", "Battery range"),
            ("charging time", "Charging time"),
            ("fast charging", "Charging time"),
            ("charging speed", "Charging time"),
            ("battery life", "Battery health"),
            ("battery health", "Battery health"),
            ("charging", "Charging time"),
            ("battery", "Battery"),
            ("charge", "Charging time"),
            ("range", "Battery range"),
            ("wallbox", "Charging time"),
            ("kwh", "Battery capacity")
        ]
    },
    "Mileage": {
        "priority_terms": [
            ("fuel economy", "Fuel economy"),
            ("fuel efficiency", "Fuel efficiency"),
            ("gas mileage", "Mileage"),
            ("fuel consumption", "Fuel economy"),
            ("mileage", "Mileage"),
            ("kmpl", "Mileage"),
            ("mpg", "Mileage"),
            ("efficiency", "Fuel efficiency"),
            ("consumption", "Fuel economy")
        ]
    },
    "Engine": {
        "priority_terms": [
            ("engine performance", "Engine"),
            ("horsepower", "Engine power"),
            ("torque", "Engine torque"),
            ("acceleration", "Acceleration"),
            ("transmission", "Transmission"),
            ("gearbox", "Transmission"),
            ("motor", "Engine"),
            ("power", "Engine power"),
            ("pickup", "Acceleration"),
            ("cylinder", "Engine"),
            ("turbo", "Engine turbo"),
            ("v6", "Engine"),
            ("v8", "Engine"),
            ("engine", "Engine")
        ]
    },
    "Comfort": {
        "priority_terms": [
            ("ride quality", "Ride quality"),
            ("cabin noise", "Cabin quietness"),
            ("seat comfort", "Seat comfort"),
            ("rear seat space", "Cabin space"),
            ("legroom", "Legroom"),
            ("headroom", "Headroom"),
            ("suspension", "Suspension"),
            ("comfortable", "Comfort"),
            ("comfort", "Comfort"),
            ("seats", "Seat comfort"),
            ("seating", "Seat comfort"),
            ("cabin", "Cabin comfort"),
            ("smooth", "Ride quality"),
            ("quiet", "Cabin quietness"),
            ("noise", "Cabin quietness"),
            ("spacious", "Cabin space"),
            ("ergonomic", "Comfort")
        ]
    },
    "Safety": {
        "priority_terms": [
            ("lane assist", "Driver assistance"),
            ("crash test", "Crash safety"),
            ("driver assistance", "Driver assistance"),
            ("blind spot", "Driver assistance"),
            ("airbag", "Airbag safety"),
            ("collision", "Collision safety"),
            ("safety", "Safety"),
            ("brake", "Braking system"),
            ("braking", "Braking system"),
            ("abs", "Braking system"),
            ("crash", "Crash safety"),
            ("emergency", "Emergency safety"),
            ("seatbelt", "Safety"),
            ("adas", "Driver assistance")
        ]
    },
    "Infotainment": {
        "priority_terms": [
            ("apple carplay", "CarPlay / Android Auto"),
            ("android auto", "CarPlay / Android Auto"),
            ("sound system", "Sound system"),
            ("touchscreen display", "Touchscreen"),
            ("dashboard display", "Infotainment display"),
            ("touchscreen", "Touchscreen"),
            ("infotainment", "Infotainment"),
            ("screen", "Touchscreen"),
            ("navigation", "Navigation system"),
            ("audio", "Audio system"),
            ("bluetooth", "Bluetooth connectivity"),
            ("speakers", "Sound system"),
            ("display", "Infotainment display"),
            ("connectivity", "Connectivity")
        ]
    },
    "Service": {
        "priority_terms": [
            ("customer service", "Customer service"),
            ("service center", "Service center"),
            ("after-sales", "After-sales service"),
            ("servicing", "Service"),
            ("maintenance", "Maintenance service"),
            ("dealership", "Dealership service"),
            ("repair", "Repair service"),
            ("dealer", "Dealership service"),
            ("warranty", "Warranty service"),
            ("mechanic", "Service repair"),
            ("service", "Service")
        ]
    },
    "Price": {
        "priority_terms": [
            ("value for money", "Value for money"),
            ("maintenance cost", "Maintenance cost"),
            ("sticker price", "Purchase price"),
            ("resale value", "Resale value"),
            ("price", "Price"),
            ("cost", "Cost / Maintenance"),
            ("expensive", "Price"),
            ("affordable", "Price affordability"),
            ("value", "Value for money"),
            ("money", "Value for money"),
            ("overpriced", "Price"),
            ("budget", "Budget"),
            ("cheap", "Price"),
            ("economical", "Economical value"),
            ("worth", "Value for money"),
            ("pricing", "Price")
        ]
    },
    "Vehicle": {
        "priority_terms": [
            ("build quality", "Build quality"),
            ("fit and finish", "Build quality"),
            ("exterior design", "Exterior styling"),
            ("road presence", "Road presence"),
            ("handling", "Handling & steering"),
            ("steering", "Handling & steering"),
            ("exterior", "Exterior styling"),
            ("styling", "Styling"),
            ("design", "Design"),
            ("look", "Appearance"),
            ("appearance", "Appearance"),
            ("chassis", "Chassis"),
            ("car", "Vehicle"),
            ("vehicle", "Vehicle"),
            ("model", "Vehicle")
        ]
    }
}

# Domain polarity cues for clause-level sentiment calibration
POSITIVE_CUES = {
    "excellent", "great", "good", "amazing", "incredible", "impressive",
    "class-leading", "reliable", "smooth", "love", "loved", "best", "top",
    "wonderful", "trouble-free", "comfortable", "perfect", "fantastic",
    "solid", "fast", "efficient", "exceeds", "enjoyable", "exhilarating"
}

NEGATIVE_CUES = {
    "poor", "disappointing", "disappointed", "bad", "terrible", "horrible",
    "worst", "unacceptable", "waste", "lemon", "issues", "issue", "problem",
    "expensive", "overpriced", "too long", "slow", "noisy", "uncomfortable",
    "broken", "breakdown", "failure", "fail", "defective", "rude", "costly",
    "lacking", "subpar", "sluggish"
}

NEUTRAL_CUES = {
    "average", "adequate", "acceptable", "fair", "okay", "ok", "decent",
    "ordinary", "moderate", "mixed", "reasonable"
}


def extract_aspect_terms(clause: str) -> List[Dict[str, str]]:
    """
    Extracts automotive aspect mentions from a single clause.
    Returns list of dicts: [{"domain": "Battery", "term": "Battery range", "keyword": "battery range"}]
    """
    clause_lower = clause.lower()
    matches = []
    seen_terms = set()

    for domain, config in AUTOMOTIVE_ASPECTS.items():
        for keyword, display_name in config["priority_terms"]:
            # Word boundary matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, clause_lower):
                if display_name not in seen_terms:
                    seen_terms.add(display_name)
                    matches.append({
                        "domain": domain,
                        "term": display_name,
                        "keyword": keyword
                    })
                break  # Pick highest priority term for this domain in this clause

    return matches


def get_clause_lexicon_polarity(clause: str) -> Tuple[Optional[str], float]:
    """
    Evaluates strong polarity cues in short phrases to support transformer inference.
    """
    clause_lower = clause.lower()
    tokens = set(re.findall(r'\b[a-z\-]+\b', clause_lower))
    
    # Check for "too long", "too noisy", etc.
    if "too long" in clause_lower or "too noisy" in clause_lower or "too expensive" in clause_lower:
        return "Negative", 0.75

    pos_hits = len(tokens.intersection(POSITIVE_CUES))
    neg_hits = len(tokens.intersection(NEGATIVE_CUES))
    neu_hits = len(tokens.intersection(NEUTRAL_CUES))

    # Check for negation: "not good", "not comfortable"
    has_negation = bool(re.search(r'\b(not|never|no|isnt|didn|wasnt)\b', clause_lower))
    if has_negation and pos_hits > 0:
        return "Negative", 0.75

    if pos_hits > neg_hits and pos_hits > neu_hits:
        return "Positive", 0.75
    elif neg_hits > pos_hits and neg_hits > neu_hits:
        return "Negative", 0.75
    elif neu_hits > 0:
        return "Neutral", 0.70

    return None, 0.0
