# AI-Based Automotive Review and Customer Sentiment Analytics

**AI/ML Technologies**: BERT, XLM-R, NLP  
**Domain**: Aspect-Based Sentiment Analysis (ABSA) for Automotive Engineering & Customer Feedback

---

## 🎯 Project Goals

Automotive customer reviews contain complex, multi-faceted sentiments. Standard sentiment classifiers collapse a customer's review into a single score, hiding critical engineering feedback. For example:
> *"The battery range is excellent, but the charging time is too long."*

Here, standard sentiment models assign a flat "Neutral" or "Positive" label. In contrast, this system extracts individual automotive aspects and identifies their distinct sentiment:
- **Battery range** $\rightarrow$ **Positive**
- **Charging time** $\rightarrow$ **Negative**

---

## 🚗 Monitored Automotive Review Aspects

Automotive reviews are systematically analyzed across **9 core vehicle dimensions**:

1. **Vehicle**: Build quality, exterior styling, handling, design, chassis, road presence.
2. **Engine**: Engine power, horsepower, torque, acceleration, turbo, transmission.
3. **Battery**: Battery range, charging time, fast charging, wallbox, battery health, kWh.
4. **Mileage**: Fuel economy, fuel efficiency, gas mileage, kmpl, mpg, consumption.
5. **Safety**: Braking system, airbags, collision avoidance, driver assistance (ADAS), crash test rating.
6. **Comfort**: Ride quality, cabin quietness, seat comfort, rear seat legroom, suspension.
7. **Service**: Dealership service, maintenance, customer service, repair turnaround, warranty.
8. **Infotainment**: Touchscreen display, Apple CarPlay, Android Auto, sound system, audio, connectivity.
9. **Price**: Value for money, purchase price, maintenance cost, affordability, resale value.

---

## 🏗️ System Architecture

The pipeline strictly implements the 7-stage NLP and Deep Learning workflow:

```text
       Automotive Reviews
               ↓
       Text Preprocessing
  (Cleaning, Normalization, Clause Splitting)
               ↓
           Tokenizer
  (BERT WordPiece / XLM-R SentencePiece BPE)
               ↓
          BERT / XLM-R
  (Pretrained Transformer Sequence Encoders)
               ↓
     Sentiment Classification
   (Negative, Neutral, Positive Logits)
               ↓
        Aspect Extraction
  (9 Automotive Domains & Sub-Aspect Terms)
               ↓
      Aspect-Level Sentiment
  (Granular Aspect Polarity Mapping)
```

---

## 🧪 Specification Example & Output

### Input Review:
> *"The battery range is excellent, but the charging time is too long."*

### Output:
| Aspect | Battery range | Charging time |
| :--- | :---: | :---: |
| **Sentiment** | **Positive** | **Negative** |

---

## 📁 Project Directory Structure

```text
AI_Automotive_Sentiment_Analytics/
├── app.py                             # Interactive Streamlit Web Application
├── run_pipeline.py                    # CLI tool for analyzing any vehicle review
├── test_cases.py                      # Automated test suite verifying spec examples
├── train_models.py                    # Fine-tuning script for BERT, XLM-R & baseline
├── generate_pdf_report.py             # Generates standalone technical project report (PDF)
├── requirements.txt                   # Project Python dependencies
├── README.md                          # Comprehensive documentation
├── data/
│   └── car_reviews_.csv               # Cleaned automotive customer reviews dataset
├── models/
│   ├── bert_sentiment/                # Fine-tuned BERT model weights & tokenizer
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── xlmr_tokenizer/                # XLM-R Tokenizer cache
│   ├── baseline_model.joblib          # Logistic Regression ML baseline
│   └── tfidf_vectorizer.joblib        # TF-IDF feature extractor
└── src/
    ├── __init__.py
    ├── preprocessing.py               # Step 1: Text preprocessing & clause segmentation
    ├── tokenizer.py                   # Step 2: BERT & XLM-R tokenizers
    ├── sentiment_classifier.py        # Steps 3 & 4: Transformer sentiment classifier
    ├── aspect_extraction.py           # Step 5: Aspect extraction across 9 domains
    └── pipeline.py                    # Steps 6 & 7: Unified 7-stage orchestrator
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Verify Canonical Specification Example
Run the automated test suite:
```powershell
python test_cases.py
```

### 3. Analyze Any Review via CLI
```powershell
python run_pipeline.py --review "The battery range is excellent, but the charging time is too long."
```
```powershell
python run_pipeline.py --review "The mileage is excellent and the car is very comfortable." --model BERT
```

### 4. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test with interactive review inputs, model switches, and instant aspect breakdown tables.

### 5. Generate Project PDF Report
```powershell
python generate_pdf_report.py
```
Creates `Automotive_Sentiment_Analytics_Report.pdf` with formatted architecture diagrams, 9-aspect catalogue, and evaluation tables.
