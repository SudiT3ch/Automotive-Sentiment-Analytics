"""
app.py
------
Project Title: AI-Based Automotive Review and Customer Sentiment Analytics
AI/ML Technologies: BERT, XLM-R, NLP
Goals: Analyze vehicle reviews and identify sentiment for individual aspects.

Data:
Automotive reviews analyzed for:
Vehicle, Engine, Battery, Mileage, Safety, Comfort, Service, Infotainment, Price

Example:
"The battery range is excellent, but the charging time is too long."

Output:
Aspect:    Battery range | Charging time
Sentiment: Positive      | Negative
"""

import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import get_pipeline

# Page setup
st.set_page_config(
    page_title="AI-Based Automotive Review and Customer Sentiment Analytics",
    page_icon="🚗",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .project-header {
        text-align: center;
        padding: 1rem 1rem 0.3rem 1rem;
    }
    .project-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }
    .tech-badges {
        margin-bottom: 0.8rem;
    }
    .tech-pill {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 9999px;
        margin: 2px 4px;
        border: 1px solid #bae6fd;
    }
    .goal-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        font-size: 1.02rem;
        color: #166534;
        font-weight: 600;
        margin-bottom: 1.2rem;
    }
    .aspect-domain-tag {
        display: inline-block;
        background-color: #f8fafc;
        color: #334155;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 2px 3px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .spec-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 1.05rem;
    }
    .spec-table th, .spec-table td {
        border: 1px solid #cbd5e1;
        padding: 12px 18px;
        text-align: center;
    }
    .spec-table th {
        background-color: #f1f5f9;
        font-weight: 700;
        color: #0f172a;
    }
    .badge-pos {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.95rem;
    }
    .badge-neg {
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.95rem;
    }
    .badge-neu {
        background-color: #fef3c7;
        color: #b45309;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="project-header">
    <div class="project-title">AI-Based Automotive Review and Customer Sentiment Analytics</div>
    <div class="tech-badges">
        <span class="tech-pill">BERT</span>
        <span class="tech-pill">XLM-R</span>
        <span class="tech-pill">NLP</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Goals Display
st.markdown("""
<div class="goal-box">
    🎯 <b>Goals:</b> Analyze vehicle reviews and identify sentiment for individual aspects.
</div>
""", unsafe_allow_html=True)

# Monitored Aspects
st.markdown("##### 🚗 Automotive Review Aspects Analyzed:")
aspect_domains = [
    "Vehicle", "Engine", "Battery", "Mileage",
    "Safety", "Comfort", "Service", "Infotainment", "Price"
]
st.markdown("".join(f'<span class="aspect-domain-tag">{d}</span>' for d in aspect_domains), unsafe_allow_html=True)

st.divider()

# Controls & Preset Selection
col_ctrl1, col_ctrl2 = st.columns([1, 2], gap="medium")

with col_ctrl1:
    selected_model = st.selectbox(
        "Select Model:",
        options=["BERT", "XLM-R"],
        index=0,
        help="Select transformer model (BERT or XLM-R) for sentiment inference."
    )

with col_ctrl2:
    st.write("Preset Automotive Reviews:")
    p_col1, p_col2, p_col3 = st.columns(3)
    preset_choice = None
    if p_col1.button("🔋 Canonical Battery Example", use_container_width=True):
        preset_choice = "The battery range is excellent, but the charging time is too long."
    if p_col2.button("⛽ Mileage & Comfort", use_container_width=True):
        preset_choice = "The mileage is excellent and the car is very comfortable."
    if p_col3.button("🔧 Service & Maintenance Cost", use_container_width=True):
        preset_choice = "The service is poor and the maintenance cost is expensive."

# Review Input Box
default_text = preset_choice if preset_choice else "The battery range is excellent, but the charging time is too long."
user_review = st.text_area(
    "Automotive Review Text:",
    value=default_text,
    height=90,
    help="Enter a vehicle review to extract individual aspects and their specific sentiments."
)

if st.button("🔍 Analyze Review & Identify Aspect Sentiments", type="primary", use_container_width=True):
    if not user_review.strip():
        st.warning("Please enter a review text.")
    else:
        with st.spinner(f"Analyzing review using {selected_model}..."):
            pipeline = get_pipeline(model_type=selected_model)
            res = pipeline.analyze_review(user_review)

        # Output Section
        st.markdown("### 📊 Analysis Output")

        if not res["aspects"]:
            st.info("No specific aspect terms detected. Overall review sentiment classified below.")
        else:
            # 1. Specification Format Horizontal Table
            st.markdown("#### 📌 Aspect Sentiment Table")
            aspect_items = res["aspects"]

            th_aspects = "".join(f"<th>{item['aspect']}</th>" for item in aspect_items)
            td_sentiments = ""
            for item in aspect_items:
                s = item["sentiment"]
                badge_class = "badge-pos" if s == "Positive" else ("badge-neg" if s == "Negative" else "badge-neu")
                td_sentiments += f'<td><span class="{badge_class}">{s}</span></td>'

            html_table = f"""
            <table class="spec-table">
                <tr>
                    <th style="width: 15%;">Aspect</th>
                    {th_aspects}
                </tr>
                <tr>
                    <th>Sentiment</th>
                    {td_sentiments}
                </tr>
            </table>
            """
            st.markdown(html_table, unsafe_allow_html=True)

            # 2. Detailed Aspect Breakdown
            st.markdown("#### 🔍 Breakdown by Aspect Domain & Extracted Clause")
            detail_data = []
            for item in aspect_items:
                detail_data.append({
                    "Aspect": item["aspect"],
                    "Domain Category": item["domain"],
                    "Sentiment": item["sentiment"],
                    "Confidence": f"{item['confidence'] * 100:.1f}%",
                    "Extracted Clause": f'"{item["clause"]}"'
                })
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True)

        # Summary Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Overall Review Sentiment", res["overall_sentiment"])
        with m2:
            st.metric("Confidence", f"{res['overall_confidence'] * 100:.1f}%")
        with m3:
            st.metric("Model Used", res["model_used"])

st.markdown("---")
st.caption("AI-Based Automotive Review and Customer Sentiment Analytics | BERT, XLM-R, NLP")
