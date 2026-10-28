"""
Unit Tests for Advanced AI Engines:
1. Instant Image Classifier (MobileNetV2 ImageNet-1k)
2. NLP Engine (Sentiment, Keywords, Summarization, Classifier)
3. Universal Recommendation Engine (Content-Based & Collaborative)
4. Time-Series Forecasting Engine (Trend + Harmonics + 95% CI)
5. Unsupervised Anomaly Detection (Isolation Forest + PCA)
6. Generative AI Engine (Synthetic Tabular Generator & Executive Storyteller)
"""

import io
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from src.image_trainer import predict_any_image_instant
from src.nlp_engine import (
    analyze_text_sentiment,
    extract_keywords_and_ngrams,
    summarize_text_extractive,
    train_text_classifier,
    analyze_dataset_text_column,
)
from src.recommender import build_content_recommender, get_recommendations
from src.forecasting import (
    detect_time_series_columns,
    prepare_time_series,
    train_time_series_forecast,
)
from src.anomaly_engine import detect_anomalies, export_clean_dataset_without_anomalies
from src.genai_engine import (
    generate_synthetic_tabular_data,
    generate_executive_narrative,
)


def test_instant_image_prediction_pil_and_bytes():
    # 1. Test with PIL Image
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    res = predict_any_image_instant(img, top_k=5)

    assert "predicted_class" in res
    assert isinstance(res["predicted_class"], str)
    assert res["confidence"] > 0
    assert len(res["top_predictions"]) == 5
    assert "inference_time_ms" in res

    # 2. Test with raw file bytes (regression test for io.BytesIO)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_data = buf.getvalue()

    res_bytes = predict_any_image_instant(byte_data, top_k=3)
    assert "predicted_class" in res_bytes
    assert len(res_bytes["top_predictions"]) == 3


def test_nlp_sentiment_and_keywords():
    pos_res = analyze_text_sentiment("The user interface is fantastic, fast, reliable, and clean!")
    assert pos_res["label"] == "POSITIVE"
    assert pos_res["polarity"] > 0

    neg_res = analyze_text_sentiment("This process is slow, broken, full of errors and awful bugs.")
    assert neg_res["label"] == "NEGATIVE"
    assert neg_res["polarity"] < 0

    neu_res = analyze_text_sentiment("The table contains seven rows and three columns.")
    assert neu_res["label"] == "NEUTRAL"

    # Keywords
    kw_res = extract_keywords_and_ngrams(["Machine learning models provide high accuracy.", "Deep learning neural networks."], top_n=5)
    assert not kw_res["unigrams"].empty

    # Summarization
    long_text = "DataFlow is an automated AI platform. It ingests any dataset cleanly. It runs AutoML algorithms. It predicts values in milliseconds. It provides voice translation in Tamil."
    summary_res = summarize_text_extractive(long_text, max_sentences=2)
    assert len(summary_res["summary"]) > 0

    # Text Classification
    df_text = pd.DataFrame({
        "feedback": [
            "Great product, loved it!", "Best customer support ever!", "Amazing service and speed.",
            "Worst experience, broke immediately.", "Terrible customer service, rude staff.", "Completely useless and failed.",
        ],
        "category": ["Positive", "Positive", "Positive", "Negative", "Negative", "Negative"],
    })
    clf_res = train_text_classifier(df_text, "feedback", "category")
    assert clf_res["cv_score"] > 0
    pred = clf_res["predict_fn"]("Super happy and delighted!")
    assert pred["prediction"] == "Positive"


def test_universal_recommender():
    df_items = pd.DataFrame({
        "item_name": ["Laptop Pro", "Gaming Laptop", "Wireless Mouse", "Mechanical Keyboard", "Coffee Mug"],
        "category": ["Electronics", "Electronics", "Accessories", "Accessories", "Kitchenware"],
        "price": [1200, 1500, 30, 90, 15],
        "rating": [4.8, 4.7, 4.2, 4.6, 4.0],
    })

    rec_obj = build_content_recommender(df_items, item_col="item_name")
    assert rec_obj["total_items"] == 5

    res = get_recommendations(rec_obj, selected_item="Laptop Pro", top_n=3)
    assert len(res["recommendations_df"]) == 3
    # Gaming Laptop should be ranked higher than Coffee Mug for Laptop Pro
    top_rec = res["recommendations_df"].iloc[0]["item_name"]
    assert top_rec == "Gaming Laptop"


def test_time_series_forecasting():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    values = np.sin(np.linspace(0, 10, 30)) * 10 + np.linspace(50, 100, 30)
    df_ts = pd.DataFrame({"record_date": dates, "revenue": values})

    d_cols, n_cols = detect_time_series_columns(df_ts)
    assert "record_date" in d_cols
    assert "revenue" in n_cols

    prep_df = prepare_time_series(df_ts, "record_date", "revenue")
    res = train_time_series_forecast(prep_df, "record_date", "revenue", horizon=7)

    assert len(res["forecast_df"]) == 7
    assert "Lower Bound (95%)" in res["forecast_df"]
    assert "Upper Bound (95%)" in res["forecast_df"]
    assert res["metrics"]["rmse"] >= 0
    # Upper bound should be strictly greater than lower bound
    assert (res["forecast_df"]["Upper Bound (95%)"] >= res["forecast_df"]["Lower Bound (95%)"]).all()


def test_anomaly_detection_isolation_forest():
    # 50 normal points around 10.0, plus 3 extreme outliers
    np.random.seed(42)
    norm_vals = np.random.normal(10, 1, 50)
    outliers = np.array([500.0, -300.0, 999.0])
    all_vals = np.concatenate([norm_vals, outliers])
    df_anom = pd.DataFrame({"sensor_reading": all_vals, "ambient_temp": all_vals * 0.5})

    res = detect_anomalies(df_anom, contamination=0.06)
    assert res["anomaly_count"] > 0
    assert "_is_anomaly" in res["results_df"]
    assert not res["anomalies_only_df"].empty

    clean_df = export_clean_dataset_without_anomalies(df_anom, res["results_df"])
    assert len(clean_df) < len(df_anom)


def test_generative_ai_synthetic_data_and_story():
    df_real = pd.DataFrame({
        "age": [25, 30, 45, 50, 22, 60],
        "salary": [50000, 65000, 90000, 110000, 42000, 130000],
        "department": ["IT", "Sales", "IT", "HR", "Sales", "IT"],
    })

    synth_df, meta = generate_synthetic_tabular_data(df_real, num_samples=25)
    assert len(synth_df) == 25
    assert set(synth_df.columns) == set(df_real.columns)
    assert meta["fidelity_score_pct"] > 50

    narrative = generate_executive_narrative(df_real, dataset_name="Employee Data")
    assert "narrative" in narrative
    assert len(narrative["narrative"]) > 100
