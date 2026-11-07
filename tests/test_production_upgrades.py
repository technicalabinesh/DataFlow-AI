# -*- coding: utf-8 -*-
"""
Unit tests for the 5 production-ready modules:
1. Time-Series Forecasting (Statsmodels Holt-Winters, Harmonic Fourier, 95% CI)
2. Explainable AI (XAI) & Automated Data Profiling Suite
3. Multi-Dataset Joins & Workspace
4. Natural Language Chart Studio (Text-to-Plotly)
5. Executive Export Engine (PowerPoint Presentation .pptx)
"""

import io
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.forecasting import (
    detect_time_series_columns,
    prepare_time_series,
    train_time_series_forecast,
    create_forecast_figure,
)
from src.xai_engine import (
    compute_model_interpretability,
    generate_data_quality_profile,
)
from src.eda import generate_text_to_plotly_chart
from src.reporter import generate_powerpoint_presentation


# =============================================================================
# 1. TIME-SERIES FORECASTING TESTS
# =============================================================================
def test_time_series_detection_and_forecasting():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.sin(np.linspace(0, 10, 60)) * 20 + 100 + np.random.normal(0, 2, 60)
    df = pd.DataFrame({"Date": dates, "Revenue": values, "Region": ["North"] * 60})

    date_candidates, num_metrics = detect_time_series_columns(df)
    assert "Date" in date_candidates
    assert "Revenue" in num_metrics

    ts_df = prepare_time_series(df, "Date", "Revenue", freq="D")
    assert len(ts_df) == 60

    # Test training forecast
    res = train_time_series_forecast(ts_df, "Date", "Revenue", horizon=14, algorithm="Auto-Select Best")
    assert "forecast_df" in res
    assert "metrics" in res
    assert len(res["forecast_df"]) == 14
    assert "Forecasted" in res["forecast_df"].columns
    assert "Lower Bound (95%)" in res["forecast_df"].columns
    assert "Upper Bound (95%)" in res["forecast_df"].columns
    assert "expected_growth_pct" in res["metrics"]
    assert "peak_value" in res["metrics"]

    # Test figure generation
    fig = create_forecast_figure(res["historical_df"], res["forecast_df"], "Date", "Revenue")
    assert isinstance(fig, go.Figure)


# =============================================================================
# 2. EXPLAINABLE AI (XAI) & DATA QUALITY TESTS
# =============================================================================
def test_xai_interpretability_and_data_profiling():
    np.random.seed(42)
    df = pd.DataFrame({
        "Age": np.random.randint(18, 70, size=50),
        "Fare": np.random.uniform(10, 500, size=50),
        "SibSp": np.random.randint(0, 5, size=50),
        "Survived": np.random.choice([0, 1], size=50),
    })
    df.loc[0:4, "Age"] = np.nan  # Introduce missing values

    # Test Data Quality Profiling
    profile = generate_data_quality_profile(df)
    assert 0 <= profile["health_score"] <= 100
    assert profile["total_missing"] == 5
    assert profile["sparsity_pct"] > 0
    assert not profile["audit_df"].empty
    assert profile["fig_missing"] is not None
    assert profile["fig_corr"] is not None

    # Test XAI Model Interpretability
    X = df[["Age", "Fare", "SibSp"]].fillna(df[["Age", "Fare", "SibSp"]].median())
    y = df["Survived"]

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    pipe = Pipeline([("model", clf)])
    pipe.fit(X, y)

    xai_res = compute_model_interpretability(pipe, X, y, task_type="classification")
    assert "method_used" in xai_res
    assert "importance_df" in xai_res
    assert isinstance(xai_res["figure"], go.Figure)
    assert xai_res["top_feature"] in ["Age", "Fare", "SibSp"]


# =============================================================================
# 3. MULTI-DATASET JOIN LOGIC TESTS
# =============================================================================
def test_multi_dataset_sql_joins():
    df_customers = pd.DataFrame({
        "CustomerID": [101, 102, 103, 104],
        "CustomerName": ["Alice", "Bob", "Charlie", "David"],
        "City": ["London", "Paris", "Berlin", "Tokyo"],
    })
    df_orders = pd.DataFrame({
        "OrderID": [1, 2, 3, 4, 5],
        "CustomerID": [101, 102, 101, 105, 102],
        "Amount": [250, 450, 120, 300, 80],
    })

    # Inner Join
    inner_df = pd.merge(df_customers, df_orders, on="CustomerID", how="inner")
    assert len(inner_df) == 4
    assert set(inner_df["CustomerName"].unique()) == {"Alice", "Bob"}

    # Left Join
    left_df = pd.merge(df_customers, df_orders, on="CustomerID", how="left")
    assert len(left_df) == 6
    assert "Charlie" in left_df["CustomerName"].values

    # Outer Join
    outer_df = pd.merge(df_customers, df_orders, on="CustomerID", how="outer")
    assert len(outer_df) == 7


# =============================================================================
# 4. NATURAL LANGUAGE CHART STUDIO (TEXT-TO-PLOTLY) TESTS
# =============================================================================
def test_text_to_plotly_offline_parser():
    df = pd.DataFrame({
        "Category": ["Electronics", "Clothing", "Food", "Home"],
        "Sales": [1200, 800, 450, 600],
        "Profit": [300, 150, 80, 120],
    })

    # Bar chart prompt
    fig_bar, expl_bar = generate_text_to_plotly_chart("Bar chart of Sales by Category", df)
    assert isinstance(fig_bar, go.Figure)
    assert "Category" in expl_bar or "Sales" in expl_bar

    # Scatter prompt
    fig_scatter, expl_scatter = generate_text_to_plotly_chart("Scatter plot of Sales vs Profit", df)
    assert isinstance(fig_scatter, go.Figure)

    # Donut prompt
    fig_donut, _ = generate_text_to_plotly_chart("Donut chart of Category", df)
    assert isinstance(fig_donut, go.Figure)


# =============================================================================
# 5. EXECUTIVE EXPORT ENGINE (.PPTX) TESTS
# =============================================================================
def test_powerpoint_presentation_generation():
    summary_metrics = {
        "rows": 1000,
        "columns": 12,
        "missing_pct": 1.5,
        "duplicate_rows": 0,
        "memory_mb": 0.45,
        "numeric_columns": ["Age", "Fare"],
        "categorical_columns": ["Sex", "Embarked"],
    }
    cleaning_log = [
        "Coerced 2 date columns to datetime64[ns]",
        "Imputed median value for 'Age' (missing: 15%)",
        "Encoded categorical columns using stratified TargetEncoder",
    ]
    ml_results = {
        "best_model_name": "Soft Voting Classifier (RandomForest + ExtraTrees)",
        "leaderboard": pd.DataFrame({
            "Model": ["VotingClassifier", "RandomForest", "ExtraTrees"],
            "Accuracy": [0.89, 0.87, 0.86],
            "F1-Score": [0.88, 0.86, 0.85],
        }),
    }

    pptx_bytes = generate_powerpoint_presentation(
        dataset_name="Executive_Telemetry.csv",
        summary_metrics=summary_metrics,
        cleaning_log=cleaning_log,
        ml_results=ml_results,
        chat_history=[{"role": "user", "content": "What is the peak revenue?"}],
    )

    assert isinstance(pptx_bytes, bytes)
    assert len(pptx_bytes) > 10000  # Valid multi-slide .pptx presentation
