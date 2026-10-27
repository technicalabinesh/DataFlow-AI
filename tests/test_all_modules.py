"""
=============================================================================
End-to-End Test Suite for All Universal AI Data Analyst Modules
=============================================================================
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.cleaner import execute_full_cleaning_pipeline, get_downloadable_bytes
from src.eda import (
    generate_correlation_heatmap,
    generate_distribution_chart,
    generate_categorical_chart,
    build_custom_chart,
)
from src.chatbot import check_code_safety, execute_safe_query, offline_fallback_analyst
from src.reporter import generate_pdf_report, generate_word_report
from src.loader import compute_dataset_summary
from src.samples import ensure_sample_datasets


@pytest.fixture(scope="session")
def sample_data():
    paths = ensure_sample_datasets()
    df_t = pd.read_csv(paths["titanic"])
    return df_t


def test_cleaner_pipeline(sample_data):
    cleaned_df, log = execute_full_cleaning_pipeline(
        sample_data,
        standardize_headers=True,
        coerce_types=True,
        numeric_impute_strategy="median",
        cat_impute_strategy="mode",
        outlier_method="clip",
    )
    assert cleaned_df.isnull().sum().sum() == 0  # Zero missing values!
    assert len(log) > 0
    csv_bytes = get_downloadable_bytes(cleaned_df, "csv")
    assert len(csv_bytes) > 0


def test_eda_generators(sample_data):
    fig_corr, insight_corr = generate_correlation_heatmap(sample_data)
    assert fig_corr is not None
    assert "Insight" in insight_corr

    fig_dist, insight_dist = generate_distribution_chart(sample_data, "Fare")
    assert fig_dist is not None

    fig_cat, insight_cat = generate_categorical_chart(sample_data, "Sex")
    assert fig_cat is not None

    custom_fig = build_custom_chart(sample_data, "Scatter", "Age", "Fare", "Survived")
    assert custom_fig is not None

    from src.eda import generate_powerbi_matrix, generate_powerbi_gauge
    matrix_df = generate_powerbi_matrix(sample_data, "Sex", "Fare")
    assert not matrix_df.empty
    assert "Sum" in matrix_df.columns
    assert "Mean" in matrix_df.columns

    fig_gauge = generate_powerbi_gauge(32.5, 0, 100, "Average Fare")
    assert fig_gauge is not None


def test_chatbot_safety_and_sandbox(sample_data):
    # Safety filter test
    safe, _ = check_code_safety("import os; os.system('calc')")
    assert safe is False

    safe, _ = check_code_safety("result = df['Age'].mean()")
    assert safe is True

    # Sandboxed execution test
    exec_res = execute_safe_query("result = df['Fare'].max()", sample_data)
    assert exec_res["type"] == "scalar"
    assert exec_res["result"] == sample_data["Fare"].max()


def test_chatbot_offline_fallback(sample_data):
    res = offline_fallback_analyst("How many rows in dataset?", sample_data)
    assert "200" in res["answer"]
    assert res["type"] == "scalar"


def test_report_generation(sample_data):
    summary = compute_dataset_summary(sample_data)
    log = ["Imputed missing values", "Standardized headers"]
    chat = [
        {"role": "user", "content": "What is the average fare?", "timestamp": "12:00:00"},
        {"role": "assistant", "content": "The average fare is 32.20", "code": "result = df['Fare'].mean()", "timestamp": "12:00:01"},
    ]

    pdf_bytes = generate_pdf_report("Titanic Test", summary, log, chat_history=chat)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    docx_bytes = generate_word_report("Titanic Test", summary, log, chat_history=chat)
    assert len(docx_bytes) > 1000

    from src.reporter import generate_json_session_export
    json_export = generate_json_session_export("Titanic Test", summary, log, chat_history=chat)
    assert "average fare" in json_export

