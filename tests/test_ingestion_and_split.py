"""
=============================================================================
Unit Test Suite: Universal Ingestion, Folder Detection & Auto-Split
=============================================================================
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.loader import (
    load_tabular_from_bytes,
    detect_encoding_and_delimiter,
    compute_dataset_summary,
    scan_dataset_directory,
)
from src.splitter import split_tabular_data, split_image_dataset
from src.automl import train_tabular_models, infer_task_type
from src.samples import ensure_sample_datasets


@pytest.fixture(scope="session")
def sample_paths():
    return ensure_sample_datasets()


def test_csv_ingestion_and_summary(sample_paths):
    with open(sample_paths["titanic"], "rb") as f:
        file_bytes = f.read()

    df, meta = load_tabular_from_bytes(file_bytes, "titanic.csv")
    assert df is not None
    assert len(df) == 200
    assert meta["encoding"] == "utf-8"
    assert meta["delimiter"] == ","

    summary = compute_dataset_summary(df)
    assert summary["rows"] == 200
    assert summary["columns"] == 10
    assert summary["missing_cells"] > 0
    assert summary["memory_mb"] > 0


def test_folder_scanner_image_mode(sample_paths):
    img_folder = sample_paths["image_folder"]
    res = scan_dataset_directory(img_folder)
    assert res["exists"] is True
    assert res["mode"] == "images_per_class"
    assert len(res["classes"]) == 2
    assert "cats" in res["classes"]
    assert "dogs" in res["classes"]
    assert len(res["images"]) == 40


def test_stratified_tabular_split_zero_leakage(sample_paths):
    df = pd.read_csv(sample_paths["titanic"])
    train_df, val_df, test_df, report = split_tabular_data(
        df, target_col="Survived", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    assert len(train_df) == 140
    assert len(val_df) == 30
    assert len(test_df) == 30
    assert report["stratified"] is True
    assert report["duplicates_across_splits"] == 0  # Zero data leakage!


def test_automl_tabular_pipeline(sample_paths):
    df = pd.read_csv(sample_paths["titanic"])
    train_df, val_df, test_df, _ = split_tabular_data(df, target_col="Survived")
    results = train_tabular_models(train_df, test_df, target_col="Survived")
    
    assert results["task_type"] == "classification"
    assert len(results["leaderboard"]) >= 3
    assert results["best_pipeline"] is not None
    assert os.path.exists(results["model_save_path"])
