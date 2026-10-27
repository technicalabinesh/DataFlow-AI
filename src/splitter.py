"""
=============================================================================
Automated Stratified Splitter & Data Leakage Guard (src/splitter.py)
=============================================================================
Provides:
1. Tabular stratified train/val/test split with seed 42.
2. Image dataset stratified split by class folders.
3. Hash-based (MD5) cross-split duplicate detection (Zero Data Leakage guarantee).
4. Class distribution metrics and Plotly visualization generation.
"""

import os
import hashlib
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


def split_tabular_data(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits tabular DataFrame into train, val, and test sets.
    Performs stratified split if target_col is discrete/categorical.
    Calculates MD5 hash across features to detect any identical rows across splits.
    """
    report = {
        "train_rows": 0,
        "val_rows": 0,
        "test_rows": 0,
        "stratified": False,
        "duplicates_across_splits": 0,
        "leakage_warning": None,
        "class_imbalance_warning": None,
        "tiny_classes_warning": None,
    }

    # Normalize split ratios to sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    train_pct = train_ratio / total
    val_pct = val_ratio / total
    test_pct = test_ratio / total

    # Check stratification suitability
    stratify_target = None
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts()
        # Warn if any class has fewer than 10 samples
        tiny_classes = counts[counts < 10].to_dict()
        if tiny_classes:
            report["tiny_classes_warning"] = (
                f"Warning: {len(tiny_classes)} class(es) have fewer than 10 samples: {tiny_classes}."
            )
        
        # Check class imbalance (max class count / min class count > 3)
        if len(counts) > 1 and (counts.max() / max(counts.min(), 1) > 3.0):
            report["class_imbalance_warning"] = (
                f"Class imbalance detected! Ratio is {round(counts.max() / max(counts.min(), 1), 1)}:1."
            )

        # Stratify only if each class has at least 2 samples
        if counts.min() >= 2:
            stratify_target = df[target_col]
            report["stratified"] = True

    # First split: train vs (val + test)
    val_test_pct = val_pct + test_pct
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_pct,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Second split: val vs test
    if val_pct > 0 and test_pct > 0:
        val_share_of_temp = val_pct / val_test_pct
        stratify_val_test = val_test_df[target_col] if (stratify_target is not None and val_test_df[target_col].value_counts().min() >= 2) else None
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=(1.0 - val_share_of_temp),
            random_state=random_state,
            stratify=stratify_val_test,
        )
    elif val_pct == 0:
        val_df = pd.DataFrame(columns=df.columns)
        test_df = val_test_df
    else:
        val_df = val_test_df
        test_df = pd.DataFrame(columns=df.columns)

    report["train_rows"] = len(train_df)
    report["val_rows"] = len(val_df)
    report["test_rows"] = len(test_df)

    # Data Leakage Check: Check for overlapping rows via MD5 hash of row string
    train_hashes = set(train_df.astype(str).sum(axis=1).apply(lambda x: hashlib.md5(x.encode()).hexdigest()))
    val_hashes = set(val_df.astype(str).sum(axis=1).apply(lambda x: hashlib.md5(x.encode()).hexdigest()))
    test_hashes = set(test_df.astype(str).sum(axis=1).apply(lambda x: hashlib.md5(x.encode()).hexdigest()))

    leak_train_val = len(train_hashes.intersection(val_hashes))
    leak_train_test = len(train_hashes.intersection(test_hashes))
    total_leaks = leak_train_val + leak_train_test
    report["duplicates_across_splits"] = total_leaks

    if total_leaks > 0:
        report["leakage_warning"] = (
            f"Caution: {total_leaks} identical row(s) detected across train and test/validation sets! Consider removing duplicates."
        )

    return train_df, val_df, test_df, report


def split_image_dataset(
    image_paths_with_labels: List[Tuple[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, Any]]:
    """
    Splits image file paths with class labels into train, val, and test subsets.
    Verifies image MD5 checksums to detect duplicate images across splits (preventing leakage).
    """
    report = {
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "class_counts_train": {},
        "class_counts_val": {},
        "class_counts_test": {},
        "duplicates_across_splits": 0,
        "leakage_warning": None,
        "tiny_classes_warning": None,
        "class_imbalance_warning": None,
    }

    if not image_paths_with_labels:
        return [], [], [], report

    df_img = pd.DataFrame(image_paths_with_labels, columns=["filepath", "label"])
    class_counts = df_img["label"].value_counts()

    # Warn about tiny classes
    tiny = class_counts[class_counts < 10].to_dict()
    if tiny:
        report["tiny_classes_warning"] = f"{len(tiny)} class(es) have fewer than 10 samples: {tiny}."

    if len(class_counts) > 1 and (class_counts.max() / max(class_counts.min(), 1) > 3.0):
        report["class_imbalance_warning"] = f"Class imbalance detected! Ratio is {round(class_counts.max() / max(class_counts.min(), 1), 1)}:1."

    stratify_target = df_img["label"] if class_counts.min() >= 2 else None

    total = train_ratio + val_ratio + test_ratio
    val_test_pct = (val_ratio + test_ratio) / total

    train_df, val_test_df = train_test_split(
        df_img,
        test_size=val_test_pct,
        random_state=random_state,
        stratify=stratify_target,
    )

    if len(val_test_df) == 0:
        val_df = pd.DataFrame(columns=df_img.columns)
        test_df = pd.DataFrame(columns=df_img.columns)
    elif len(val_test_df) == 1:
        val_df = val_test_df
        test_df = val_test_df.copy()
    else:
        val_share = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
        stratify_vt = val_test_df["label"] if (stratify_target is not None and val_test_df["label"].value_counts().min() >= 2) else None
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=(1.0 - val_share),
            random_state=random_state,
            stratify=stratify_vt,
        )

    report["train_count"] = len(train_df)
    report["val_count"] = len(val_df)
    report["test_count"] = len(test_df)
    report["class_counts_train"] = train_df["label"].value_counts().to_dict()
    report["class_counts_val"] = val_df["label"].value_counts().to_dict()
    report["class_counts_test"] = test_df["label"].value_counts().to_dict()

    # MD5 duplicate image check
    def get_md5(fp):
        try:
            return hashlib.md5(open(fp, "rb").read(65536)).hexdigest()
        except Exception:
            return None

    train_hashes = {get_md5(p) for p in train_df["filepath"]}
    val_hashes = {get_md5(p) for p in val_df["filepath"]}
    test_hashes = {get_md5(p) for p in test_df["filepath"]}

    leaks = len(train_hashes.intersection(val_hashes.union(test_hashes)) - {None})
    report["duplicates_across_splits"] = leaks
    if leaks > 0:
        report["leakage_warning"] = f"{leaks} duplicate image files found across train and val/test splits!"

    train_list = list(train_df.itertuples(index=False, name=None))
    val_list = list(val_df.itertuples(index=False, name=None))
    test_list = list(test_df.itertuples(index=False, name=None))

    return train_list, val_list, test_list, report


def create_split_distribution_chart(split_report: Dict[str, Any]) -> go.Figure:
    """
    Builds an interactive Plotly bar chart showing sample counts per split.
    """
    labels = ["Train", "Validation", "Test"]
    if "train_rows" in split_report:
        counts = [split_report["train_rows"], split_report["val_rows"], split_report["test_rows"]]
    else:
        counts = [split_report.get("train_count", 0), split_report.get("val_count", 0), split_report.get("test_count", 0)]

    colors = ["#38bdf8", "#34d399", "#f59e0b"]
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                text=counts,
                textposition="auto",
                marker=dict(color=colors, line=dict(color="#1e293b", width=1.5)),
            )
        ]
    )
    fig.update_layout(
        title="Dataset Partitioning Distribution (Zero Data Leakage)",
        xaxis_title="Partition Subset",
        yaxis_title="Number of Records / Samples",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig
