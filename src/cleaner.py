"""
=============================================================================
Autonomous Data Cleaning Engine (src/cleaner.py)
=============================================================================
Features:
1. Universal type detection & coercion (numeric strings, date parsing, whitespace).
2. Missing value strategies: Mean, Median, Mode, Constant, Forward-fill, Drop.
3. Outlier handling via Interquartile Range (IQR) with toggle (Keep, Clip, Drop).
4. Duplicate row detection and removal.
5. Column name standardization (trim, lower, snake_case).
6. Comprehensive audit log of every cleaning operation with timestamps.
7. Clean vs. Raw comparison metrics.
8. Instant export to CSV and Excel (.xlsx) formats.
"""

import io
import re
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np


def clean_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Standardizes column names: strips whitespace, removes special characters,
    converts to clean snake_case.
    """
    new_cols = []
    log = []
    for col in df.columns:
        original = str(col)
        cleaned = re.sub(r"[^\w\s]", "", original).strip()
        cleaned = re.sub(r"\s+", "_", cleaned).lower()
        if not cleaned:
            cleaned = "column"
        new_cols.append(cleaned)
        if cleaned != original:
            log.append(f"Renamed column '{original}' → '{cleaned}'")

    cleaned_df = df.copy()
    cleaned_df.columns = new_cols
    return cleaned_df, log


def auto_coerce_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Auto-detects columns stored as object/string that are actually numeric or datetime.
    """
    cleaned_df = df.copy()
    log = []

    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            sample = cleaned_df[col].dropna().astype(str).str.strip()
            if len(sample) == 0:
                continue

            # Check if numeric formatted with currency/commas/percentages (e.g. $1,234.50 or 95%)
            clean_num = sample.str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            num_converted = pd.to_numeric(clean_num, errors="coerce")
            
            # If > 80% successfully convert to numeric, coerce the column
            valid_pct = num_converted.notnull().sum() / len(sample)
            if valid_pct > 0.8:
                orig_type = str(cleaned_df[col].dtype)
                cleaned_df[col] = pd.to_numeric(
                    cleaned_df[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
                    errors="coerce",
                )
                log.append(f"Auto-coerced column '{col}' from {orig_type} to Numeric (float/int).")
                continue

            # Check if datetime (only if strings contain numeric date separators)
            has_date_pattern = sample[:50].str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{2}", regex=True).any()
            if has_date_pattern:
                try:
                    date_converted = pd.to_datetime(sample[:50], errors="coerce", format="mixed")
                    if date_converted.notnull().sum() / len(sample[:50]) > 0.8:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce", format="mixed")
                        log.append(f"Auto-converted column '{col}' from Object to Datetime.")
                except Exception:
                    pass
        else:
            # Strip trailing/leading whitespace for pure text
            if cleaned_df[col].dtype == "object":
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    return cleaned_df, log


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    custom_constant: Optional[Any] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Handles missing values based on user strategy:
    numeric_strategy: 'median', 'mean', 'mode', 'constant', 'drop', 'ffill'
    categorical_strategy: 'mode', 'constant', 'drop', 'ffill'
    """
    cleaned_df = df.copy()
    log = []

    for col in cleaned_df.columns:
        missing_count = int(cleaned_df[col].isnull().sum())
        if missing_count == 0:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(cleaned_df[col])

        if is_numeric:
            if numeric_strategy == "median":
                fill_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log.append(f"Imputed {missing_count} missing value(s) in numeric column '{col}' with Median ({round(fill_val, 2)}).")
            elif numeric_strategy == "mean":
                fill_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log.append(f"Imputed {missing_count} missing value(s) in numeric column '{col}' with Mean ({round(fill_val, 2)}).")
            elif numeric_strategy == "mode":
                fill_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log.append(f"Imputed {missing_count} missing value(s) in '{col}' with Mode ({fill_val}).")
            elif numeric_strategy == "constant":
                val = custom_constant if custom_constant is not None else 0
                cleaned_df[col] = cleaned_df[col].fillna(val)
                log.append(f"Filled {missing_count} missing value(s) in '{col}' with Constant ({val}).")
            elif numeric_strategy == "ffill":
                cleaned_df[col] = cleaned_df[col].ffill().bfill()
                log.append(f"Applied Forward/Backward fill for {missing_count} missing values in '{col}'.")
            elif numeric_strategy == "drop":
                cleaned_df = cleaned_df.dropna(subset=[col])
                log.append(f"Dropped {missing_count} row(s) containing missing values in '{col}'.")

        else:
            if categorical_strategy == "mode":
                mode_series = cleaned_df[col].dropna().mode()
                fill_val = mode_series[0] if not mode_series.empty else "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                log.append(f"Imputed {missing_count} missing categorical value(s) in '{col}' with Mode ('{fill_val}').")
            elif categorical_strategy == "constant":
                val = str(custom_constant) if custom_constant is not None else "Missing"
                cleaned_df[col] = cleaned_df[col].fillna(val)
                log.append(f"Filled {missing_count} missing values in '{col}' with Constant ('{val}').")
            elif categorical_strategy == "ffill":
                cleaned_df[col] = cleaned_df[col].ffill().bfill()
                log.append(f"Applied Forward/Backward fill for {missing_count} categorical missing values in '{col}'.")
            elif categorical_strategy == "drop":
                cleaned_df = cleaned_df.dropna(subset=[col])
                log.append(f"Dropped {missing_count} row(s) containing missing values in '{col}'.")

    return cleaned_df, log


def handle_outliers_iqr(
    df: pd.DataFrame,
    method: str = "clip",  # 'clip', 'drop', 'ignore'
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detects and handles numeric outliers using the Interquartile Range (IQR) method:
    Lower Bound = Q1 - 1.5 * IQR
    Upper Bound = Q3 + 1.5 * IQR
    """
    if method == "ignore":
        return df, ["Outlier treatment skipped by user."]

    cleaned_df = df.copy()
    log = []
    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        # Skip binary or low-cardinality indicators
        if cleaned_df[col].nunique() <= 2:
            continue

        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outliers = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
        outlier_count = int(outliers.sum())

        if outlier_count > 0:
            if method == "clip":
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                log.append(
                    f"Capped (Winsorized) {outlier_count} outlier(s) in '{col}' to range [{round(lower_bound, 2)}, {round(upper_bound, 2)}]."
                )
            elif method == "drop":
                cleaned_df = cleaned_df[~outliers]
                log.append(
                    f"Removed {outlier_count} outlier row(s) from '{col}' outside [{round(lower_bound, 2)}, {round(upper_bound, 2)}]."
                )

    return cleaned_df, log


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Identifies and drops exact duplicate rows from DataFrame."""
    duplicate_count = int(df.duplicated().sum())
    log = []
    if duplicate_count > 0:
        cleaned_df = df.drop_duplicates()
        log.append(f"Detected and purged {duplicate_count} exact duplicate row(s).")
    else:
        cleaned_df = df.copy()
        log.append("Duplicate check: 0 duplicate rows detected.")
    return cleaned_df, log


def execute_full_cleaning_pipeline(
    df: pd.DataFrame,
    standardize_headers: bool = True,
    coerce_types: bool = True,
    numeric_impute_strategy: str = "median",
    cat_impute_strategy: str = "mode",
    outlier_method: str = "clip",
    drop_dupes: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Executes the comprehensive autonomous cleaning pipeline step-by-step.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_log = [f"=== Autonomous Cleaning Pipeline Executed at {timestamp} ==="]
    cleaned = df.copy()

    # Step 1: Standardize column names
    if standardize_headers:
        cleaned, log1 = clean_column_names(cleaned)
        full_log.extend(log1)

    # Step 2: Auto coerce types
    if coerce_types:
        cleaned, log2 = auto_coerce_types(cleaned)
        full_log.extend(log2)

    # Step 3: Remove duplicate rows
    if drop_dupes:
        cleaned, log3 = remove_duplicates(cleaned)
        full_log.extend(log3)

    # Step 4: Missing values
    cleaned, log4 = handle_missing_values(
        cleaned,
        numeric_strategy=numeric_impute_strategy,
        categorical_strategy=cat_impute_strategy,
    )
    full_log.extend(log4)

    # Step 5: Outliers
    if outlier_method != "ignore":
        cleaned, log5 = handle_outliers_iqr(cleaned, method=outlier_method)
        full_log.extend(log5)

    full_log.append(
        f"Pipeline Complete: Final Shape is {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns."
    )
    return cleaned, full_log


def get_downloadable_bytes(df: pd.DataFrame, format: str = "csv") -> bytes:
    """Converts cleaned DataFrame into downloadable CSV or Excel bytes."""
    if format == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif format == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
        return output.getvalue()
    return b""
