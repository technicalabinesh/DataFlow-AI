"""
=============================================================================
Universal Data & Folder Ingestion Engine (src/loader.py)
=============================================================================
Supports:
1. Single files: CSV, TSV, XLSX, XLS, JSON, Parquet
   - Auto-detection of encoding (utf-8, latin-1, cp1252, etc.)
   - Auto-detection of delimiters (,, \t, ;, |)
   - Multi-sheet Excel parsing with interactive sheet selector
   - Large file threshold alerts & optional downsampling
   - Safe raw file preservation in data/raw/
2. Folder & ZIP inputs:
   - Recursive scanning & pattern detection
   - Case 1: Tabular files (auto-append same columns, join suggestion, or pick file)
   - Case 2: Image classification folders (subfolder per class)
   - Case 3: Pre-split folders (train/val/test)
   - Case 4: Image folder + Labels CSV
   - Case 5: Mixed content resolution
   - Corrupt file detection & skip logging
"""

import os
import io
import csv
import zipfile
import shutil
import hashlib
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
from PIL import Image

RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}


def save_raw_file(file_bytes: bytes, original_name: str) -> str:
    """
    Saves an uploaded file untouched into data/raw/ with a timestamp to prevent overwriting.
    Returns the absolute path to the saved file.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(c for c in original_name if c.isalnum() or c in (".", "_", "-"))
    saved_filename = f"{timestamp}_{clean_name}"
    target_path = os.path.join(RAW_DATA_DIR, saved_filename)
    
    with open(target_path, "wb") as f:
        f.write(file_bytes)
        
    return target_path


def detect_encoding_and_delimiter(file_bytes: bytes) -> Tuple[str, str]:
    """
    Autodetects the text encoding and CSV delimiter using a sample of the bytes.
    Tries standard encodings in order: utf-8, latin-1, cp1252, iso-8859-1.
    """
    sample_bytes = file_bytes[:65536]  # first 64 KB for quick sniffing
    detected_encoding = "utf-8"
    
    for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            text_sample = sample_bytes.decode(enc)
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        text_sample = sample_bytes.decode("utf-8", errors="ignore")
        detected_encoding = "utf-8"

    # Detect delimiter using csv.Sniffer
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text_sample)
        detected_delimiter = dialect.delimiter
    except Exception:
        # Fallback to frequency count among common delimiters
        common_delims = [",", "\t", ";", "|"]
        counts = {d: text_sample.count(d) for d in common_delims}
        detected_delimiter = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

    return detected_encoding, detected_delimiter


def get_excel_sheets(file_bytes: bytes, extension: str) -> List[str]:
    """Returns the list of sheet names present in an Excel workbook."""
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
        return excel_file.sheet_names
    except Exception as e:
        return [f"Default (Error reading sheets: {str(e)})"]


def load_tabular_from_bytes(
    file_bytes: bytes,
    filename: str,
    selected_sheet: Optional[str] = None,
    max_sample_rows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Universal reader for any single tabular file from raw bytes.
    Returns (DataFrame, metadata_dict).
    """
    meta = {
        "filename": filename,
        "extension": os.path.splitext(filename)[1].lower(),
        "filesize_bytes": len(file_bytes),
        "encoding": None,
        "delimiter": None,
        "sheets": [],
        "warnings": [],
        "error": None,
    }

    if len(file_bytes) == 0:
        meta["error"] = "Uploaded file is completely empty (0 bytes)."
        return None, meta

    ext = meta["extension"]

    try:
        # 1. CSV / TSV / TXT
        if ext in [".csv", ".tsv", ".txt"]:
            enc, delim = detect_encoding_and_delimiter(file_bytes)
            meta["encoding"] = enc
            meta["delimiter"] = delim
            
            # Read with pandas
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=enc,
                sep=delim,
                nrows=max_sample_rows,
                on_bad_lines="skip",
            )
            
        # 2. Excel (.xlsx, .xls)
        elif ext in [".xlsx", ".xls"]:
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            sheets = get_excel_sheets(file_bytes, ext)
            meta["sheets"] = sheets
            sheet_to_read = selected_sheet if (selected_sheet and selected_sheet in sheets) else 0
            df = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=sheet_to_read,
                engine=engine,
                nrows=max_sample_rows,
            )

        # 3. JSON
        elif ext == ".json":
            enc, _ = detect_encoding_and_delimiter(file_bytes)
            meta["encoding"] = enc
            df = pd.read_json(io.BytesIO(file_bytes), encoding=enc)
            if max_sample_rows and len(df) > max_sample_rows:
                df = df.iloc[:max_sample_rows]

        # 4. Parquet
        elif ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(file_bytes))
            if max_sample_rows and len(df) > max_sample_rows:
                df = df.iloc[:max_sample_rows]

        else:
            meta["error"] = f"Unsupported file extension '{ext}'. Expected CSV, TSV, XLSX, XLS, JSON, or Parquet."
            return None, meta

        # Empty dataframe check
        if df is None or df.empty:
            meta["error"] = "File loaded but contains no rows or data."
            return None, meta

        return df, meta

    except Exception as e:
        meta["error"] = f"Failed to parse file: {str(e)}"
        return None, meta


def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes universal dataset summary metrics:
    shape, dtypes count, missing percentage, duplicates, memory footprint.
    """
    if df is None or df.empty:
        return {}

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / len(df) * 100), 2) if len(df) > 0 else 0.0

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    datetime_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "datetime_count": len(datetime_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "memory_mb": memory_mb,
    }


# =============================================================================
# FOLDER & ZIP RECURSIVE SCANNER ENGINE
# =============================================================================

def extract_zip_to_raw(zip_bytes: bytes, zip_filename: str) -> Tuple[str, List[str]]:
    """
    Extracts uploaded zip archive into data/raw/extracted_<timestamp> safely.
    Returns (extracted_folder_path, list_of_corrupt_files).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"extracted_{timestamp}_{os.path.splitext(zip_filename)[0]}"
    target_dir = os.path.join(RAW_DATA_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    corrupted_files = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for member in z.infolist():
            # Security against Zip Slip vulnerability
            filename = member.filename
            if filename.startswith("/") or ".." in filename:
                continue
            try:
                z.extract(member, target_dir)
            except Exception:
                corrupted_files.append(filename)

    return target_dir, corrupted_files


def scan_dataset_directory(folder_path: str) -> Dict[str, Any]:
    """
    Recursively inspects folder_path and determines:
    1. Structure type (Images Per Class, Pre-split Image Train/Val/Test, Images + Labels CSV, Tabular Folder, Mixed)
    2. File counts, valid files, corrupted files
    3. Class labels and sample counts per class
    4. Tabular schema comparability (can append? can join?)
    """
    results = {
        "path": folder_path,
        "exists": False,
        "is_empty": False,
        "mode": None,  # 'images_per_class', 'images_presplit', 'images_with_csv', 'tables_mergeable', 'tables_single', 'mixed'
        "images": [],
        "tables": [],
        "corrupt_files": [],
        "classes": {},
        "split_folders": {},
        "summary": "",
        "table_columns_map": {},
        "candidate_key_cols": [],
        "can_append_tables": False,
    }

    if not os.path.exists(folder_path):
        results["summary"] = f"Directory not found: '{folder_path}'"
        return results

    results["exists"] = True

    # Recursive scan
    all_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            all_files.append(os.path.join(root, f))

    if not all_files:
        results["is_empty"] = True
        results["summary"] = "The specified directory is empty."
        return results

    # Classify files
    for filepath in all_files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            # Check if image is valid or corrupted
            try:
                with Image.open(filepath) as img:
                    img.verify()
                results["images"].append(filepath)
            except Exception:
                results["corrupt_files"].append(filepath)
        elif ext in TABULAR_EXTENSIONS:
            results["tables"].append(filepath)

    # DETECT STRUCTURE:
    
    # Check for pre-split structure (train/val/test or train/test)
    subdirs = [d.lower() for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    has_train = "train" in subdirs
    has_test = "test" in subdirs or "val" in subdirs or "validation" in subdirs

    if has_train and len(results["images"]) > 0:
        results["mode"] = "images_presplit"
        # Collect splits
        for s in ["train", "val", "validation", "test"]:
            split_dir = os.path.join(folder_path, s)
            if os.path.isdir(split_dir):
                s_images = []
                for r, _, fs in os.walk(split_dir):
                    for f in fs:
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                            s_images.append(os.path.join(r, f))
                results["split_folders"][s] = s_images
        results["summary"] = f"Detected pre-split image dataset ({', '.join(results['split_folders'].keys())}) with {len(results['images'])} total valid images."
        return results

    # Check for Images with Labels CSV
    if len(results["images"]) > 0 and len(results["tables"]) >= 1:
        # Check if one of the tables has a column matching image filenames
        csv_path = results["tables"][0]
        try:
            sample_df = pd.read_csv(csv_path, nrows=10)
            img_basenames = {os.path.basename(p) for p in results["images"][:20]}
            matches = False
            for col in sample_df.columns:
                if any(str(v) in img_basenames or (str(v) + ".jpg") in img_basenames for v in sample_df[col].dropna()):
                    matches = True
                    break
            if matches:
                results["mode"] = "images_with_csv"
                results["summary"] = f"Detected {len(results['images'])} images with matching labels table '{os.path.basename(csv_path)}'."
                return results
        except Exception:
            pass

    # Check for Image Classification by Subfolders (Class Folders)
    if len(results["images"]) > 0 and len(results["tables"]) == 0:
        results["mode"] = "images_per_class"
        # Group by direct parent directory name
        class_counts = {}
        for img_path in results["images"]:
            parent = os.path.basename(os.path.dirname(img_path))
            # Don't use root folder name if all images are directly in root
            if os.path.abspath(os.path.dirname(img_path)) == os.path.abspath(folder_path):
                parent = "unlabeled"
            class_counts[parent] = class_counts.get(parent, 0) + 1

        results["classes"] = class_counts
        results["summary"] = f"Detected Image Classification folder with {len(class_counts)} classes ({', '.join(list(class_counts.keys())[:5])}{'...' if len(class_counts) > 5 else ''}) and {len(results['images'])} images."
        return results

    # Check for Tabular Folder
    if len(results["tables"]) > 0 and len(results["images"]) == 0:
        if len(results["tables"]) == 1:
            results["mode"] = "tables_single"
            results["summary"] = f"Detected 1 tabular dataset: '{os.path.basename(results['tables'][0])}'."
        else:
            # Check schemas of tables
            col_sets = []
            for t_path in results["tables"][:10]:
                try:
                    ext = os.path.splitext(t_path)[1].lower()
                    if ext in [".csv", ".tsv"]:
                        tdf = pd.read_csv(t_path, nrows=5)
                    elif ext in [".xlsx", ".xls"]:
                        tdf = pd.read_excel(t_path, nrows=5)
                    elif ext == ".parquet":
                        tdf = pd.read_parquet(t_path)
                    else:
                        continue
                    cols = list(tdf.columns)
                    results["table_columns_map"][os.path.basename(t_path)] = cols
                    col_sets.append(set(cols))
                except Exception:
                    continue

            # Compare schemas
            if col_sets and all(cs == col_sets[0] for cs in col_sets):
                results["can_append_tables"] = True
                results["mode"] = "tables_mergeable"
                results["summary"] = f"Detected {len(results['tables'])} table files with IDENTICAL columns ({len(col_sets[0])} columns). Ready to auto-append (concatenate)."
            elif col_sets:
                # Find common intersection columns (possible join keys)
                intersection = set.intersection(*col_sets)
                results["candidate_key_cols"] = list(intersection)
                results["mode"] = "tables_mergeable"
                results["summary"] = f"Detected {len(results['tables'])} table files with overlapping keys: {list(intersection)[:4]}."
            else:
                results["mode"] = "tables_single"
                results["summary"] = f"Detected {len(results['tables'])} distinct table files."

        return results

    # Mixed or General
    results["mode"] = "mixed"
    results["summary"] = f"Mixed content detected: {len(results['tables'])} table(s), {len(results['images'])} image(s)."
    return results


def load_and_append_tables(table_paths: List[str]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Appends multiple tabular files into a unified dataframe.
    Returns (combined_df, list_of_errors).
    """
    dfs = []
    errors = []
    for p in table_paths:
        try:
            ext = os.path.splitext(p)[1].lower()
            if ext in [".csv", ".tsv"]:
                enc, delim = detect_encoding_and_delimiter(open(p, "rb").read(10000))
                df = pd.read_csv(p, encoding=enc, sep=delim)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(p)
            elif ext == ".parquet":
                df = pd.read_parquet(p)
            elif ext == ".json":
                df = pd.read_json(p)
            else:
                continue
            df["_source_file"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            errors.append(f"{os.path.basename(p)}: {str(e)}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        return combined, errors
    return None, errors
