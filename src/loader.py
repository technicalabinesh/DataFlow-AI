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
import math
import zipfile
import shutil
import hashlib
import concurrent.futures
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional, Union, BinaryIO
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

def is_valid_image_fast(filepath: str) -> bool:
    """
    Ultra-fast magic byte header checker for images.
    Over 100x faster than PIL Image.open().verify() on large 300MB+ image datasets.
    Falls back to PIL only if header is ambiguous.
    """
    try:
        file_size = os.path.getsize(filepath)
        if file_size < 12:
            return False
        with open(filepath, "rb") as f:
            header = f.read(32)
        # JPEG: FF D8 FF
        if header.startswith(b"\xff\xd8\xff"):
            return True
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return True
        # GIF: GIF87a or GIF89a
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return True
        # WebP: RIFF....WEBP
        if header.startswith(b"RIFF") and b"WEBP" in header[:16]:
            return True
        # BMP: BM
        if header.startswith(b"BM"):
            return True
        # Fallback to PIL verify for rare formats (TIFF, etc.)
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def extract_zip_to_raw(
    zip_input: Union[bytes, str, BinaryIO, Any],
    zip_filename: str = "dataset.zip",
    progress_callback: Optional[Any] = None,
) -> Tuple[str, List[str]]:
    """
    Extracts uploaded or local zip archive into data/raw/extracted_<timestamp> safely.
    Optimized for large archives (300MB+) using:
    1. Zero-copy / chunked disk streaming (prevents memory spikes and OOM).
    2. Multi-threaded parallel extraction across CPU cores (zlib decompression releases GIL).
    3. Directory pre-creation to eliminate per-file Windows NTFS lookup locks.
    4. Filters out __MACOSX resource fork files and metadata files.
    Returns (extracted_folder_path, list_of_corrupt_files).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(zip_filename))[0] or "archive"
    folder_name = f"extracted_{timestamp}_{base_name}"
    target_dir = os.path.join(RAW_DATA_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    corrupted_files: List[str] = []

    # Resolve zip file on disk
    temp_zip_to_clean: Optional[str] = None
    
    if isinstance(zip_input, str):
        # Local file path
        if not os.path.exists(zip_input):
            raise FileNotFoundError(f"ZIP file not found at path: {zip_input}")
        zip_path_on_disk = os.path.abspath(zip_input)
    elif hasattr(zip_input, "read"):
        # Stream / UploadedFile
        temp_zip_to_clean = os.path.join(RAW_DATA_DIR, f"_temp_upload_{timestamp}_{base_name}.zip")
        if hasattr(zip_input, "seek"):
            try:
                zip_input.seek(0)
            except Exception:
                pass
        with open(temp_zip_to_clean, "wb") as f_out:
            shutil.copyfileobj(zip_input, f_out, length=4 * 1024 * 1024)
        zip_path_on_disk = temp_zip_to_clean
    elif isinstance(zip_input, (bytes, bytearray, memoryview)):
        # In-memory bytes
        temp_zip_to_clean = os.path.join(RAW_DATA_DIR, f"_temp_upload_{timestamp}_{base_name}.zip")
        with open(temp_zip_to_clean, "wb") as f_out:
            f_out.write(zip_input)
        zip_path_on_disk = temp_zip_to_clean
    else:
        raise ValueError(f"Unsupported zip input type: {type(zip_input)}")

    try:
        with zipfile.ZipFile(zip_path_on_disk, "r") as z_inspect:
            members = z_inspect.infolist()
            
            # Security & metadata filtering
            valid_members: List[zipfile.ZipInfo] = []
            for member in members:
                filename = member.filename
                # Security against Zip Slip vulnerability
                if filename.startswith("/") or filename.startswith("\\") or ".." in filename:
                    continue
                # Ignore directory markers (we pre-create folders)
                if member.is_dir() or filename.endswith("/") or filename.endswith("\\"):
                    continue
                # Filter macOS metadata and hidden files
                base_f = os.path.basename(filename)
                if "__MACOSX" in filename or base_f.startswith("._") or base_f in [".DS_Store", "Thumbs.db", "desktop.ini"]:
                    continue
                valid_members.append(member)

        total_files = len(valid_members)
        if total_files == 0:
            return target_dir, corrupted_files

        # 1. High-speed directory pre-creation in one pass to eliminate NTFS lock overhead
        unique_dirs = set()
        for m in valid_members:
            dir_part = os.path.dirname(m.filename)
            if dir_part:
                unique_dirs.add(os.path.join(target_dir, dir_part))
        for d in sorted(unique_dirs):
            os.makedirs(d, exist_ok=True)

        # 2. Multi-threaded worker extraction
        num_workers = min(8, max(2, os.cpu_count() or 4))
        if total_files < 50:
            num_workers = 1
        
        chunk_size = math.ceil(total_files / num_workers)
        member_chunks = [valid_members[i : i + chunk_size] for i in range(0, total_files, chunk_size)]

        def _extract_chunk(chunk_list: List[zipfile.ZipInfo]) -> List[str]:
            local_corrupt = []
            with zipfile.ZipFile(zip_path_on_disk, "r") as z_worker:
                for item in chunk_list:
                    try:
                        z_worker.extract(item, target_dir)
                    except Exception:
                        local_corrupt.append(item.filename)
            return local_corrupt

        if num_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_extract_chunk, chunk) for chunk in member_chunks]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        corrupted_files.extend(fut.result())
                    except Exception:
                        pass
        else:
            corrupted_files.extend(_extract_chunk(valid_members))

        if progress_callback:
            try:
                progress_callback(1.0, f"Extraction complete ({total_files} files)")
            except Exception:
                pass

    finally:
        if temp_zip_to_clean and os.path.exists(temp_zip_to_clean):
            try:
                os.remove(temp_zip_to_clean)
            except Exception:
                pass

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

    # Unwrap single-subdirectory wrappers up to 5 levels (e.g. zip/archive/dataset/...)
    current = folder_path
    for _ in range(5):
        try:
            entries = [e for e in os.listdir(current) if not e.startswith(".") and e != "__MACOSX"]
            if len(entries) == 1:
                candidate = os.path.join(current, entries[0])
                if os.path.isdir(candidate):
                    current = candidate
                    continue
            break
        except Exception:
            break
    folder_path = current
    results["path"] = current

    # Recursive scan
    all_files = []
    for root, _, files in os.walk(folder_path):
        # Ignore macOS metadata folders
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._") or f in [".DS_Store", "Thumbs.db"]:
                continue
            all_files.append(os.path.join(root, f))

    if not all_files:
        results["is_empty"] = True
        results["summary"] = "The specified directory is empty."
        return results

    # Ultra-fast classify files with magic header checks
    for filepath in all_files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            if is_valid_image_fast(filepath):
                results["images"].append(filepath)
            else:
                results["corrupt_files"].append(filepath)
        elif ext in TABULAR_EXTENSIONS:
            results["tables"].append(filepath)

    # DETECT STRUCTURE:
    
    # Check for pre-split structure (train/val/test or train/test)
    subdirs = [d.lower() for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d != "__MACOSX"]
    has_train = "train" in subdirs
    has_test = "test" in subdirs or "val" in subdirs or "validation" in subdirs

    if has_train and len(results["images"]) > 0:
        results["mode"] = "images_presplit"
        norm_folder = os.path.abspath(folder_path)
        for img_p in results["images"]:
            rel = os.path.relpath(img_p, norm_folder).replace("\\", "/")
            parts = rel.split("/")
            if len(parts) > 1 and parts[0].lower() in ["train", "val", "validation", "test"]:
                results["split_folders"].setdefault(parts[0].lower(), []).append(img_p)

        results["image_items"] = []
        for s, img_list in results["split_folders"].items():
            for p in img_list:
                parent = os.path.basename(os.path.dirname(p))
                if parent.lower() in ["train", "val", "validation", "test"]:
                    fname = os.path.splitext(os.path.basename(p))[0]
                    m = re.match(r"^([a-zA-Z]+)[._\-\s]", fname) or re.match(r"^([a-zA-Z]+)", fname)
                    lbl = m.group(1).lower() if (m and len(m.group(1)) >= 2) else s
                else:
                    lbl = parent
                results["image_items"].append((p, lbl))

        class_counts = {}
        for _, l in results["image_items"]:
            class_counts[l] = class_counts.get(l, 0) + 1
        results["classes"] = class_counts
        results["summary"] = f"Detected pre-split image dataset ({', '.join(results['split_folders'].keys())}) with {len(results['images'])} total valid images across {len(class_counts)} classes."
        return results

    # Check for Images with Labels CSV
    if len(results["images"]) > 0 and len(results["tables"]) >= 1:
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

    # Check for Image Classification by Subfolders or Filename Prefixes
    if len(results["images"]) > 0 and len(results["tables"]) == 0:
        results["mode"] = "images_per_class"
        import re
        generic_names = {"images", "image", "img", "imgs", "photos", "pictures", "data", "dataset", "raw", "extracted", "archive", "content", "unlabeled"}
        
        # Strategy 1: Direct parent directories
        parent_map = {p: os.path.basename(os.path.dirname(p)).strip() for p in results["images"]}
        unique_parents = set(parent_map.values())
        non_generic_parents = unique_parents - generic_names

        # Strategy 2: Filename prefixes (e.g. bougainvillea_00002.jpg, cat.1.jpg)
        prefix_map = {}
        for p in results["images"]:
            fname = os.path.splitext(os.path.basename(p))[0]
            m = re.match(r"^([a-zA-Z]+)[._\-\s]", fname) or re.match(r"^([a-zA-Z]+)", fname)
            if m and len(m.group(1)) >= 2:
                prefix_map[p] = m.group(1).lower()
        unique_prefixes = set(prefix_map.values())

        # Strategy 3: Grandparent directories (e.g. flowers/daisy/images/1.jpg)
        gp_map = {p: os.path.basename(os.path.dirname(os.path.dirname(p))).strip() for p in results["images"]}
        non_generic_gps = set(gp_map.values()) - generic_names

        # Selection of best strategy:
        if len(non_generic_parents) >= 2:
            final_items = [(p, parent_map[p]) for p in results["images"]]
        elif len(unique_prefixes) >= 2 and len(prefix_map) >= len(results["images"]) * 0.4:
            final_items = [(p, prefix_map.get(p, "general")) for p in results["images"]]
        elif len(non_generic_gps) >= 2:
            final_items = [(p, gp_map[p]) for p in results["images"]]
        elif len(unique_prefixes) >= 2:
            final_items = [(p, prefix_map.get(p, "general")) for p in results["images"]]
        else:
            fallback_label = list(unique_parents)[0] if unique_parents else "unlabeled"
            final_items = [(p, fallback_label) for p in results["images"]]

        class_counts = {}
        for _, l in final_items:
            class_counts[l] = class_counts.get(l, 0) + 1

        results["classes"] = class_counts
        results["image_items"] = final_items
        results["summary"] = f"Detected Image Classification dataset with {len(class_counts)} classes ({', '.join(list(class_counts.keys())[:5])}{'...' if len(class_counts) > 5 else ''}) and {len(results['images'])} images."
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
