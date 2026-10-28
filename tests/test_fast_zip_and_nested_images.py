"""
Tests for high-speed multi-core zip extraction, instant image validation,
and nested image folder class detection.
"""

import os
import io
import zipfile
import pytest
from PIL import Image

from src.loader import (
    extract_zip_to_raw,
    is_valid_image_fast,
    scan_dataset_directory,
    RAW_DATA_DIR,
)


def create_dummy_image_bytes(fmt="JPEG", size=(32, 32)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_is_valid_image_fast():
    jpeg_bytes = create_dummy_image_bytes("JPEG")
    png_bytes = create_dummy_image_bytes("PNG")
    
    # Save temporary test files
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    jp_path = os.path.join(RAW_DATA_DIR, "_test_fast_valid.jpg")
    png_path = os.path.join(RAW_DATA_DIR, "_test_fast_valid.png")
    corrupt_path = os.path.join(RAW_DATA_DIR, "_test_corrupt.jpg")
    
    with open(jp_path, "wb") as f:
        f.write(jpeg_bytes)
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_HEADER_AT_ALL_CORRUPTED")
        
    try:
        assert is_valid_image_fast(jp_path) is True
        assert is_valid_image_fast(png_path) is True
        assert is_valid_image_fast(corrupt_path) is False
    finally:
        for p in [jp_path, png_path, corrupt_path]:
            if os.path.exists(p):
                os.remove(p)


def test_extract_zip_with_nested_image_folders():
    # Build a zip archive with nested structure:
    # archive.zip:
    #   my_project/
    #     images/
    #       cats/
    #         cat_01.jpg
    #         cat_02.jpg
    #       dogs/
    #         dog_01.jpg
    #         dog_02.jpg
    jpeg_bytes = create_dummy_image_bytes("JPEG")
    
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("my_project/images/cats/cat_01.jpg", jpeg_bytes)
        zf.writestr("my_project/images/cats/cat_02.jpg", jpeg_bytes)
        zf.writestr("my_project/images/dogs/dog_01.jpg", jpeg_bytes)
        zf.writestr("my_project/images/dogs/dog_02.jpg", jpeg_bytes)
        
    zip_bytes = zip_buf.getvalue()
    
    # Extract using the high-speed multi-core unzipper
    extracted_dir, corrupt = extract_zip_to_raw(zip_bytes, "test_nested_images.zip")
    
    assert os.path.exists(extracted_dir)
    assert len(corrupt) == 0
    
    # Scan the extracted directory
    res = scan_dataset_directory(extracted_dir)
    assert res["exists"] is True
    assert len(res["images"]) == 4
    assert "cats" in res["classes"]
    assert "dogs" in res["classes"]
    assert res["classes"]["cats"] == 2
    assert res["classes"]["dogs"] == 2


def test_extract_zip_from_file_path_directly():
    jpeg_bytes = create_dummy_image_bytes("JPEG")
    temp_zip_file = os.path.join(RAW_DATA_DIR, "_test_local_disk_upload.zip")
    
    with zipfile.ZipFile(temp_zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cats/cat_10.jpg", jpeg_bytes)
        zf.writestr("dogs/dog_10.jpg", jpeg_bytes)
        
    try:
        # Pass file path string directly (0 seconds upload wait)
        extracted_dir, corrupt = extract_zip_to_raw(temp_zip_file, "test_local.zip")
        assert os.path.exists(extracted_dir)
        res = scan_dataset_directory(extracted_dir)
        assert len(res["images"]) == 2
        assert "cats" in res["classes"]
        assert "dogs" in res["classes"]
    finally:
        if os.path.exists(temp_zip_file):
            os.remove(temp_zip_file)


def test_flat_folder_flower_prefixes():
    # Simulate flowers dataset: all in single folder 'flowers', named with prefixes
    jpeg_bytes = create_dummy_image_bytes("JPEG")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("flowers/bougainvillea_001.jpg", jpeg_bytes)
        zf.writestr("flowers/bougainvillea_002.jpg", jpeg_bytes)
        zf.writestr("flowers/daisies_001.jpg", jpeg_bytes)
        zf.writestr("flowers/daisies_002.jpg", jpeg_bytes)
        zf.writestr("flowers/hibiscus_001.jpg", jpeg_bytes)

    extracted_dir, _ = extract_zip_to_raw(zip_buf.getvalue(), "flowers.zip")
    res = scan_dataset_directory(extracted_dir)
    assert res["exists"] is True
    assert len(res["classes"]) == 3
    assert "bougainvillea" in res["classes"]
    assert "daisies" in res["classes"]
    assert "hibiscus" in res["classes"]
    assert len(res["image_items"]) == 5
    # Verify exact item mapping
    item_labels = {item[1] for item in res["image_items"]}
    assert item_labels == {"bougainvillea", "daisies", "hibiscus"}


def test_single_class_auto_partition():
    from src.image_trainer import train_mobilenet_v2
    # Single class dataset with 10 identical label items
    p1 = os.path.join(RAW_DATA_DIR, "_dummy_flower1.jpg")
    p2 = os.path.join(RAW_DATA_DIR, "_dummy_flower2.jpg")
    with open(p1, "wb") as f:
        f.write(create_dummy_image_bytes("JPEG", size=(64, 64)))
    with open(p2, "wb") as f:
        f.write(create_dummy_image_bytes("JPEG", size=(64, 64)))

    try:
        tr = [(p1, "flower"), (p2, "flower")] * 5
        va = [(p1, "flower"), (p2, "flower")] * 2
        te = [(p1, "flower"), (p2, "flower")] * 2
        res = train_mobilenet_v2(tr, va, te, epochs=1, quick_test_mode=True)
        assert res is not None
        assert len(res["metadata"]["classes"]) >= 2
    finally:
        for p in [p1, p2]:
            if os.path.exists(p):
                os.remove(p)

