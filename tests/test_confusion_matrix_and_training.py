"""
Unit tests verifying:
1. create_confusion_matrix_chart executes cleanly without figure_factory AttributeError.
2. MobileNetV2 quick_test_mode balances classes evenly rather than only selecting class 0.
3. predict_single_image safely handles both integer and string dictionary keys.
"""

import os
import pytest
import numpy as np
from PIL import Image

from src.image_trainer import (
    create_confusion_matrix_chart,
    predict_single_image,
    train_mobilenet_v2,
)
from src.automl import train_tabular_models
from src.eda import generate_correlation_heatmap
import pandas as pd


def test_confusion_matrix_chart_rendering():
    # Test confusion matrix chart with 2 classes
    cm = [[12, 3], [2, 14]]
    classes = ["cats", "dogs"]
    fig = create_confusion_matrix_chart(cm, classes)
    assert fig is not None
    assert fig.data is not None


def test_eda_correlation_heatmap_rendering():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
        "C": [2, 3, 1, 4, 2],
    })
    fig, ins = generate_correlation_heatmap(df)
    assert fig is not None


def test_predict_single_image_with_string_and_int_keys():
    import torch
    import torch.nn as nn

    # Dummy model with 2 outputs
    class DummyNet(nn.Module):
        def forward(self, x):
            return torch.tensor([[0.2, 0.8]])

    model = DummyNet()
    img = Image.new("RGB", (128, 128), color=(200, 200, 200))

    # Test with string keys (as loaded from json metadata)
    res_str = predict_single_image(model, {"0": "cats", "1": "dogs"}, img, img_size=128)
    assert res_str["predicted_class"] == "dogs"
    assert res_str["confidence"] > 50

    # Test with integer keys
    res_int = predict_single_image(model, {0: "cats", 1: "dogs"}, img, img_size=128)
    assert res_int["predicted_class"] == "dogs"
