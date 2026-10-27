"""
=============================================================================
Computer Vision Engine: MobileNetV2 Transfer Learning (src/image_trainer.py)
=============================================================================
Features:
1. Auto GPU / CPU detection with CPU performance warnings.
2. Transfer learning backbone freezing & custom linear classifier head.
3. Data augmentation (rotations, flips, color jitter) & standard ImageNet normalization.
4. Live training & validation curves (loss & accuracy) with early stopping.
5. Strict single evaluation on hold-out test set:
   - Precision, Recall, F1-score (weighted), Accuracy, Confusion Matrix.
   - Identification of top misclassified sample images.
6. Serialization of trained weights (.pt) and class metadata (.json) to models/.
7. Instant inference engine for user-uploaded single images.
"""

import os
import time
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Callable
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import plotly.graph_objects as go
import plotly.figure_factory as ff

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


class UniversalImageDataset(Dataset):
    """PyTorch Dataset loading images from disk with PIL and applying transforms."""
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filepath, label = self.items[idx]
        try:
            with Image.open(filepath) as img:
                img = img.convert("RGB")
        except Exception:
            # Fallback black image if corrupt
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), filepath


def get_image_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training transforms (with augmentation) and validation/test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def train_mobilenet_v2(
    train_data: List[Tuple[str, str]],
    val_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    epochs: int = 5,
    batch_size: int = 16,
    img_size: int = 224,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 3,
    quick_test_mode: bool = False,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """
    Executes MobileNetV2 transfer learning with live metrics and hold-out test evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"

    # Extract unique classes
    all_labels = sorted(list({label for _, label in (train_data + val_data + test_data)}))
    class_to_idx = {label: i for i, label in enumerate(all_labels)}
    idx_to_class = {i: label for label, i in class_to_idx.items()}
    num_classes = len(all_labels)

    # Convert to index-based tuples
    train_items = [(p, class_to_idx[l]) for p, l in train_data]
    val_items = [(p, class_to_idx[l]) for p, l in val_data]
    test_items = [(p, class_to_idx[l]) for p, l in test_data]

    # Quick test mode: cap samples for swift defense demo
    if quick_test_mode:
        train_items = train_items[:min(len(train_items), 40)]
        val_items = val_items[:min(len(val_items), 20)]
        test_items = test_items[:min(len(test_items), 20)]

    train_tf, eval_tf = get_image_transforms(img_size)

    train_loader = DataLoader(
        UniversalImageDataset(train_items, transform=train_tf),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        UniversalImageDataset(val_items, transform=eval_tf),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        UniversalImageDataset(test_items, transform=eval_tf),
        batch_size=batch_size,
        shuffle=False,
    )

    # Build model with pretrained backbone
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(total_train, 1)
        epoch_train_acc = (correct_train / max(total_train, 1)) * 100.0

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / max(total_val, 1)
        epoch_val_acc = (correct_val / max(total_val, 1)) * 100.0

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))

        # Check early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1

        if progress_callback:
            progress_callback(
                epoch,
                epochs,
                {
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                },
            )

        if patience_counter >= early_stopping_patience:
            break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # FINAL EVALUATION ON HOLD-OUT TEST SET
    model.eval()
    y_true = []
    y_pred = []
    y_conf = []
    misclassified = []

    with torch.no_grad():
        for imgs, labels, paths in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                true_lbl = labels[i].item()
                pred_lbl = preds[i].item()
                conf = probs[i][pred_lbl].item()

                y_true.append(true_lbl)
                y_pred.append(pred_lbl)
                y_conf.append(conf)

                if true_lbl != pred_lbl:
                    misclassified.append({
                        "filepath": paths[i],
                        "true_class": idx_to_class[true_lbl],
                        "predicted_class": idx_to_class[pred_lbl],
                        "confidence": round(conf * 100, 2),
                    })

    # Test metrics
    test_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    ) if y_true else (0.0, 0.0, 0.0, None)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist() if y_true else []

    # Save trained model to disk
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"mobilenet_v2_{timestamp}.pt"
    model_save_path = os.path.join(MODELS_DIR, model_filename)
    torch.save(model.state_dict(), model_save_path)

    metadata = {
        "model_type": "MobileNetV2 (Transfer Learning)",
        "timestamp": timestamp,
        "classes": all_labels,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "device": str(device_name),
        "epochs_completed": len(history["train_loss"]),
        "metrics": {
            "test_accuracy": round(float(test_acc) * 100, 2),
            "test_precision": round(float(prec) * 100, 2),
            "test_recall": round(float(rec) * 100, 2),
            "test_f1": round(float(f1) * 100, 2),
        },
        "confusion_matrix": cm,
        "misclassified_samples": misclassified[:10],
    }

    meta_save_path = os.path.join(MODELS_DIR, f"mobilenet_v2_{timestamp}_meta.json")
    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model": model,
        "model_path": model_save_path,
        "meta_path": meta_save_path,
        "history": history,
        "metadata": metadata,
        "gpu_available": gpu_available,
        "device_name": device_name,
    }


def predict_single_image(
    model: nn.Module,
    idx_to_class: Dict[int, str],
    image_input,
    img_size: int = 224,
) -> Dict[str, Any]:
    """
    Performs inference on a single PIL Image or image file bytes.
    Returns predicted class, confidence percentage, and top class probabilities.
    """
    device = next(model.parameters()).device
    model.eval()

    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")

    _, eval_tf = get_image_transforms(img_size)
    tensor = eval_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    pred_class = idx_to_class[top_idx]
    confidence = float(probs[top_idx]) * 100.0

    class_probs = {idx_to_class[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "all_probabilities": class_probs,
    }


def create_training_curves_chart(history: Dict[str, List[float]]) -> go.Figure:
    """Creates dual-axis interactive Plotly training loss and accuracy curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()

    # Loss curves
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss", line=dict(color="#f43f5e", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss", line=dict(color="#fb923c", width=2, dash="dash")))

    # Accuracy curves
    fig.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train Acc (%)", yaxis="y2", line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_acc"], name="Val Acc (%)", yaxis="y2", line=dict(color="#34d399", width=2, dash="dash")))

    fig.update_layout(
        title="Live Training & Validation Convergence Curves",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Cross-Entropy Loss", titlefont=dict(color="#f43f5e")),
        yaxis2=dict(title="Accuracy (%)", titlefont=dict(color="#38bdf8"), overlaying="y", side="right"),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_confusion_matrix_chart(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """Generates an annotated Plotly Confusion Matrix heatmap."""
    z = cm
    x = classes
    y = classes
    z_text = [[str(val) for val in row] for row in z]

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale="Blues")
    fig.update_layout(
        title="Test Set Confusion Matrix (Held-out Evaluation)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        height=380,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig
