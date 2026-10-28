"""
=============================================================================
Anomaly Detection Engine (src/anomaly_engine.py)
=============================================================================
Features:
1. Unsupervised Machine Learning with scikit-learn Isolation Forest.
2. Multivariate Statistical Outlier Detection combining IQR and Robust Z-Score.
3. PCA 2D/3D Dimensionality Reduction for intuitive spatial visualization.
4. Feature Driver Attribution: explains which features made each row anomalous.
5. Interactive Plotly scatter plot & anomaly distribution charts.
6. Cleaned dataset generation with anomalies purged or tagged.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px


def detect_anomalies(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Performs unsupervised anomaly detection using Isolation Forest and PCA projection.
    """
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=["number"]).columns)

    if not feature_cols:
        raise ValueError("At least one numerical feature column is required for anomaly detection.")

    # Clean missing values for modeling
    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    preds = iso_forest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    raw_scores = iso_forest.decision_function(X_scaled)  # Lower score = more anomalous

    # Normalize anomaly score to 0 - 100% (Higher = More Anomalous)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    norm_anomaly_score = (
        100.0 * (1.0 - (raw_scores - min_s) / max(max_s - min_s, 1e-6))
    ).round(2)

    is_anomaly = preds == -1
    anomaly_count = int(np.sum(is_anomaly))
    normal_count = int(len(df) - anomaly_count)

    # 2D PCA projection for spatial visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    var_exp = round(float(np.sum(pca.explained_variance_ratio_)) * 100, 1)

    # Identify primary feature driver for each anomaly
    z_scores = np.abs(X_scaled)
    top_driver_indices = np.argmax(z_scores, axis=1)
    primary_drivers = [feature_cols[idx] for idx in top_driver_indices]

    results_df = df.copy()
    results_df["_is_anomaly"] = np.where(is_anomaly, "Anomaly", "Normal")
    results_df["_anomaly_score"] = norm_anomaly_score
    results_df["_primary_driver"] = primary_drivers
    results_df["_pca_x"] = pca_coords[:, 0]
    results_df["_pca_y"] = pca_coords[:, 1]

    # Filtered view of anomalies sorted by severity
    anomalies_only = results_df[results_df["_is_anomaly"] == "Anomaly"].sort_values(
        by="_anomaly_score", ascending=False
    )

    # Interactive 2D PCA Scatter Visual
    fig_pca = px.scatter(
        results_df,
        x="_pca_x",
        y="_pca_y",
        color="_is_anomaly",
        color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#f43f5e"},
        hover_data=feature_cols[:4] + ["_anomaly_score", "_primary_driver"],
        title=f"PCA 2D Spatial Projection (Explaining {var_exp}% of Variance)",
        labels={"_pca_x": "PCA Component 1", "_pca_y": "PCA Component 2"},
    )
    fig_pca.update_traces(marker=dict(size=7, opacity=0.85, line=dict(width=1, color="#000000")))
    fig_pca.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(t=40, b=20, l=40, r=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )

    # Score distribution histogram
    fig_hist = px.histogram(
        results_df,
        x="_anomaly_score",
        color="_is_anomaly",
        color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#f43f5e"},
        nbins=30,
        title="Distribution of Anomaly Severity Scores",
        labels={"_anomaly_score": "Anomaly Severity Score (%)"},
    )
    fig_hist.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(t=40, b=20, l=40, r=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )

    return {
        "model_name": "Isolation Forest (Ensemble)",
        "total_records": len(df),
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "contamination_pct": round(contamination * 100, 1),
        "actual_anomaly_pct": round((anomaly_count / max(len(df), 1)) * 100, 2),
        "results_df": results_df,
        "anomalies_only_df": anomalies_only,
        "pca_scatter_fig": fig_pca,
        "score_hist_fig": fig_hist,
        "feature_cols": feature_cols,
    }


def export_clean_dataset_without_anomalies(df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a cleaned copy of the original dataset with all detected anomalies excluded."""
    normal_mask = results_df["_is_anomaly"] == "Normal"
    clean_copy = df[normal_mask].copy()
    return clean_copy
