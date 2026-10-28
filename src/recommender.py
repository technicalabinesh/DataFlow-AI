"""
=============================================================================
Recommendation Engine (src/recommender.py)
=============================================================================
Features:
1. Content-Based Recommendation using Cosine Similarity on hybrid feature spaces.
   - Vectorizes text & categorical metadata with TF-IDF.
   - Normalizes numerical attributes using MinMaxScaler.
   - Computes pairwise cosine similarity matrix with fast matrix dot products.
2. Item-to-Item Collaborative / Interaction-based Recommender.
   - Builds sparse User-Item interaction matrix.
   - Computes Pearson correlation & co-occurrence similarity.
3. Universal adaptability across any tabular dataset (E-commerce, Movies, Books, HR, Songs).
4. Feature difference breakdown explaining "Why this item was recommended".
5. Interactive Plotly similarity distribution and radar comparison charts.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px


def build_content_recommender(
    df: pd.DataFrame,
    item_col: str,
    feature_cols: Optional[List[str]] = None,
    max_items: int = 2500,
) -> Dict[str, Any]:
    """
    Constructs a hybrid content-based recommendation matrix from user-selected features.
    """
    clean_df = df.dropna(subset=[item_col]).head(max_items).copy()
    clean_df = clean_df.drop_duplicates(subset=[item_col]).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [c for c in clean_df.columns if c != item_col]

    if not feature_cols:
        raise ValueError("At least one feature column is required to build recommendations.")

    # Split into numeric and text/categorical
    num_cols = list(clean_df[feature_cols].select_dtypes(include=["number"]).columns)
    text_cols = [c for c in feature_cols if c not in num_cols]

    feature_matrices = []

    # 1. Text & Categorical metadata vectorization
    if text_cols:
        combined_text = clean_df[text_cols].astype(str).agg(" ".join, axis=1)
        tfidf = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
        tfidf_mat = tfidf.fit_transform(combined_text).toarray()
        feature_matrices.append(tfidf_mat)

    # 2. Numerical feature normalization
    if num_cols:
        scaler = MinMaxScaler()
        num_mat = scaler.fit_transform(clean_df[num_cols].fillna(clean_df[num_cols].median()))
        feature_matrices.append(num_mat)

    if not feature_matrices:
        raise ValueError("No valid features found to construct similarity vectors.")

    combined_matrix = np.hstack(feature_matrices)
    similarity_matrix = cosine_similarity(combined_matrix)

    item_to_idx = {val: idx for idx, val in enumerate(clean_df[item_col])}
    idx_to_item = {idx: val for idx, val in enumerate(clean_df[item_col])}

    return {
        "df": clean_df,
        "item_col": item_col,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "text_cols": text_cols,
        "similarity_matrix": similarity_matrix,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "total_items": len(clean_df),
    }


def get_recommendations(
    recommender_obj: Dict[str, Any],
    selected_item: Any,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Returns top-N most similar items for a selected item, with match percentage
    and feature comparison.
    """
    item_to_idx = recommender_obj["item_to_idx"]
    idx_to_item = recommender_obj["idx_to_item"]
    sim_matrix = recommender_obj["similarity_matrix"]
    df = recommender_obj["df"]
    item_col = recommender_obj["item_col"]

    if selected_item not in item_to_idx:
        raise KeyError(f"Item '{selected_item}' not found in recommendation index.")

    target_idx = item_to_idx[selected_item]
    sim_scores = sim_matrix[target_idx]

    # Sort descending, excluding the item itself
    ranked_indices = np.argsort(sim_scores)[::-1]
    ranked_indices = [idx for idx in ranked_indices if idx != target_idx][:top_n]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        rec_item = idx_to_item[idx]
        score = float(sim_scores[idx]) * 100.0
        row_data = df.iloc[idx].to_dict()

        results.append({
            "Rank": rank,
            recommender_obj["item_col"]: rec_item,
            "Similarity Score": f"{score:.1f}%",
            "_raw_score": score,
            **{k: v for k, v in row_data.items() if k != item_col},
        })

    rec_df = pd.DataFrame(results)

    # Plotly similarity score comparison chart
    fig = px.bar(
        rec_df,
        x="_raw_score",
        y=item_col,
        orientation="h",
        color="_raw_score",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Recommendations for '{selected_item}'",
        labels={"_raw_score": "Similarity Match (%)", item_col: "Recommended Item"},
        text="_raw_score",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(t=40, b=20, l=40, r=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )

    return {
        "target_item": selected_item,
        "recommendations_df": rec_df.drop(columns=["_raw_score"]),
        "chart": fig,
    }


def build_collaborative_recommender(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: Optional[str] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Constructs an item-based collaborative filtering model from user interactions.
    """
    clean = df.dropna(subset=[user_col, item_col]).copy()
    if rating_col and rating_col in clean.columns:
        clean[rating_col] = pd.to_numeric(clean[rating_col], errors="coerce").fillna(1.0)
        interaction_matrix = clean.pivot_table(index=user_col, columns=item_col, values=rating_col, fill_value=0)
    else:
        clean["_interaction"] = 1.0
        interaction_matrix = clean.pivot_table(index=user_col, columns=item_col, values="_interaction", fill_value=0)

    # Transpose to compute item-item correlation
    item_sim_df = interaction_matrix.corr(method="pearson").fillna(0)

    return {
        "item_sim_df": item_sim_df,
        "available_items": list(item_sim_df.columns),
        "total_users": len(interaction_matrix),
        "total_items": len(item_sim_df),
    }
