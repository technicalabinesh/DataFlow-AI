"""
=============================================================================
Natural Language Processing (NLP) Engine (src/nlp_engine.py)
=============================================================================
Features:
1. Sentiment Analysis & Polarity Scoring (Positive / Neutral / Negative).
2. Lexicon & Feature-weighted sentiment scoring with confidence badges.
3. Batch Dataset Text Column Profiler with sentiment breakdown charts.
4. Keyword, N-Gram & Topic Frequency Extraction with stopword purging.
5. Extractive Text Summarization & Readability Metrics.
6. Trainable Scikit-Learn Text Classification Pipeline (TF-IDF + Classifier).
7. Interactive Plotly charts for sentiment distributions and word frequency.
"""

import re
import math
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px

# Curated high-impact positive and negative lexicons
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "terrific",
    "outstanding", "superb", "brilliant", "love", "loved", "loving", "best",
    "awesome", "happy", "pleased", "delighted", "positive", "impressive",
    "perfect", "flawless", "reliable", "fast", "efficient", "helpful", "clean",
    "smooth", "safe", "gain", "profit", "success", "successful", "superior",
    "valuable", "satisfying", "satisfied", "recommend", "recommended", "worth",
    "win", "winner", "innovative", "top", "favorite", "fabulous", "nice", "enjoy"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "worst", "hate", "hated",
    "disappointing", "disappointed", "disappointment", "fail", "failed", "failure",
    "broken", "defect", "defective", "slow", "sluggish", "useless", "annoying",
    "waste", "rubbish", "trash", "error", "bug", "crash", "loss", "losing",
    "lost", "expensive", "unreliable", "rude", "dirty", "pain", "painful",
    "problem", "problems", "issue", "issues", "complaint", "fraud", "scam",
    "danger", "dangerous", "delay", "delayed", "boring", "ugly", "suck", "sucks"
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment of a text string using normalized valence scoring.
    Returns polarity (-1.0 to +1.0), label (POSITIVE, NEUTRAL, NEGATIVE), and signals.
    """
    if not text or not isinstance(text, str):
        return {
            "label": "NEUTRAL",
            "polarity": 0.0,
            "confidence": 50.0,
            "positive_count": 0,
            "negative_count": 0,
            "detected_keywords": [],
        }

    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    if not words:
        return {
            "label": "NEUTRAL",
            "polarity": 0.0,
            "confidence": 50.0,
            "positive_count": 0,
            "negative_count": 0,
            "detected_keywords": [],
        }

    pos_hits = [w for w in words if w in POSITIVE_WORDS]
    neg_hits = [w for w in words if w in NEGATIVE_WORDS]

    pos_count = len(pos_hits)
    neg_count = len(neg_hits)
    total_hits = pos_count + neg_count

    if total_hits == 0:
        polarity = 0.0
        label = "NEUTRAL"
        confidence = 65.0
    else:
        raw_score = (pos_count - neg_count) / total_hits
        # Dampen by total words to reflect overall document intensity
        intensity = min(1.0, math.sqrt(total_hits / max(len(words), 1)) * 1.5)
        polarity = round(raw_score * intensity, 3)

        if polarity >= 0.15:
            label = "POSITIVE"
            confidence = round(min(99.0, 50.0 + abs(polarity) * 50.0), 1)
        elif polarity <= -0.15:
            label = "NEGATIVE"
            confidence = round(min(99.0, 50.0 + abs(polarity) * 50.0), 1)
        else:
            label = "NEUTRAL"
            confidence = round(100.0 - (abs(polarity) * 100.0), 1)

    return {
        "label": label,
        "polarity": polarity,
        "confidence": confidence,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "detected_keywords": list(set(pos_hits + neg_hits))[:10],
    }


def analyze_dataset_text_column(
    df: pd.DataFrame, text_col: str, max_rows: int = 1500
) -> Dict[str, Any]:
    """
    Performs vectorized batch sentiment and readability analysis across an entire text column.
    """
    sample_df = df[df[text_col].notna()].head(max_rows).copy()
    if sample_df.empty:
        return {"error": "Selected column contains no valid text records."}

    sentiments = []
    polarities = []
    for val in sample_df[text_col].astype(str):
        s = analyze_text_sentiment(val)
        sentiments.append(s["label"])
        polarities.append(s["polarity"])

    sample_df["_sentiment"] = sentiments
    sample_df["_polarity"] = polarities

    counts = Counter(sentiments)
    total = len(sentiments)
    summary_stats = {
        "total_analyzed": total,
        "positive_pct": round((counts.get("POSITIVE", 0) / total) * 100, 1),
        "neutral_pct": round((counts.get("NEUTRAL", 0) / total) * 100, 1),
        "negative_pct": round((counts.get("NEGATIVE", 0) / total) * 100, 1),
        "mean_polarity": round(float(np.mean(polarities)), 3),
    }

    # Donut visual for sentiment split
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[
                    counts.get("POSITIVE", 0),
                    counts.get("NEUTRAL", 0),
                    counts.get("NEGATIVE", 0),
                ],
                hole=0.55,
                marker=dict(colors=["#bef264", "#94a3b8", "#f43f5e"]),
                textinfo="label+percent",
            )
        ]
    )
    fig_donut.update_layout(
        title=f"Sentiment Distribution for '{text_col}'",
        template="plotly_dark",
        height=320,
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )

    return {
        "summary": summary_stats,
        "sentiment_df": sample_df[[text_col, "_sentiment", "_polarity"]].head(25),
        "donut_fig": fig_donut,
        "counts": counts,
    }


def extract_keywords_and_ngrams(
    text_data, top_n: int = 15
) -> Dict[str, Any]:
    """
    Extracts top unigram and bigram frequency keywords, filtering out common stopwords.
    """
    if isinstance(text_data, (pd.Series, list)):
        corpus = " ".join([str(t) for t in text_data if pd.notna(t)])
    else:
        corpus = str(text_data)

    words = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{3,}\b", corpus)
        if w.lower() not in STOPWORDS
    ]

    if not words:
        return {"unigrams": [], "bigrams": [], "chart": None}

    # Unigram counts
    uni_counts = Counter(words).most_common(top_n)
    df_uni = pd.DataFrame(uni_counts, columns=["Keyword", "Frequency"])

    # Bigram counts
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bi_counts = Counter(bigrams).most_common(top_n)
    df_bi = pd.DataFrame(bi_counts, columns=["Bigram Phrase", "Frequency"])

    # Interactive Bar chart for top keywords
    fig_bar = px.bar(
        df_uni.sort_values("Frequency", ascending=True),
        x="Frequency",
        y="Keyword",
        orientation="h",
        color="Frequency",
        color_continuous_scale="Purples",
        title=f"Top {top_n} Salient Keywords",
    )
    fig_bar.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(t=40, b=20, l=40, r=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )

    return {
        "unigrams": df_uni,
        "bigrams": df_bi,
        "chart": fig_bar,
    }


def summarize_text_extractive(
    text: str, max_sentences: int = 3
) -> Dict[str, Any]:
    """
    Lightweight extractive text summarizer using word frequency scoring.
    Computes readability metrics (word count, reading time, sentence count).
    """
    if not text or not isinstance(text, str):
        return {"summary": "", "metrics": {}}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]
    if len(sentences) <= max_sentences:
        return {
            "summary": text,
            "metrics": {
                "word_count": len(text.split()),
                "sentence_count": len(sentences),
                "reading_time_sec": max(1, round(len(text.split()) / 3.5)),
            },
        }

    words = [
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)
        if w.lower() not in STOPWORDS
    ]
    freqs = Counter(words)
    max_f = max(freqs.values()) if freqs else 1
    norm_freqs = {w: count / max_f for w, count in freqs.items()}

    sentence_scores = []
    for s in sentences:
        s_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", s)]
        score = sum(norm_freqs.get(w, 0) for w in s_words) / max(len(s_words), 1)
        sentence_scores.append((score, s))

    ranked = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    summary = " ".join([s for _, s in sorted(ranked, key=lambda x: sentences.index(x[1]))])

    total_words = len(text.split())
    summary_words = len(summary.split())
    compression = round((1 - (summary_words / max(total_words, 1))) * 100, 1)

    return {
        "summary": summary,
        "metrics": {
            "original_words": total_words,
            "summary_words": summary_words,
            "compression_ratio": f"{compression}% reduced",
            "reading_time_sec": max(1, round(total_words / 3.5)),
        },
    }


def train_text_classifier(
    df: pd.DataFrame, text_col: str, label_col: str
) -> Dict[str, Any]:
    """
    Builds a full TF-IDF + Logistic Regression classification pipeline.
    Returns test accuracy, cross-validation score, and a live inference function.
    """
    clean_df = df[[text_col, label_col]].dropna().copy()
    clean_df = clean_df[clean_df[text_col].astype(str).str.strip().str.len() > 3]

    if clean_df[label_col].nunique() < 2:
        raise ValueError("Target label column must have at least 2 distinct classes.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])

    X = clean_df[text_col].astype(str)
    y = clean_df[label_col].astype(str)

    # 3-Fold Cross-Validation
    cv_scores = cross_val_score(pipeline, X, y, cv=min(3, len(clean_df)))
    pipeline.fit(X, y)

    def predict_new_text(sample_text: str) -> Dict[str, Any]:
        feat_vec = pipeline.named_steps["tfidf"].transform([sample_text])
        classes = list(pipeline.named_steps["clf"].classes_)

        # If zero features matched vocabulary, check sentiment lexicon fallback
        if feat_vec.nnz == 0:
            s = analyze_text_sentiment(sample_text)
            class_lower = [str(c).lower() for c in classes]
            if s["label"].lower() in class_lower:
                match_idx = class_lower.index(s["label"].lower())
                pred = classes[match_idx]
                confidence = s["confidence"]
                class_prob_map = {c: 15.0 for c in classes}
                class_prob_map[pred] = confidence
                return {
                    "prediction": pred,
                    "confidence": confidence,
                    "probabilities": class_prob_map,
                }

        pred = pipeline.predict([sample_text])[0]
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probs = pipeline.predict_proba([sample_text])[0]
            class_prob_map = {classes[i]: round(float(probs[i]) * 100, 1) for i in range(len(classes))}
            confidence = round(float(np.max(probs)) * 100, 1)
        else:
            class_prob_map = {pred: 100.0}
            confidence = 100.0

        return {
            "prediction": pred,
            "confidence": confidence,
            "probabilities": class_prob_map,
        }

    return {
        "pipeline": pipeline,
        "cv_score": round(float(np.mean(cv_scores)) * 100, 2),
        "classes": list(clean_df[label_col].unique()),
        "sample_size": len(clean_df),
        "predict_fn": predict_new_text,
    }
