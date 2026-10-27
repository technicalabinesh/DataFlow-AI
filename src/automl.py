"""
=============================================================================
Automated Machine Learning Engine: Tabular (src/automl.py)
=============================================================================
Features:
1. Auto-detect task type: Classification vs. Regression based on target column.
2. Leak-free scikit-learn Pipeline with ColumnTransformer:
   - Median imputation + StandardScaler for numeric features
   - Most-frequent imputation + OneHotEncoder for categorical features
3. Trains 3+ models with cross-validation:
   - Classification: Logistic Regression, Random Forest, Gradient Boosting
   - Regression: Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor
4. Comprehensive evaluation metrics:
   - Classification: Accuracy, Precision, Recall, F1 (weighted), Confusion Matrix
   - Regression: RMSE, MAE, R-squared (R2)
5. Model comparison leaderboard & Feature Importance extraction.
6. Serialization of best pipeline to models/<name>.pkl.
7. Dynamic inference function for user-submitted feature inputs.
"""

import os
import joblib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import plotly.graph_objects as go
import plotly.figure_factory as ff

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def infer_task_type(series: pd.Series) -> str:
    """
    Infers whether the machine learning task is 'classification' or 'regression'.
    - Categorical/object/boolean/string -> classification
    - Numeric with <= 10 unique values -> classification
    - Numeric with > 10 unique values -> regression
    """
    clean_series = series.dropna()
    if clean_series.dtype == "object" or clean_series.dtype.name == "category" or clean_series.dtype == "bool":
        return "classification"
    
    unique_vals = clean_series.nunique()
    if unique_vals <= 10:
        return "classification"
    return "regression"


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Constructs a leak-free ColumnTransformer pipeline for numeric and categorical features.
    """
    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols


def train_tabular_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    task_type: Optional[str] = None,
    quick_tune: bool = False,
) -> Dict[str, Any]:
    """
    Trains multiple models using leak-free scikit-learn pipelines.
    Evaluates on test_df and returns performance leaderboard, feature importances,
    confusion matrix (for classification), and the serialized best pipeline.
    """
    if task_type is None:
        task_type = infer_task_type(train_df[target_col])

    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Handle missing values in target by dropping rows
    valid_train_idx = y_train.dropna().index
    X_train, y_train = X_train.loc[valid_train_idx], y_train.loc[valid_train_idx]
    valid_test_idx = y_test.dropna().index
    X_test, y_test = X_test.loc[valid_test_idx], y_test.loc[valid_test_idx]

    # Drop obvious high-cardinality ID columns (e.g. Name, ID, OrderID)
    id_cols = [c for c in X_train.columns if "id" in c.lower() or "name" in c.lower() or X_train[c].nunique() == len(X_train)]
    X_train = X_train.drop(columns=id_cols)
    X_test = X_test.drop(columns=[c for c in id_cols if c in X_test.columns])

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    # Initialize model suites
    if task_type == "classification":
        models_dict = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        }
        metric_key = "Accuracy"
    else:
        models_dict = {
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        }
        metric_key = "R2 Score"

    leaderboard = []
    trained_pipelines = {}
    best_model_name = None
    best_score = -float("inf")

    # Fit and evaluate each pipeline
    for name, model_instance in models_dict.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_instance),
        ])

        # Cross-validation on train set (3-fold)
        scoring = "accuracy" if task_type == "classification" else "r2"
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=scoring)
            cv_mean = float(np.mean(cv_scores))
        except Exception:
            cv_mean = 0.0

        # Fit on full train set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predict on test set
        preds = pipeline.predict(X_test)

        if task_type == "classification":
            acc = float(accuracy_score(y_test, preds))
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
            row = {
                "Model": name,
                "Accuracy (%)": round(acc * 100, 2),
                "Precision (%)": round(float(prec) * 100, 2),
                "Recall (%)": round(float(rec) * 100, 2),
                "F1 Score (%)": round(float(f1) * 100, 2),
                "CV Score (%)": round(cv_mean * 100, 2),
            }
            leaderboard.append(row)
            if acc > best_score:
                best_score = acc
                best_model_name = name
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            row = {
                "Model": name,
                "R2 Score": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "CV Score (R2)": round(cv_mean, 4),
            }
            leaderboard.append(row)
            if r2 > best_score:
                best_score = r2
                best_model_name = name

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(
        by=leaderboard[0].keys().__iter__().__next__(), ascending=False
    )

    best_pipeline = trained_pipelines[best_model_name]

    # Compute Confusion Matrix if classification
    cm_fig = None
    if task_type == "classification":
        best_preds = best_pipeline.predict(X_test)
        labels = sorted(list(set(y_test).union(set(best_preds))))
        labels_str = [str(l) for l in labels]
        cm = confusion_matrix(y_test, best_preds, labels=labels).tolist()
        cm_fig = ff.create_annotated_heatmap(
            cm,
            x=labels_str,
            y=labels_str,
            annotation_text=[[str(v) for v in r] for r in cm],
            colorscale="Viridis",
        )
        cm_fig.update_layout(
            title=f"Test Confusion Matrix — Best Model ({best_model_name})",
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=360,
        )

    # Feature Importances for Tree-based models
    feature_imp_fig = None
    try:
        model_step = best_pipeline.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            # Extract feature names from preprocessor
            ohe = best_pipeline.named_steps["preprocessor"].named_transformers_.get("cat")
            cat_feature_names = []
            if ohe and hasattr(ohe.named_steps["onehot"], "get_feature_names_out"):
                cat_feature_names = list(ohe.named_steps["onehot"].get_feature_names_out(cat_cols))
            
            all_feature_names = num_cols + cat_feature_names
            importances = model_step.feature_importances_
            
            if len(all_feature_names) == len(importances):
                imp_df = pd.DataFrame({
                    "Feature": all_feature_names,
                    "Importance": importances,
                }).sort_values(by="Importance", ascending=True).tail(12)

                feature_imp_fig = go.Figure(
                    go.Bar(
                        x=imp_df["Importance"],
                        y=imp_df["Feature"],
                        orientation="h",
                        marker=dict(color="#38bdf8"),
                    )
                )
                feature_imp_fig.update_layout(
                    title=f"Top Feature Importances ({best_model_name})",
                    xaxis_title="Relative Importance Score",
                    yaxis_title="Feature Name",
                    template="plotly_dark",
                    paper_bgcolor="rgba(15, 23, 42, 0.4)",
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    height=360,
                    margin=dict(l=80, r=30, t=50, b=40),
                )
    except Exception:
        pass

    # Save best model to disk
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_target = "".join(c for c in target_col if c.isalnum() or c == "_")
    model_filename = f"best_model_{clean_target}_{timestamp}.pkl"
    model_save_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(best_pipeline, model_save_path)

    return {
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": list(X_train.columns),
        "leaderboard": leaderboard_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "model_save_path": model_save_path,
        "confusion_matrix_fig": cm_fig,
        "feature_importance_fig": feature_imp_fig,
        "dropped_id_cols": id_cols,
    }


def predict_tabular_sample(pipeline: Pipeline, input_dict: Dict[str, Any]) -> Any:
    """
    Takes a single dictionary of feature values and generates a prediction using the trained pipeline.
    """
    df_sample = pd.DataFrame([input_dict])
    prediction = pipeline.predict(df_sample)[0]
    return prediction
