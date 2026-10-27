"""
=============================================================================
Power BI-Style Automated Analytical Dashboard Engine (src/eda.py)
=============================================================================
Features:
1. Power BI Executive Visualizations (Donut decomposition, Clustered Column, Area Trend, Treemap).
2. Clean high-contrast white card layouts with data labels.
3. Pearson Correlation Heatmap with automated mathematical insight.
4. Distribution & Skewness Profiler (Histogram with marginal box).
5. Dynamic Cross-Filtering Slicers (Category & Metric multi-selectors).
6. Power BI Custom Visual Studio (User-configurable X, Y, Legend, and Chart types).
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Power BI Inspired Color Palette
PBI_PALETTE = ["#0078D4", "#107C41", "#5C2D91", "#D83B01", "#008272", "#FFB900", "#E3008C", "#004E8C"]


def generate_powerbi_donut(df: pd.DataFrame, cat_col: str, top_n: int = 6) -> Tuple[go.Figure, str]:
    """
    Generates a Power BI style Donut chart for category decomposition.
    """
    vc = df[cat_col].value_counts().head(top_n).reset_index()
    vc.columns = [cat_col, "Count"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=vc[cat_col],
                values=vc["Count"],
                hole=0.55,
                marker=dict(colors=PBI_PALETTE[:len(vc)], line=dict(color="#000000", width=1.5)),
                textinfo="percent+label",
                insidetextorientation="radial",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"<b>Category Share: {cat_col}</b>", font=dict(size=15, color="#000000")),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    )

    top_label = vc.iloc[0][cat_col]
    top_pct = (vc.iloc[0]["Count"] / len(df[cat_col].dropna()) * 100) if len(df[cat_col].dropna()) > 0 else 0
    insight = f"💡 **Power BI Insight**: **'{top_label}'** is the dominant category, representing **{top_pct:.1f}%** of all records."
    return fig, insight


def generate_powerbi_clustered_bar(
    df: pd.DataFrame, cat_col: str, num_col: Optional[str] = None, agg_func: str = "mean", top_n: int = 8
) -> Tuple[go.Figure, str]:
    """
    Generates a Power BI clustered column/bar chart with data labels.
    """
    if num_col and num_col in df.columns and pd.api.types.is_numeric_dtype(df[num_col]):
        if agg_func == "sum":
            grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(top_n).reset_index()
            y_title = f"Total {num_col}"
        else:
            grouped = df.groupby(cat_col)[num_col].mean().round(2).sort_values(ascending=False).head(top_n).reset_index()
            y_title = f"Average {num_col}"
        y_val = num_col
    else:
        grouped = df[cat_col].value_counts().head(top_n).reset_index()
        grouped.columns = [cat_col, "Count"]
        y_title = "Record Count"
        y_val = "Count"

    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped[cat_col],
                y=grouped[y_val],
                text=grouped[y_val],
                textposition="auto",
                marker=dict(color="#0078D4", line=dict(color="#000000", width=1.5)),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"<b>{y_title} by {cat_col}</b>", font=dict(size=15, color="#000000")),
        xaxis=dict(title=cat_col, tickangle=-20),
        yaxis=dict(title=y_title),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=380,
        margin=dict(l=30, r=30, t=50, b=40),
    )

    max_c = grouped.iloc[0][cat_col]
    max_v = grouped.iloc[0][y_val]
    insight = f"💡 **Power BI Insight**: Highest ranking segment is **'{max_c}'** with **{max_v}** ({y_title.lower()})."
    return fig, insight


def generate_powerbi_area(df: pd.DataFrame, date_col: str, num_col: str) -> Tuple[Optional[go.Figure], str]:
    """
    Generates a Power BI style smooth Area chart for time-series trend analysis.
    """
    try:
        temp = df[[date_col, num_col]].dropna().copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce", format="mixed")
        temp = temp.dropna().sort_values(by=date_col)
        daily = temp.groupby(date_col)[num_col].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily[date_col],
                y=daily[num_col],
                fill="tozeroy",
                fillcolor="rgba(0, 120, 212, 0.2)",
                line=dict(color="#0078D4", width=2.5),
                mode="lines",
                name=num_col,
            )
        )
        fig.update_layout(
            title=dict(text=f"<b>Timeline Evolution: {num_col} over {date_col}</b>", font=dict(size=15, color="#000000")),
            xaxis=dict(title=date_col),
            yaxis=dict(title=f"Cumulative {num_col}"),
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            height=380,
            margin=dict(l=30, r=30, t=50, b=40),
        )

        diff = daily[num_col].iloc[-1] - daily[num_col].iloc[0] if len(daily) > 1 else 0
        direction = "Growth" if diff > 0 else "Decline" if diff < 0 else "Stable"
        insight = f"💡 **Timeline Insight**: Overall **{direction}** recorded across chronological sequence (Net Delta: {diff:+,.2f})."
        return fig, insight
    except Exception as e:
        return None, f"Could not compute time series: {str(e)}"


# Backwards compatibility alias
generate_time_trend_chart = generate_powerbi_area


def generate_powerbi_treemap(df: pd.DataFrame, cat_col: str, num_col: Optional[str] = None) -> Tuple[go.Figure, str]:

    """
    Generates a Power BI style Treemap visual for multi-level hierarchy or magnitude decomposition.
    """
    if num_col and num_col in df.columns and pd.api.types.is_numeric_dtype(df[num_col]):
        grouped = df.groupby(cat_col)[num_col].sum().reset_index()
        val_col = num_col
    else:
        grouped = df[cat_col].value_counts().reset_index()
        grouped.columns = [cat_col, "Count"]
        val_col = "Count"

    fig = px.treemap(
        grouped,
        path=[cat_col],
        values=val_col,
        color=val_col,
        color_continuous_scale="Blues",
        title=f"<b>Treemap Decomposition: {cat_col}</b>",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    insight = f"💡 **Treemap Insight**: Proportional surface area visualizes volumetric hierarchy of `{cat_col}`."
    return fig, insight


def generate_correlation_heatmap(df: pd.DataFrame) -> Tuple[Optional[go.Figure], str]:
    """
    Computes Pearson correlation matrix with Power BI light aesthetic.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return None, "At least 2 numeric columns required for correlation matrix."

    corr = numeric_df.corr().round(2)
    cols = list(corr.columns)
    z = corr.values.tolist()
    z_text = [[str(val) for val in row] for row in z]

    fig = ff.create_annotated_heatmap(
        z,
        x=cols,
        y=cols,
        annotation_text=z_text,
        colorscale="Blues",
    )
    fig.update_layout(
        title=dict(text="<b>Correlation Matrix Heatmap</b>", font=dict(size=15, color="#000000")),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    np.fill_diagonal(corr.values, 0)
    max_val = corr.abs().max().max()
    max_pair = None
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 != c2 and abs(corr.loc[c1, c2]) == max_val:
                max_pair = (c1, c2, corr.loc[c1, c2])
                break
        if max_pair:
            break

    if max_pair:
        c1, c2, val = max_pair
        direction = "positive" if val > 0 else "negative"
        insight = f"💡 **Correlation Insight**: Strongest relationship is between `{c1}` and `{c2}` ({direction} $r = {val:.2f}$). "
    else:
        insight = "💡 **Correlation Insight**: Variables show near-zero mutual linear dependencies."

    return fig, insight


def generate_distribution_chart(df: pd.DataFrame, col: str) -> Tuple[go.Figure, str]:
    """
    Builds a distribution histogram with boxplot marginal in Power BI style.
    """
    series = df[col].dropna()
    skew = series.skew()
    mean_val = series.mean()
    median_val = series.median()

    fig = px.histogram(
        df,
        x=col,
        marginal="box",
        nbins=25,
        color_discrete_sequence=["#0078D4"],
        template="plotly_white",
    )
    fig.update_layout(
        title=dict(text=f"<b>Distribution Spread: {col}</b>", font=dict(size=15, color="#000000")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
    )

    skew_txt = "symmetric" if abs(skew) < 0.5 else "right-skewed" if skew > 0.5 else "left-skewed"
    insight = f"💡 **Distribution Insight**: `{col}` is **{skew_txt}** (skew = {skew:.2f}). Mean: {mean_val:.2f}, Median: {median_val:.2f}."
    return fig, insight


def build_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
) -> go.Figure:
    """
    Universal Power BI Visual Studio.
    Supports: Donut, Treemap, Clustered Bar, Line, Scatter, Box, Histogram.
    """
    kwargs = {
        "data_frame": df,
        "x": x_col,
        "template": "plotly_white",
        "color_discrete_sequence": PBI_PALETTE,
    }
    if y_col and y_col != "None":
        kwargs["y"] = y_col
    if color_col and color_col != "None":
        kwargs["color"] = color_col

    if chart_type == "Donut":
        return generate_powerbi_donut(df, x_col)[0]
    elif chart_type == "Treemap":
        return generate_powerbi_treemap(df, x_col, y_col if y_col != "None" else None)[0]
    elif chart_type == "Clustered Bar":
        return generate_powerbi_clustered_bar(df, x_col, y_col if y_col != "None" else None)[0]
    elif chart_type == "Scatter":
        fig = px.scatter(**kwargs)
    elif chart_type == "Line":
        fig = px.line(**kwargs)
    elif chart_type == "Box":
        fig = px.box(**kwargs)
    elif chart_type == "Histogram":
        fig = px.histogram(**kwargs)
    else:
        fig = px.bar(**kwargs)

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=400,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def generate_categorical_chart(df: pd.DataFrame, cat_col: str, top_n: int = 10) -> Tuple[go.Figure, str]:
    """Backwards compatibility alias for Power BI categorical visual."""
    return generate_powerbi_clustered_bar(df, cat_col, None)


def generate_powerbi_matrix(df: pd.DataFrame, cat_col: str, num_col: str) -> pd.DataFrame:
    """
    Builds a Power BI style matrix table with aggregation metrics
    (Count, Total Sum, Average, Minimum, Maximum, and Share %).
    """
    if cat_col not in df.columns or num_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(cat_col)[num_col].agg(
        Count="count",
        Sum="sum",
        Mean="mean",
        Min="min",
        Max="max",
    ).reset_index()

    total_sum = grouped["Sum"].sum()
    grouped["Share %"] = ((grouped["Sum"] / total_sum) * 100).round(2) if total_sum > 0 else 0
    grouped["Mean"] = grouped["Mean"].round(2)
    grouped["Sum"] = grouped["Sum"].round(2)
    grouped = grouped.sort_values(by="Sum", ascending=False).reset_index(drop=True)
    return grouped


def generate_powerbi_gauge(value: float, min_val: float, max_val: float, title: str) -> go.Figure:
    """
    Builds a Power BI style Gauge KPI chart with target thresholds.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{title}</b>", "font": {"size": 14, "color": "#000000"}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": "#000000"},
                "bar": {"color": "#0078D4"},
                "bgcolor": "#f1f5f9",
                "borderwidth": 1.5,
                "bordercolor": "#000000",
                "steps": [
                    {"range": [min_val, min_val + 0.5 * (max_val - min_val)], "color": "#fee2e2"},
                    {"range": [min_val + 0.5 * (max_val - min_val), min_val + 0.8 * (max_val - min_val)], "color": "#fef08a"},
                    {"range": [min_val + 0.8 * (max_val - min_val), max_val], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


