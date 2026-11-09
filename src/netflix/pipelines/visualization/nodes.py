"""Nodes of the 'visualization' pipeline: exploratory charts of the clean table.

Figures are built with ``matplotlib.figure.Figure`` (not pyplot), so no GUI backend
is needed and nothing stays open in memory. Each ``plot_*`` function returns one
figure; ``plot_eda`` collects them, keyed by output file name.
"""

import math

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

COLORS = {0: "#4C78A8", 1: "#E45756"}  # stayed, churned
MAX_DISCRETE_LEVELS = 8  # numeric columns with fewer levels are treated as discrete
CLIP_QUANTILE = 0.99


def _grid(n: int, ncols: int = 3) -> tuple[Figure, np.ndarray]:
    """A figure with ``n`` axes in a grid; unused cells are hidden."""
    nrows = math.ceil(n / ncols)
    fig = Figure(figsize=(4.2 * ncols, 3.2 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _feature_columns(df: pd.DataFrame, parameters: dict) -> tuple[list, list]:
    """Numeric and categorical feature names, without the id and the target."""
    features = df.drop(columns=[parameters["id_column"], parameters["target_column"]])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def plot_target_balance(df: pd.DataFrame, parameters: dict) -> Figure:
    counts = df[parameters["target_column"]].value_counts().sort_index()
    fig = Figure(figsize=(4, 3.2), layout="constrained")
    ax = fig.subplots()
    ax.bar(
        ["Stayed", "Churned"],
        counts.to_numpy(),
        color=[COLORS[0], COLORS[1]],
    )
    for i, value in enumerate(counts):
        ax.text(
            i, value, f"{value} ({value / counts.sum():.1%})", ha="center", va="bottom"
        )
    ax.set_ylabel("Customers")
    ax.set_title("Target balance")
    return fig


def plot_numeric_distributions(df: pd.DataFrame, parameters: dict) -> Figure:
    """Histogram of each numeric column, one colour per class.

    The upper tail is cut at the 99th percentile so a few extreme values do not
    squash the chart.
    """
    target = parameters["target_column"]
    numeric, _ = _feature_columns(df, parameters)
    fig, axes = _grid(len(numeric))
    for ax, col in zip(axes, numeric, strict=True):
        bins = np.linspace(df[col].min(), df[col].quantile(CLIP_QUANTILE), 31)
        for label in (0, 1):
            ax.hist(
                df.loc[df[target] == label, col],
                bins=bins,
                density=True,
                alpha=0.6,
                color=COLORS[label],
                label=["Stayed", "Churned"][label],
            )
        ax.set_title(col)
        ax.set_yticks([])
    axes[0].legend()
    fig.suptitle("Distribution by class (density, upper tail clipped at p99)")
    return fig


def _churn_rate_bars(ax: Axes, df: pd.DataFrame, col: str, target: str, bins: int):
    """Bar chart of churn rate per level of ``col`` (numeric columns are binned)."""
    series = df[col]
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > MAX_DISCRETE_LEVELS:
        groups = pd.qcut(series, bins, duplicates="drop")
        labels = [f"{iv.left:.3g}–{iv.right:.3g}" for iv in groups.cat.categories]
    else:
        groups = series
        labels = None
    rate = df.groupby(groups, observed=True)[target].mean()
    ax.bar(
        labels or rate.index.astype(str), rate.to_numpy(), color=COLORS[1], alpha=0.8
    )
    ax.axhline(df[target].mean(), color="grey", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title(col)
    ax.tick_params(axis="x", labelrotation=30)


def plot_churn_rate_numeric(df: pd.DataFrame, parameters: dict) -> Figure:
    """Churn rate across equal-size groups of each numeric column.

    Shows non-linear effects, such as the jump after ~30 days without login.
    """
    numeric, _ = _feature_columns(df, parameters)
    fig, axes = _grid(len(numeric))
    for ax, col in zip(axes, numeric, strict=True):
        _churn_rate_bars(
            ax, df, col, parameters["target_column"], parameters["numeric_bins"]
        )
    fig.suptitle("Churn rate by group (dashed line: overall churn rate)")
    return fig


def plot_churn_rate_categorical(df: pd.DataFrame, parameters: dict) -> Figure:
    _, categorical = _feature_columns(df, parameters)
    fig, axes = _grid(len(categorical))
    for ax, col in zip(axes, categorical, strict=True):
        _churn_rate_bars(
            ax, df, col, parameters["target_column"], parameters["numeric_bins"]
        )
    fig.suptitle("Churn rate by category (dashed line: overall churn rate)")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, parameters: dict) -> Figure:
    """Spearman correlation between numeric columns and the target."""
    target = parameters["target_column"]
    numeric, _ = _feature_columns(df, parameters)
    corr = df[[*numeric, target]].corr(method="spearman")

    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for (row, col), value in np.ndenumerate(corr.to_numpy()):
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation")
    return fig


def plot_eda(netflix_clean: pd.DataFrame, parameters: dict) -> dict[str, Figure]:
    """All exploratory figures, keyed by output file name."""
    return {
        "target_balance.png": plot_target_balance(netflix_clean, parameters),
        "numeric_distributions.png": plot_numeric_distributions(
            netflix_clean, parameters
        ),
        "churn_rate_numeric.png": plot_churn_rate_numeric(netflix_clean, parameters),
        "churn_rate_categorical.png": plot_churn_rate_categorical(
            netflix_clean, parameters
        ),
        "correlation_heatmap.png": plot_correlation_heatmap(netflix_clean, parameters),
    }
