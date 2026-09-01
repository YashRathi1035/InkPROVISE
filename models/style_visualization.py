# ============================================================
# HANDWRITING STYLE VISUALIZATION
# Phase 4C
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FEATURE CONSISTENCY DATAFRAME
# ============================================================

def create_consistency_dataframe(
    feature_consistency: dict,
) -> pd.DataFrame:
    """
    Convert feature consistency dictionary into
    a Pandas DataFrame.
    """

    if not feature_consistency:

        return pd.DataFrame(
            columns=[
                "Feature",
                "Consistency",
            ]
        )

    dataframe = pd.DataFrame(
        list(
            feature_consistency.items()
        ),
        columns=[
            "Feature",
            "Consistency",
        ],
    )

    dataframe = dataframe.sort_values(
        by="Consistency",
        ascending=False,
    )

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe


# ============================================================
# FEATURE SUMMARY DATAFRAME
# ============================================================

def create_feature_summary_dataframe(
    style_profile: dict,
) -> pd.DataFrame:
    """
    Create a summary table containing
    mean, standard deviation, minimum
    and maximum values.
    """

    mean_values = (
        style_profile.get(
            "mean",
            {},
        )
    )

    std_values = (
        style_profile.get(
            "std",
            {},
        )
    )

    min_values = (
        style_profile.get(
            "min",
            {},
        )
    )

    max_values = (
        style_profile.get(
            "max",
            {},
        )
    )

    feature_names = list(
        mean_values.keys()
    )

    rows = []

    for feature in feature_names:

        rows.append(
            {
                "Feature": feature,
                "Mean": mean_values.get(
                    feature,
                    0.0,
                ),
                "Std": std_values.get(
                    feature,
                    0.0,
                ),
                "Min": min_values.get(
                    feature,
                    0.0,
                ),
                "Max": max_values.get(
                    feature,
                    0.0,
                ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    return dataframe


# ============================================================
# TOP STABLE FEATURES
# ============================================================

def get_most_consistent_features(
    consistency_dataframe: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the handwriting features with
    the highest consistency.
    """

    if consistency_dataframe.empty:

        return consistency_dataframe

    return (
        consistency_dataframe
        .head(top_n)
        .copy()
    )


# ============================================================
# MOST VARIABLE FEATURES
# ============================================================

def get_most_variable_features(
    consistency_dataframe: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the handwriting features with
    the lowest consistency.
    """

    if consistency_dataframe.empty:

        return consistency_dataframe

    return (
        consistency_dataframe
        .tail(top_n)
        .sort_values(
            by="Consistency",
            ascending=True,
        )
        .copy()
    )


# ============================================================
# FEATURE CONSISTENCY CHART
# ============================================================

def plot_feature_consistency(
    consistency_dataframe: pd.DataFrame,
):
    """
    Create a bar chart showing
    handwriting feature consistency.
    """

    if consistency_dataframe.empty:

        return None

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.bar(
        consistency_dataframe[
            "Feature"
        ],
        consistency_dataframe[
            "Consistency"
        ],
    )

    axis.set_title(
        "Handwriting Feature Consistency"
    )

    axis.set_xlabel(
        "Features"
    )

    axis.set_ylabel(
        "Consistency Score (%)"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.set_ylim(
        0,
        100,
    )

    figure.tight_layout()

    return figure


# ============================================================
# FEATURE DISTRIBUTION CHART
# ============================================================

def plot_feature_distribution(
    dataframe: pd.DataFrame,
    feature_name: str,
):
    """
    Plot a selected handwriting feature
    across all uploaded samples.
    """

    if dataframe.empty:

        return None

    if feature_name not in dataframe.columns:

        return None

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.plot(
        dataframe.index.astype(
            str
        ),
        dataframe[
            feature_name
        ],
        marker="o",
    )

    axis.set_title(
        f"{feature_name} Across Samples"
    )

    axis.set_xlabel(
        "Handwriting Sample"
    )

    axis.set_ylabel(
        feature_name
    )

    axis.tick_params(
        axis="x",
        rotation=30,
    )

    figure.tight_layout()

    return figure


# ============================================================
# FEATURE DISTRIBUTION SUMMARY
# ============================================================

def get_feature_distribution_summary(
    dataframe: pd.DataFrame,
    feature_name: str,
) -> dict:
    """
    Return useful statistical information
    about one selected feature.
    """

    if dataframe.empty:

        return {}

    if feature_name not in dataframe.columns:

        return {}

    values = dataframe[
        feature_name
    ]

    return {
        "mean": float(
            values.mean()
        ),
        "std": float(
            values.std()
            if len(values) > 1
            else 0.0
        ),
        "min": float(
            values.min()
        ),
        "max": float(
            values.max()
        ),
    }