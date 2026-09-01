# ============================================================
# HANDWRITING STYLE MODEL
# Phase 4A + Phase 4B
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURE DATASET
# ============================================================

def create_feature_dataframe(
    feature_list: list[dict],
) -> pd.DataFrame:
    """
    Convert handwriting feature dictionaries into
    a Pandas DataFrame.
    """

    if not feature_list:

        raise ValueError(
            "Feature list cannot be empty."
        )

    dataframe = pd.DataFrame(
        feature_list
    )

    return dataframe


# ============================================================
# NUMERIC FEATURE SELECTION
# ============================================================

def select_numeric_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only numeric handwriting features.
    """

    if dataframe.empty:

        raise ValueError(
            "Feature dataframe cannot be empty."
        )

    numeric_dataframe = (
        dataframe.select_dtypes(
            include=[
                np.number,
            ]
        )
    )

    if numeric_dataframe.empty:

        raise ValueError(
            "No numeric features found."
        )

    return numeric_dataframe.copy()


# ============================================================
# INVALID VALUE HANDLING
# ============================================================

def replace_infinite_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace positive and negative infinity
    values with NaN.
    """

    cleaned_dataframe = (
        dataframe.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    return cleaned_dataframe


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def fill_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing numeric values using
    the median value of each feature.
    """

    cleaned_dataframe = (
        dataframe.copy()
    )

    for column in cleaned_dataframe.columns:

        median_value = (
            cleaned_dataframe[
                column
            ].median()
        )

        if pd.isna(
            median_value
        ):

            median_value = 0.0

        cleaned_dataframe[
            column
        ] = (
            cleaned_dataframe[
                column
            ].fillna(
                median_value
            )
        )

    return cleaned_dataframe


# ============================================================
# LOW VARIANCE FEATURES
# ============================================================

def get_low_variance_features(
    dataframe: pd.DataFrame,
    threshold: float = 1e-12,
) -> list[str]:
    """
    Find features that have almost no variation
    across handwriting samples.
    """

    low_variance_features = []

    variances = (
        dataframe.var()
    )

    for column, variance in (
        variances.items()
    ):

        if variance <= threshold:

            low_variance_features.append(
                column
            )

    return low_variance_features


# ============================================================
# REMOVE LOW VARIANCE FEATURES
# ============================================================

def remove_low_variance_features(
    dataframe: pd.DataFrame,
    threshold: float = 1e-12,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features with almost no variation.
    """

    low_variance_features = (
        get_low_variance_features(
            dataframe,
            threshold,
        )
    )

    filtered_dataframe = (
        dataframe.drop(
            columns=low_variance_features,
            errors="ignore",
        )
    )

    return (
        filtered_dataframe,
        low_variance_features,
    )


# ============================================================
# FEATURE CLEANING PIPELINE
# ============================================================

def clean_feature_dataframe(
    dataframe: pd.DataFrame,
    remove_low_variance: bool = False,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Clean and prepare handwriting features
    before normalization.
    """

    # --------------------------------------------------------
    # SELECT NUMERIC FEATURES
    # --------------------------------------------------------

    cleaned_dataframe = (
        select_numeric_features(
            dataframe
        )
    )

    original_feature_count = (
        cleaned_dataframe.shape[1]
    )

    # --------------------------------------------------------
    # REPLACE INFINITE VALUES
    # --------------------------------------------------------

    cleaned_dataframe = (
        replace_infinite_values(
            cleaned_dataframe
        )
    )

    # --------------------------------------------------------
    # FILL MISSING VALUES
    # --------------------------------------------------------

    cleaned_dataframe = (
        fill_missing_values(
            cleaned_dataframe
        )
    )

    removed_features = []

    # --------------------------------------------------------
    # OPTIONAL LOW VARIANCE REMOVAL
    # --------------------------------------------------------

    if remove_low_variance:

        (
            cleaned_dataframe,
            removed_features,
        ) = remove_low_variance_features(
            cleaned_dataframe
        )

    if cleaned_dataframe.empty:

        raise ValueError(
            "No usable features remain after cleaning."
        )

    metadata = {
        "original_feature_count": (
            original_feature_count
        ),
        "final_feature_count": (
            cleaned_dataframe.shape[1]
        ),
        "removed_low_variance_features": (
            removed_features
        ),
    }

    return (
        cleaned_dataframe,
        metadata,
    )


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(
    dataframe: pd.DataFrame,
) -> tuple[
    np.ndarray,
    StandardScaler,
]:
    """
    Normalize handwriting features using
    StandardScaler.
    """

    if dataframe.empty:

        raise ValueError(
            "Feature dataframe cannot be empty."
        )

    scaler = StandardScaler()

    normalized_features = (
        scaler.fit_transform(
            dataframe
        )
    )

    return (
        normalized_features,
        scaler,
    )


# ============================================================
# NORMALIZED DATAFRAME
# ============================================================

def create_normalized_dataframe(
    normalized_features: np.ndarray,
    feature_names: list[str],
    sample_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert normalized NumPy features back
    into a labeled Pandas DataFrame.
    """

    dataframe = pd.DataFrame(
        normalized_features,
        columns=feature_names,
    )

    if sample_names:

        if (
            len(sample_names)
            == len(dataframe)
        ):

            dataframe.index = (
                sample_names
            )

            dataframe.index.name = (
                "Sample"
            )

    return dataframe


# ============================================================
# STATISTICAL STYLE PROFILE
# ============================================================

def create_style_profile(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Create a statistical handwriting style profile.
    """

    if dataframe.empty:

        raise ValueError(
            "Feature dataframe cannot be empty."
        )

    profile = {
        "mean": (
            dataframe.mean()
            .to_dict()
        ),
        "std": (
            dataframe.std()
            .fillna(0)
            .to_dict()
        ),
        "min": (
            dataframe.min()
            .to_dict()
        ),
        "max": (
            dataframe.max()
            .to_dict()
        ),
    }

    return profile


# ============================================================
# FEATURE CONSISTENCY
# ============================================================

def calculate_feature_consistency(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calculate consistency for each handwriting feature.

    Higher score means greater consistency.
    """

    if dataframe.empty:

        raise ValueError(
            "Feature dataframe cannot be empty."
        )

    consistency = {}

    for column in dataframe.columns:

        values = dataframe[
            column
        ]

        mean_value = (
            values.mean()
        )

        std_value = (
            values.std()
        )

        # Handle single sample
        if pd.isna(
            std_value
        ):

            std_value = 0.0

        # Avoid division by zero
        if abs(
            mean_value
        ) < 1e-12:

            if std_value < 1e-12:

                score = 100.0

            else:

                score = 0.0

        else:

            coefficient_variation = (
                abs(
                    std_value
                    / mean_value
                )
            )

            score = max(
                0.0,
                100.0
                * (
                    1.0
                    - coefficient_variation
                ),
            )

        consistency[
            column
        ] = float(
            score
        )

    return consistency


# ============================================================
# OVERALL CONSISTENCY
# ============================================================

def calculate_overall_consistency(
    consistency: dict,
) -> float:
    """
    Calculate overall handwriting consistency score.
    """

    if not consistency:

        return 0.0

    values = list(
        consistency.values()
    )

    return float(
        np.mean(
            values
        )
    )


# ============================================================
# FEATURE METADATA
# ============================================================

def create_feature_metadata(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Create metadata describing the ML-ready
    handwriting feature dataset.
    """

    metadata = {
        "number_of_samples": int(
            dataframe.shape[0]
        ),
        "number_of_features": int(
            dataframe.shape[1]
        ),
        "feature_names": list(
            dataframe.columns
        ),
    }

    return metadata


# ============================================================
# COMPLETE STYLE MODEL
# ============================================================

def build_handwriting_style_model(
    feature_list: list[dict],
    sample_names: list[str] | None = None,
    remove_low_variance: bool = False,
) -> dict:
    """
    Build a complete Machine Learning-ready
    handwriting style model from multiple
    handwriting samples.
    """

    # --------------------------------------------------------
    # CREATE RAW DATAFRAME
    # --------------------------------------------------------

    raw_dataframe = (
        create_feature_dataframe(
            feature_list
        )
    )

    # --------------------------------------------------------
    # CLEAN FEATURES
    # --------------------------------------------------------

    (
        cleaned_dataframe,
        cleaning_metadata,
    ) = clean_feature_dataframe(
        raw_dataframe,
        remove_low_variance=(
            remove_low_variance
        ),
    )

    # --------------------------------------------------------
    # NORMALIZE FEATURES
    # --------------------------------------------------------

    (
        normalized_features,
        scaler,
    ) = normalize_features(
        cleaned_dataframe
    )

    # --------------------------------------------------------
    # CREATE NORMALIZED DATAFRAME
    # --------------------------------------------------------

    normalized_dataframe = (
        create_normalized_dataframe(
            normalized_features,
            list(
                cleaned_dataframe.columns
            ),
            sample_names,
        )
    )

    # --------------------------------------------------------
    # CREATE STYLE PROFILE
    # --------------------------------------------------------

    style_profile = (
        create_style_profile(
            cleaned_dataframe
        )
    )

    # --------------------------------------------------------
    # FEATURE CONSISTENCY
    # --------------------------------------------------------

    feature_consistency = (
        calculate_feature_consistency(
            cleaned_dataframe
        )
    )

    # --------------------------------------------------------
    # OVERALL CONSISTENCY
    # --------------------------------------------------------

    overall_consistency = (
        calculate_overall_consistency(
            feature_consistency
        )
    )

    # --------------------------------------------------------
    # FEATURE METADATA
    # --------------------------------------------------------

    feature_metadata = (
        create_feature_metadata(
            cleaned_dataframe
        )
    )

    return {
        "raw_dataframe": (
            raw_dataframe
        ),
        "cleaned_dataframe": (
            cleaned_dataframe
        ),
        "normalized_features": (
            normalized_features
        ),
        "normalized_dataframe": (
            normalized_dataframe
        ),
        "scaler": scaler,
        "style_profile": (
            style_profile
        ),
        "feature_consistency": (
            feature_consistency
        ),
        "overall_consistency": (
            overall_consistency
        ),
        "cleaning_metadata": (
            cleaning_metadata
        ),
        "feature_metadata": (
            feature_metadata
        ),
    }