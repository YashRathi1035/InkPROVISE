# ============================================================
# HANDWRITING SIMILARITY MODEL
# Phase 4D
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity,
)


# ============================================================
# SAMPLE SIMILARITY MATRIX
# ============================================================

def calculate_similarity_matrix(
    normalized_features: np.ndarray,
    sample_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate cosine similarity between
    all handwriting samples.
    """

    if (
        normalized_features is None
        or len(normalized_features) == 0
    ):

        raise ValueError(
            "Normalized features cannot be empty."
        )

    similarity_matrix = (
        cosine_similarity(
            normalized_features
        )
    )

    number_of_samples = (
        len(
            normalized_features
        )
    )

    if (
        sample_names
        and len(sample_names)
        == number_of_samples
    ):

        labels = sample_names

    else:

        labels = [
            f"Sample {index + 1}"
            for index in range(
                number_of_samples
            )
        ]

    dataframe = pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels,
    )

    return dataframe


# ============================================================
# AVERAGE SAMPLE SIMILARITY
# ============================================================

def calculate_average_sample_similarity(
    similarity_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average similarity of every
    sample with all other samples.
    """

    if similarity_dataframe.empty:

        return pd.DataFrame(
            columns=[
                "Sample",
                "Average Similarity",
            ]
        )

    similarity_values = (
        similarity_dataframe.to_numpy()
    )

    sample_names = list(
        similarity_dataframe.index
    )

    results = []

    for index, sample_name in enumerate(
        sample_names
    ):

        other_similarities = np.delete(
            similarity_values[
                index
            ],
            index,
        )

        if len(
            other_similarities
        ) == 0:

            average_similarity = 1.0

        else:

            average_similarity = float(
                np.mean(
                    other_similarities
                )
            )

        results.append(
            {
                "Sample": sample_name,
                "Average Similarity": (
                    average_similarity
                ),
            }
        )

    dataframe = pd.DataFrame(
        results
    )

    dataframe[
        "Average Similarity (%)"
    ] = (
        dataframe[
            "Average Similarity"
        ]
        * 100
    )

    return dataframe


# ============================================================
# DISTANCE FROM STYLE CENTER
# ============================================================

def calculate_distance_from_style_center(
    normalized_features: np.ndarray,
    sample_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate Euclidean distance of each sample
    from the average handwriting style.
    """

    if (
        normalized_features is None
        or len(normalized_features) == 0
    ):

        raise ValueError(
            "Normalized features cannot be empty."
        )

    style_center = np.mean(
        normalized_features,
        axis=0,
    )

    distances = np.linalg.norm(
        normalized_features
        - style_center,
        axis=1,
    )

    number_of_samples = len(
        normalized_features
    )

    if (
        sample_names
        and len(sample_names)
        == number_of_samples
    ):

        labels = sample_names

    else:

        labels = [
            f"Sample {index + 1}"
            for index in range(
                number_of_samples
            )
        ]

    dataframe = pd.DataFrame(
        {
            "Sample": labels,
            "Distance From Style Center": (
                distances
            ),
        }
    )

    return dataframe


# ============================================================
# OUTLIER DETECTION
# ============================================================

def detect_handwriting_outliers(
    distance_dataframe: pd.DataFrame,
    threshold_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Detect handwriting samples that are unusually
    far from the average handwriting style.

    Uses mean + threshold_multiplier * std.
    """

    if distance_dataframe.empty:

        return distance_dataframe.copy()

    dataframe = (
        distance_dataframe.copy()
    )

    distances = dataframe[
        "Distance From Style Center"
    ]

    mean_distance = (
        distances.mean()
    )

    std_distance = (
        distances.std()
    )

    if pd.isna(
        std_distance
    ):

        std_distance = 0.0

    outlier_threshold = (
        mean_distance
        + (
            threshold_multiplier
            * std_distance
        )
    )

    dataframe[
        "Outlier Threshold"
    ] = outlier_threshold

    dataframe[
        "Is Outlier"
    ] = (
        dataframe[
            "Distance From Style Center"
        ]
        > outlier_threshold
    )

    return dataframe


# ============================================================
# OUTLIER SUMMARY
# ============================================================

def get_outlier_summary(
    outlier_dataframe: pd.DataFrame,
) -> dict:
    """
    Generate a summary of detected outliers.
    """

    if outlier_dataframe.empty:

        return {
            "total_samples": 0,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "outlier_samples": [],
        }

    total_samples = len(
        outlier_dataframe
    )

    outliers = outlier_dataframe[
        outlier_dataframe[
            "Is Outlier"
        ]
    ]

    outlier_count = len(
        outliers
    )

    outlier_ratio = (
        outlier_count
        / total_samples
        if total_samples > 0
        else 0.0
    )

    outlier_samples = list(
        outliers[
            "Sample"
        ]
    )

    return {
        "total_samples": (
            total_samples
        ),
        "outlier_count": (
            outlier_count
        ),
        "outlier_ratio": (
            outlier_ratio
        ),
        "outlier_samples": (
            outlier_samples
        ),
    }


# ============================================================
# PROFILE CONFIDENCE
# ============================================================

def calculate_profile_confidence(
    overall_consistency: float,
    average_similarity_dataframe: pd.DataFrame,
    outlier_summary: dict,
) -> float:
    """
    Calculate overall confidence in the
    handwriting style profile.

    Confidence is based on:

    - Handwriting consistency
    - Average similarity
    - Number of outliers
    """

    consistency_score = (
        overall_consistency
        / 100
    )

    if (
        average_similarity_dataframe.empty
    ):

        similarity_score = 0.0

    else:

        similarity_score = float(
            average_similarity_dataframe[
                "Average Similarity"
            ].mean()
        )

        # Cosine similarity can be negative.
        similarity_score = max(
            0.0,
            min(
                1.0,
                similarity_score,
            ),
        )

    outlier_ratio = (
        outlier_summary.get(
            "outlier_ratio",
            0.0,
        )
    )

    outlier_score = (
        1.0
        - outlier_ratio
    )

    # Weighted confidence score

    confidence = (
        0.40
        * consistency_score
        + 0.40
        * similarity_score
        + 0.20
        * outlier_score
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    return float(
        confidence * 100
    )


# ============================================================
# COMPLETE SIMILARITY ANALYSIS
# ============================================================

def analyze_handwriting_similarity(
    normalized_features: np.ndarray,
    sample_names: list[str] | None,
    overall_consistency: float,
) -> dict:
    """
    Run the complete handwriting similarity
    and outlier analysis pipeline.
    """

    # --------------------------------------------------------
    # SIMILARITY MATRIX
    # --------------------------------------------------------

    similarity_matrix = (
        calculate_similarity_matrix(
            normalized_features,
            sample_names,
        )
    )

    # --------------------------------------------------------
    # AVERAGE SIMILARITY
    # --------------------------------------------------------

    average_similarity = (
        calculate_average_sample_similarity(
            similarity_matrix
        )
    )

    # --------------------------------------------------------
    # DISTANCE FROM STYLE CENTER
    # --------------------------------------------------------

    distance_dataframe = (
        calculate_distance_from_style_center(
            normalized_features,
            sample_names,
        )
    )

    # --------------------------------------------------------
    # OUTLIER DETECTION
    # --------------------------------------------------------

    outlier_dataframe = (
        detect_handwriting_outliers(
            distance_dataframe
        )
    )

    # --------------------------------------------------------
    # OUTLIER SUMMARY
    # --------------------------------------------------------

    outlier_summary = (
        get_outlier_summary(
            outlier_dataframe
        )
    )

    # --------------------------------------------------------
    # PROFILE CONFIDENCE
    # --------------------------------------------------------

    profile_confidence = (
        calculate_profile_confidence(
            overall_consistency,
            average_similarity,
            outlier_summary,
        )
    )

    return {
        "similarity_matrix": (
            similarity_matrix
        ),
        "average_similarity": (
            average_similarity
        ),
        "distance_dataframe": (
            distance_dataframe
        ),
        "outlier_dataframe": (
            outlier_dataframe
        ),
        "outlier_summary": (
            outlier_summary
        ),
        "profile_confidence": (
            profile_confidence
        ),
    }