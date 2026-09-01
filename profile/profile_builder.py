# ============================================================
# HANDWRITING PROFILE BUILDER
# Phase 5C
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd


# ============================================================
# READINESS THRESHOLDS
# ============================================================

MINIMUM_ACCEPTED_SAMPLES = 3

RECOMMENDED_ACCEPTED_SAMPLES = 5

MINIMUM_PROFILE_CONFIDENCE = 60.0

READY_PROFILE_CONFIDENCE = 75.0

MINIMUM_AVERAGE_QUALITY = 50.0

READY_AVERAGE_QUALITY = 70.0

MINIMUM_AVERAGE_SIMILARITY = 0.30

READY_AVERAGE_SIMILARITY = 0.55


# ============================================================
# CALCULATE AVERAGE QUALITY
# ============================================================

def calculate_average_quality(
    selection_dataframe: pd.DataFrame,
) -> float:
    """
    Calculate average quality score of
    accepted handwriting samples.
    """

    if (
        selection_dataframe is None
        or selection_dataframe.empty
    ):

        return 0.0

    accepted_dataframe = (
        selection_dataframe[
            selection_dataframe[
                "Accepted"
            ]
        ]
    )

    if accepted_dataframe.empty:

        return 0.0

    average_quality = (
        accepted_dataframe[
            "Quality Score"
        ].mean()
    )

    return float(
        average_quality
    )


# ============================================================
# CALCULATE AVERAGE SIMILARITY
# ============================================================

def calculate_average_similarity(
    selection_dataframe: pd.DataFrame,
) -> float:
    """
    Calculate average similarity of
    accepted handwriting samples.
    """

    if (
        selection_dataframe is None
        or selection_dataframe.empty
    ):

        return 0.0

    accepted_dataframe = (
        selection_dataframe[
            selection_dataframe[
                "Accepted"
            ]
        ]
    )

    if accepted_dataframe.empty:

        return 0.0

    average_similarity = (
        accepted_dataframe[
            "Average Similarity"
        ].mean()
    )

    return float(
        average_similarity
    )


# ============================================================
# CALCULATE FINAL PROFILE CONFIDENCE
# ============================================================

def calculate_final_profile_confidence(
    similarity_profile_confidence: float,
    average_quality: float,
    acceptance_rate: float,
    accepted_samples: int,
) -> float:
    """
    Calculate final handwriting profile
    confidence.

    The score considers:

    - ML similarity confidence
    - Average sample quality
    - Sample acceptance rate
    - Number of accepted samples
    """

    # --------------------------------------------------------
    # NORMALIZE INPUT VALUES
    # --------------------------------------------------------

    similarity_confidence_score = (
        similarity_profile_confidence
        / 100.0
    )

    quality_score = (
        average_quality
        / 100.0
    )

    acceptance_score = (
        acceptance_rate
        / 100.0
    )

    # --------------------------------------------------------
    # SAMPLE COUNT SCORE
    # --------------------------------------------------------

    sample_count_score = min(
        accepted_samples
        / RECOMMENDED_ACCEPTED_SAMPLES,
        1.0,
    )

    # --------------------------------------------------------
    # FINAL WEIGHTED CONFIDENCE
    # --------------------------------------------------------

    confidence = (
        0.40
        * similarity_confidence_score

        + 0.25
        * quality_score

        + 0.20
        * acceptance_score

        + 0.15
        * sample_count_score
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    return float(
        confidence
        * 100
    )


# ============================================================
# DETERMINE PROFILE STATUS
# ============================================================

def determine_profile_status(
    accepted_samples: int,
    average_quality: float,
    average_similarity: float,
    profile_confidence: float,
) -> dict:
    """
    Determine whether the handwriting
    profile is ready for generation.
    """

    # --------------------------------------------------------
    # NOT ENOUGH SAMPLES
    # --------------------------------------------------------

    if (
        accepted_samples
        < MINIMUM_ACCEPTED_SAMPLES
    ):

        return {
            "status": (
                "NOT READY"
            ),

            "emoji": "🔴",

            "message": (
                "Not enough accepted handwriting "
                "samples. Upload more samples."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # LOW QUALITY
    # --------------------------------------------------------

    if (
        average_quality
        < MINIMUM_AVERAGE_QUALITY
    ):

        return {
            "status": (
                "NEEDS BETTER QUALITY"
            ),

            "emoji": "🟠",

            "message": (
                "The accepted samples have low "
                "average image quality. Upload "
                "clearer handwriting images."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # LOW SIMILARITY
    # --------------------------------------------------------

    if (
        average_similarity
        < MINIMUM_AVERAGE_SIMILARITY
    ):

        return {
            "status": (
                "INCONSISTENT HANDWRITING"
            ),

            "emoji": "🟠",

            "message": (
                "The accepted samples have low "
                "similarity. Upload samples with "
                "a more consistent handwriting style."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if (
        profile_confidence
        < MINIMUM_PROFILE_CONFIDENCE
    ):

        return {
            "status": (
                "LOW PROFILE CONFIDENCE"
            ),

            "emoji": "🟠",

            "message": (
                "The handwriting profile needs "
                "more reliable samples before "
                "generation."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # NEEDS MORE SAMPLES
    # --------------------------------------------------------

    if (
        accepted_samples
        < RECOMMENDED_ACCEPTED_SAMPLES
    ):

        return {
            "status": (
                "NEEDS MORE SAMPLES"
            ),

            "emoji": "🟡",

            "message": (
                "The profile is usable, but "
                "uploading more handwriting "
                "samples will improve accuracy."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # QUALITY BELOW READY LEVEL
    # --------------------------------------------------------

    if (
        average_quality
        < READY_AVERAGE_QUALITY
    ):

        return {
            "status": (
                "NEEDS BETTER QUALITY"
            ),

            "emoji": "🟡",

            "message": (
                "The handwriting profile is usable, "
                "but better image quality is "
                "recommended before generation."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # SIMILARITY BELOW READY LEVEL
    # --------------------------------------------------------

    if (
        average_similarity
        < READY_AVERAGE_SIMILARITY
    ):

        return {
            "status": (
                "NEEDS MORE CONSISTENCY"
            ),

            "emoji": "🟡",

            "message": (
                "The samples are acceptable, but "
                "their handwriting patterns could "
                "be more consistent."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # READY CONFIDENCE CHECK
    # --------------------------------------------------------

    if (
        profile_confidence
        < READY_PROFILE_CONFIDENCE
    ):

        return {
            "status": (
                "PROFILE IMPROVEMENT RECOMMENDED"
            ),

            "emoji": "🟡",

            "message": (
                "The profile is mostly reliable, "
                "but additional high-quality samples "
                "can improve generation accuracy."
            ),

            "ready_for_generation": (
                False
            ),
        }

    # --------------------------------------------------------
    # READY FOR GENERATION
    # --------------------------------------------------------

    return {
        "status": (
            "READY FOR GENERATION"
        ),

        "emoji": "🟢",

        "message": (
            "The handwriting profile has enough "
            "high-quality and consistent samples "
            "for the next generation stage."
        ),

        "ready_for_generation": (
            True
        ),
    }


# ============================================================
# BUILD COMPLETE HANDWRITING PROFILE
# ============================================================

def build_handwriting_profile(
    profile_name: str,
    selection_dataframe: pd.DataFrame,
    selection_summary: dict,
    overall_consistency: float,
    similarity_profile_confidence: float,
) -> dict:
    """
    Build the complete intelligent
    handwriting profile.
    """

    # --------------------------------------------------------
    # BASIC SAMPLE INFORMATION
    # --------------------------------------------------------

    total_samples = (
        selection_summary.get(
            "total_samples",
            0,
        )
    )

    accepted_samples = (
        selection_summary.get(
            "accepted_samples",
            0,
        )
    )

    rejected_samples = (
        selection_summary.get(
            "rejected_samples",
            0,
        )
    )

    acceptance_rate = (
        selection_summary.get(
            "acceptance_rate",
            0.0,
        )
    )

    # --------------------------------------------------------
    # QUALITY METRICS
    # --------------------------------------------------------

    average_quality = (
        calculate_average_quality(
            selection_dataframe
        )
    )

    # --------------------------------------------------------
    # SIMILARITY METRICS
    # --------------------------------------------------------

    average_similarity = (
        calculate_average_similarity(
            selection_dataframe
        )
    )

    # --------------------------------------------------------
    # FINAL PROFILE CONFIDENCE
    # --------------------------------------------------------

    final_profile_confidence = (
        calculate_final_profile_confidence(
            similarity_profile_confidence=
                similarity_profile_confidence,

            average_quality=
                average_quality,

            acceptance_rate=
                acceptance_rate,

            accepted_samples=
                accepted_samples,
        )
    )

    # --------------------------------------------------------
    # GENERATION READINESS
    # --------------------------------------------------------

    readiness = (
        determine_profile_status(
            accepted_samples=
                accepted_samples,

            average_quality=
                average_quality,

            average_similarity=
                average_similarity,

            profile_confidence=
                final_profile_confidence,
        )
    )

    # --------------------------------------------------------
    # ACCEPTED / REJECTED NAMES
    # --------------------------------------------------------

    if (
        selection_dataframe is None
        or selection_dataframe.empty
    ):

        accepted_sample_names = []

        rejected_sample_names = []

    else:

        accepted_sample_names = list(
            selection_dataframe[
                selection_dataframe[
                    "Accepted"
                ]
            ][
                "Sample"
            ]
        )

        rejected_sample_names = list(
            selection_dataframe[
                ~selection_dataframe[
                    "Accepted"
                ]
            ][
                "Sample"
            ]
        )

    # --------------------------------------------------------
    # FINAL PROFILE
    # --------------------------------------------------------

    profile = {

        # Basic information

        "profile_name": (
            profile_name
        ),

        # Sample information

        "total_samples": (
            total_samples
        ),

        "accepted_samples": (
            accepted_samples
        ),

        "rejected_samples": (
            rejected_samples
        ),

        "acceptance_rate": (
            acceptance_rate
        ),

        # Sample names

        "accepted_sample_names": (
            accepted_sample_names
        ),

        "rejected_sample_names": (
            rejected_sample_names
        ),

        # Handwriting metrics

        "average_quality": (
            average_quality
        ),

        "overall_consistency": (
            overall_consistency
        ),

        "average_similarity": (
            average_similarity
        ),

        "similarity_profile_confidence": (
            similarity_profile_confidence
        ),

        "final_profile_confidence": (
            final_profile_confidence
        ),

        # Readiness

        "status": (
            readiness[
                "status"
            ]
        ),

        "status_emoji": (
            readiness[
                "emoji"
            ]
        ),

        "status_message": (
            readiness[
                "message"
            ]
        ),

        "ready_for_generation": (
            readiness[
                "ready_for_generation"
            ]
        ),
    }

    return profile


# ============================================================
# CREATE PROFILE SUMMARY DATAFRAME
# ============================================================

def create_profile_summary_dataframe(
    profile: dict,
) -> pd.DataFrame:
    """
    Convert the handwriting profile into
    a clean summary DataFrame for display.
    """

    rows = [

        {
            "Metric": (
                "Profile Name"
            ),

            "Value": (
                profile.get(
                    "profile_name",
                    "",
                )
            ),
        },

        {
            "Metric": (
                "Total Samples"
            ),

            "Value": (
                profile.get(
                    "total_samples",
                    0,
                )
            ),
        },

        {
            "Metric": (
                "Accepted Samples"
            ),

            "Value": (
                profile.get(
                    "accepted_samples",
                    0,
                )
            ),
        },

        {
            "Metric": (
                "Rejected Samples"
            ),

            "Value": (
                profile.get(
                    "rejected_samples",
                    0,
                )
            ),
        },

        {
            "Metric": (
                "Acceptance Rate"
            ),

            "Value": (
                f"{profile.get('acceptance_rate', 0.0):.1f}%"
            ),
        },

        {
            "Metric": (
                "Average Image Quality"
            ),

            "Value": (
                f"{profile.get('average_quality', 0.0):.1f}%"
            ),
        },

        {
            "Metric": (
                "Handwriting Consistency"
            ),

            "Value": (
                f"{profile.get('overall_consistency', 0.0):.1f}%"
            ),
        },

        {
            "Metric": (
                "Average Similarity"
            ),

            "Value": (
                f"{profile.get('average_similarity', 0.0) * 100:.1f}%"
            ),
        },

        {
            "Metric": (
                "Similarity Confidence"
            ),

            "Value": (
                f"{profile.get('similarity_profile_confidence', 0.0):.1f}%"
            ),
        },

        {
            "Metric": (
                "Final Profile Confidence"
            ),

            "Value": (
                f"{profile.get('final_profile_confidence', 0.0):.1f}%"
            ),
        },

        {
            "Metric": (
                "Generation Status"
            ),

            "Value": (
                f"{profile.get('status_emoji', '')} "
                f"{profile.get('status', '')}"
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )