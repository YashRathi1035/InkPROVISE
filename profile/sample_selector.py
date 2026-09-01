# ============================================================
# HANDWRITING SAMPLE SELECTION
# Phase 5B
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd


# ============================================================
# DEFAULT SETTINGS
# ============================================================

MIN_QUALITY_SCORE = 50.0

MIN_SIMILARITY_SCORE = 0.30


# ============================================================
# QUALITY DECISION
# ============================================================

def evaluate_quality(
    quality_score: float,
    minimum_quality: float = MIN_QUALITY_SCORE,
) -> bool:
    """
    Check whether the handwriting image
    has acceptable image quality.
    """

    return (
        quality_score
        >= minimum_quality
    )


# ============================================================
# SIMILARITY DECISION
# ============================================================

def evaluate_similarity(
    similarity_score: float,
    minimum_similarity: float = (
        MIN_SIMILARITY_SCORE
    ),
) -> bool:
    """
    Check whether the handwriting sample
    is sufficiently similar to the other
    handwriting samples.
    """

    return (
        similarity_score
        >= minimum_similarity
    )


# ============================================================
# SAMPLE DECISION
# ============================================================

def decide_sample_status(
    quality_score: float,
    similarity_score: float,
    is_outlier: bool,
    minimum_quality: float = (
        MIN_QUALITY_SCORE
    ),
    minimum_similarity: float = (
        MIN_SIMILARITY_SCORE
    ),
) -> dict:
    """
    Decide whether a handwriting sample
    should be accepted or rejected.

    Decision factors:

    1. Image quality
    2. Handwriting similarity
    3. Outlier detection
    """

    quality_pass = (
        evaluate_quality(
            quality_score,
            minimum_quality,
        )
    )

    similarity_pass = (
        evaluate_similarity(
            similarity_score,
            minimum_similarity,
        )
    )

    # --------------------------------------------------------
    # ACCEPT SAMPLE
    # --------------------------------------------------------

    if (
        quality_pass
        and similarity_pass
        and not is_outlier
    ):

        return {
            "status": "Accepted",
            "accepted": True,
            "reason": (
                "Good image quality and "
                "consistent handwriting style."
            ),
        }

    # --------------------------------------------------------
    # REJECT OUTLIER
    # --------------------------------------------------------

    if is_outlier:

        return {
            "status": "Rejected",
            "accepted": False,
            "reason": (
                "The handwriting pattern differs "
                "significantly from the profile."
            ),
        }

    # --------------------------------------------------------
    # REJECT LOW QUALITY
    # --------------------------------------------------------

    if not quality_pass:

        return {
            "status": "Rejected",
            "accepted": False,
            "reason": (
                "Image quality is below the "
                "minimum acceptable threshold."
            ),
        }

    # --------------------------------------------------------
    # REJECT LOW SIMILARITY
    # --------------------------------------------------------

    if not similarity_pass:

        return {
            "status": "Rejected",
            "accepted": False,
            "reason": (
                "The handwriting sample has low "
                "similarity with the other samples."
            ),
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {
        "status": "Rejected",
        "accepted": False,
        "reason": (
            "The sample did not meet the "
            "profile requirements."
        ),
    }


# ============================================================
# BUILD SAMPLE SELECTION TABLE
# ============================================================

def build_sample_selection_dataframe(
    sample_names: list,
    quality_results: list[dict],
    average_similarity_dataframe: pd.DataFrame,
    outlier_dataframe: pd.DataFrame,
    minimum_quality: float = (
        MIN_QUALITY_SCORE
    ),
    minimum_similarity: float = (
        MIN_SIMILARITY_SCORE
    ),
) -> pd.DataFrame:
    """
    Combine quality analysis, similarity
    analysis and outlier detection into
    one sample selection DataFrame.
    """

    rows = []

    # --------------------------------------------------------
    # CREATE LOOKUP DICTIONARIES
    # --------------------------------------------------------

    similarity_lookup = {}

    if (
        average_similarity_dataframe
        is not None
        and not average_similarity_dataframe.empty
    ):

        for _, row in (
            average_similarity_dataframe.iterrows()
        ):

            similarity_lookup[
                row["Sample"]
            ] = row[
                "Average Similarity"
            ]

    outlier_lookup = {}

    if (
        outlier_dataframe
        is not None
        and not outlier_dataframe.empty
    ):

        for _, row in (
            outlier_dataframe.iterrows()
        ):

            outlier_lookup[
                row["Sample"]
            ] = bool(
                row[
                    "Is Outlier"
                ]
            )

    # --------------------------------------------------------
    # PROCESS EACH SAMPLE
    # --------------------------------------------------------

    for index, sample_name in enumerate(
        sample_names
    ):

        # Quality information

        if (
            index
            < len(
                quality_results
            )
        ):

            quality_result = (
                quality_results[
                    index
                ]
            )

        else:

            quality_result = {}

        quality_score = float(
            quality_result.get(
                "quality_score",
                0.0,
            )
        )

        quality_label = (
            quality_result.get(
                "quality_label",
                "Unknown",
            )
        )

        # Similarity information

        similarity_score = float(
            similarity_lookup.get(
                sample_name,
                0.0,
            )
        )

        # Outlier information

        is_outlier = bool(
            outlier_lookup.get(
                sample_name,
                False,
            )
        )

        # Decision

        decision = (
            decide_sample_status(
                quality_score=
                    quality_score,

                similarity_score=
                    similarity_score,

                is_outlier=
                    is_outlier,

                minimum_quality=
                    minimum_quality,

                minimum_similarity=
                    minimum_similarity,
            )
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        rows.append(
            {
                "Sample": sample_name,

                "Quality Score": (
                    quality_score
                ),

                "Quality Label": (
                    quality_label
                ),

                "Average Similarity": (
                    similarity_score
                ),

                "Average Similarity (%)": (
                    similarity_score
                    * 100
                ),

                "Is Outlier": (
                    is_outlier
                ),

                "Status": (
                    decision[
                        "status"
                    ]
                ),

                "Accepted": (
                    decision[
                        "accepted"
                    ]
                ),

                "Reason": (
                    decision[
                        "reason"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAMPLE SELECTION SUMMARY
# ============================================================

def get_sample_selection_summary(
    selection_dataframe: pd.DataFrame,
) -> dict:
    """
    Generate a summary of accepted and
    rejected handwriting samples.
    """

    if (
        selection_dataframe is None
        or selection_dataframe.empty
    ):

        return {
            "total_samples": 0,
            "accepted_samples": 0,
            "rejected_samples": 0,
            "acceptance_rate": 0.0,
        }

    total_samples = len(
        selection_dataframe
    )

    accepted_samples = int(
        selection_dataframe[
            "Accepted"
        ].sum()
    )

    rejected_samples = (
        total_samples
        - accepted_samples
    )

    acceptance_rate = (
        accepted_samples
        / total_samples
        * 100
        if total_samples > 0
        else 0.0
    )

    return {
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
    }


# ============================================================
# GET ACCEPTED SAMPLES
# ============================================================

def get_accepted_sample_names(
    selection_dataframe: pd.DataFrame,
) -> list:
    """
    Return names of handwriting samples
    that were accepted.
    """

    if (
        selection_dataframe is None
        or selection_dataframe.empty
    ):

        return []

    accepted_dataframe = (
        selection_dataframe[
            selection_dataframe[
                "Accepted"
            ]
        ]
    )

    return list(
        accepted_dataframe[
            "Sample"
        ]
    )


# ============================================================
# GET REJECTED SAMPLES
# ============================================================

def get_rejected_sample_names(
    selection_dataframe: pd.DataFrame,
) -> list:
    """
    Return names of handwriting samples
    that were rejected.
    """

    if (
        selection_dataframe is None
        or selection_dataframe.empty
    ):

        return []

    rejected_dataframe = (
        selection_dataframe[
            ~selection_dataframe[
                "Accepted"
            ]
        ]
    )

    return list(
        rejected_dataframe[
            "Sample"
        ]
    )