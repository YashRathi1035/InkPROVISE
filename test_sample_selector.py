import pandas as pd

from profile.sample_selector import (
    build_sample_selection_dataframe,
    get_sample_selection_summary,
    get_accepted_sample_names,
    get_rejected_sample_names,
)


# ============================================================
# SAMPLE NAMES
# ============================================================

sample_names = [
    "Page_1.jpg",
    "Page_2.jpg",
    "Page_3.jpg",
    "Different_Page.jpg",
]


# ============================================================
# QUALITY RESULTS
# ============================================================

quality_results = [
    {
        "quality_score": 92.0,
        "quality_label": "Excellent",
    },

    {
        "quality_score": 81.0,
        "quality_label": "Good",
    },

    {
        "quality_score": 73.0,
        "quality_label": "Good",
    },

    {
        "quality_score": 88.0,
        "quality_label": "Excellent",
    },
]


# ============================================================
# SIMILARITY RESULTS
# ============================================================

average_similarity_dataframe = (
    pd.DataFrame(
        {
            "Sample": sample_names,

            "Average Similarity": [
                0.91,
                0.88,
                0.86,
                0.22,
            ],
        }
    )
)


# ============================================================
# OUTLIER RESULTS
# ============================================================

outlier_dataframe = pd.DataFrame(
    {
        "Sample": sample_names,

        "Is Outlier": [
            False,
            False,
            False,
            True,
        ],
    }
)


# ============================================================
# BUILD SELECTION TABLE
# ============================================================

selection_dataframe = (
    build_sample_selection_dataframe(
        sample_names=
            sample_names,

        quality_results=
            quality_results,

        average_similarity_dataframe=
            average_similarity_dataframe,

        outlier_dataframe=
            outlier_dataframe,
    )
)


print(
    "\nSAMPLE SELECTION RESULTS\n"
)

print(
    selection_dataframe
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    get_sample_selection_summary(
        selection_dataframe
    )
)

print(
    "\nSUMMARY\n"
)

print(
    summary
)


# ============================================================
# ACCEPTED SAMPLES
# ============================================================

accepted_samples = (
    get_accepted_sample_names(
        selection_dataframe
    )
)

print(
    "\nACCEPTED SAMPLES\n"
)

print(
    accepted_samples
)


# ============================================================
# REJECTED SAMPLES
# ============================================================

rejected_samples = (
    get_rejected_sample_names(
        selection_dataframe
    )
)

print(
    "\nREJECTED SAMPLES\n"
)

print(
    rejected_samples
)