# ============================================================
# HANDWRITING GENERATION DATASET BUILDER
# Phase 6A
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CREATE GENERATION DIRECTORIES
# ============================================================

def create_generation_directories(
    base_directory: Path,
) -> dict:
    """
    Create the directory structure used
    for handwriting generation data.
    """

    generation_directory = (
        base_directory
        / "generation"
    )

    accepted_samples_directory = (
        generation_directory
        / "accepted_samples"
    )

    lines_directory = (
        generation_directory
        / "lines"
    )

    words_directory = (
        generation_directory
        / "words"
    )

    characters_directory = (
        generation_directory
        / "characters"
    )

    metadata_directory = (
        generation_directory
        / "metadata"
    )

    directories = {
        "generation": (
            generation_directory
        ),

        "accepted_samples": (
            accepted_samples_directory
        ),

        "lines": (
            lines_directory
        ),

        "words": (
            words_directory
        ),

        "characters": (
            characters_directory
        ),

        "metadata": (
            metadata_directory
        ),
    }

    for directory in directories.values():

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


# ============================================================
# CONVERT UPLOADED SAMPLE TO IMAGE
# ============================================================

def read_uploaded_sample(
    uploaded_file,
) -> np.ndarray:
    """
    Convert an uploaded handwriting
    sample into an OpenCV image.
    """

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.getvalue()
        ),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise ValueError(
            "Unable to read handwriting image."
        )

    return image


# ============================================================
# CREATE SAFE FILE NAME
# ============================================================

def create_safe_filename(
    filename: str,
    index: int,
) -> str:
    """
    Create a consistent filename for
    generation dataset samples.
    """

    extension = (
        Path(
            filename
        ).suffix.lower()
    )

    if extension == "":

        extension = ".png"

    return (
        f"sample_{index + 1:03d}"
        f"{extension}"
    )


# ============================================================
# SAVE ACCEPTED SAMPLE
# ============================================================

def save_accepted_sample(
    image: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save an accepted handwriting sample.
    """

    success = cv2.imwrite(
        str(
            output_path
        ),
        image,
    )

    if not success:

        raise RuntimeError(
            "Unable to save handwriting sample."
        )


# ============================================================
# BUILD ACCEPTED DATASET
# ============================================================

def build_accepted_dataset(
    uploaded_samples: list,
    sample_names: list,
    accepted_sample_names: list,
    output_directory: Path,
) -> list:
    """
    Build a clean generation dataset using
    only accepted handwriting samples.

    Returns metadata for saved samples.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted_set = set(
        accepted_sample_names
    )

    dataset_metadata = []

    for index, uploaded_sample in enumerate(
        uploaded_samples
    ):

        if (
            index
            >= len(
                sample_names
            )
        ):

            continue

        sample_name = (
            sample_names[
                index
            ]
        )

        # ----------------------------------------------------
        # SKIP REJECTED SAMPLES
        # ----------------------------------------------------

        if (
            sample_name
            not in accepted_set
        ):

            continue

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = (
            read_uploaded_sample(
                uploaded_sample
            )
        )

        # ----------------------------------------------------
        # CREATE OUTPUT NAME
        # ----------------------------------------------------

        safe_filename = (
            create_safe_filename(
                sample_name,
                index,
            )
        )

        output_path = (
            output_directory
            / safe_filename
        )

        # ----------------------------------------------------
        # SAVE SAMPLE
        # ----------------------------------------------------

        save_accepted_sample(
            image=image,
            output_path=output_path,
        )

        height, width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # STORE METADATA
        # ----------------------------------------------------

        dataset_metadata.append(
            {
                "original_name": (
                    sample_name
                ),

                "dataset_name": (
                    safe_filename
                ),

                "path": (
                    str(
                        output_path
                    )
                ),

                "width": (
                    width
                ),

                "height": (
                    height
                ),

                "status": (
                    "accepted"
                ),
            }
        )

    return dataset_metadata


# ============================================================
# CREATE DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    dataset_metadata: list,
) -> dict:
    """
    Create a summary of the accepted
    handwriting generation dataset.
    """

    total_samples = len(
        dataset_metadata
    )

    if total_samples == 0:

        return {
            "total_samples": 0,
            "average_width": 0.0,
            "average_height": 0.0,
        }

    average_width = (
        sum(
            item["width"]
            for item
            in dataset_metadata
        )
        / total_samples
    )

    average_height = (
        sum(
            item["height"]
            for item
            in dataset_metadata
        )
        / total_samples
    )

    return {
        "total_samples": (
            total_samples
        ),

        "average_width": (
            float(
                average_width
            )
        ),

        "average_height": (
            float(
                average_height
            )
        ),
    }