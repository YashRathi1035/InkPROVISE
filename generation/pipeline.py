# ============================================================
# COMPLETE HANDWRITING DATASET PIPELINE
# Phase 6E
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import csv
import json

import cv2
import numpy as np

from generation.line_extraction import (
    extract_handwritten_lines,
    save_extracted_lines,
)

from generation.word_extraction import (
    extract_handwritten_words,
    save_extracted_words,
)

from generation.normalization import (
    normalize_handwriting_word,
    save_normalized_word,
)


# ============================================================
# CREATE DATASET DIRECTORIES
# ============================================================

def create_dataset_directories(
    dataset_directory: Path,
) -> dict:
    """
    Create all directories required for
    the handwriting generation dataset.
    """

    lines_directory = (
        dataset_directory
        / "lines"
    )

    words_directory = (
        dataset_directory
        / "words"
    )

    normalized_words_directory = (
        dataset_directory
        / "normalized_words"
    )

    for directory in [
        dataset_directory,
        lines_directory,
        words_directory,
        normalized_words_directory,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return {
        "dataset_directory":
            dataset_directory,

        "lines_directory":
            lines_directory,

        "words_directory":
            words_directory,

        "normalized_words_directory":
            normalized_words_directory,
    }


# ============================================================
# LOAD HANDWRITING IMAGE
# ============================================================

def load_handwriting_image(
    image_path: Path,
) -> np.ndarray:
    """
    Load a handwriting image.
    """

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:

        raise ValueError(
            f"Unable to load image:\n"
            f"{image_path}"
        )

    return image


# ============================================================
# PROCESS SINGLE HANDWRITING PAGE
# ============================================================

def process_handwriting_page(
    image_path: Path,
    dataset_directories: dict,
) -> dict:
    """
    Process one handwriting page through
    the complete pipeline.

    Steps:

    Page
        ↓
    Lines
        ↓
    Words
        ↓
    Normalized Words
    """

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = (
        load_handwriting_image(
            image_path
        )
    )

    sample_name = (
        image_path.name
    )

    sample_stem = (
        image_path.stem
    )

    # --------------------------------------------------------
    # EXTRACT LINES
    # --------------------------------------------------------

    line_results = (
        extract_handwritten_lines(
            image
        )
    )

    extracted_lines = (
        line_results[
            "extracted_lines"
        ]
    )

    line_metadata = (
        save_extracted_lines(
            extracted_lines=
                extracted_lines,

            output_directory=
                dataset_directories[
                    "lines_directory"
                ],

            sample_name=
                sample_name,
        )
    )

    # --------------------------------------------------------
    # PROCESS WORDS
    # --------------------------------------------------------

    complete_word_metadata = []

    normalized_word_metadata = []

    for line_data in (
        extracted_lines
    ):

        line_index = (
            line_data[
                "line_index"
            ]
            + 1
        )

        line_image = (
            line_data[
                "image"
            ]
        )

        # ----------------------------------------------------
        # EXTRACT WORDS
        # ----------------------------------------------------

        word_results = (
            extract_handwritten_words(
                line_image
            )
        )

        extracted_words = (
            word_results[
                "extracted_words"
            ]
        )

        # ----------------------------------------------------
        # SAVE WORDS
        # ----------------------------------------------------

        word_metadata = (
            save_extracted_words(
                extracted_words=
                    extracted_words,

                output_directory=
                    dataset_directories[
                        "words_directory"
                    ],

                sample_name=
                    sample_stem,

                line_index=
                    line_index,
            )
        )

        complete_word_metadata.extend(
            word_metadata
        )

        # ----------------------------------------------------
        # NORMALIZE EVERY WORD
        # ----------------------------------------------------

        for word_data, word_info in zip(
            extracted_words,
            word_metadata,
        ):

            word_image = (
                word_data[
                    "image"
                ]
            )

            try:

                normalization_results = (
                    normalize_handwriting_word(
                        word_image
                    )
                )

                normalized_image = (
                    normalization_results[
                        "normalized_image"
                    ]
                )

                saved_normalized_word = (
                    save_normalized_word(
                        normalized_image=
                            normalized_image,

                        output_directory=
                            dataset_directories[
                                "normalized_words_directory"
                            ],

                        original_filename=
                            word_info[
                                "filename"
                            ],
                    )
                )

                normalized_metadata = {
                    "sample_name":
                        sample_name,

                    "line_index":
                        line_index,

                    "word_index":
                        word_info[
                            "word_index"
                        ],

                    "original_word_filename":
                        word_info[
                            "filename"
                        ],

                    "original_word_path":
                        word_info[
                            "path"
                        ],

                    "normalized_filename":
                        saved_normalized_word[
                            "filename"
                        ],

                    "normalized_path":
                        saved_normalized_word[
                            "path"
                        ],

                    "original_width":
                        word_info[
                            "width"
                        ],

                    "original_height":
                        word_info[
                            "height"
                        ],

                    "normalized_width":
                        saved_normalized_word[
                            "width"
                        ],

                    "normalized_height":
                        saved_normalized_word[
                            "height"
                        ],

                    "bounding_box":
                        normalization_results[
                            "bounding_box"
                        ],

                    "estimated_baseline":
                        normalization_results[
                            "estimated_baseline"
                        ],

                    "vertical_shift":
                        normalization_results[
                            "vertical_shift"
                        ],

                    "status":
                        "success",
                }

                normalized_word_metadata.append(
                    normalized_metadata
                )

            except Exception as error:

                failed_metadata = {
                    "sample_name":
                        sample_name,

                    "line_index":
                        line_index,

                    "word_index":
                        word_info[
                            "word_index"
                        ],

                    "original_word_filename":
                        word_info[
                            "filename"
                        ],

                    "original_word_path":
                        word_info[
                            "path"
                        ],

                    "normalized_filename":
                        "",

                    "normalized_path":
                        "",

                    "original_width":
                        word_info[
                            "width"
                        ],

                    "original_height":
                        word_info[
                            "height"
                        ],

                    "normalized_width":
                        "",

                    "normalized_height":
                        "",

                    "bounding_box":
                        "",

                    "estimated_baseline":
                        "",

                    "vertical_shift":
                        "",

                    "status":
                        f"failed: {error}",
                }

                normalized_word_metadata.append(
                    failed_metadata
                )

    # --------------------------------------------------------
    # PAGE RESULTS
    # --------------------------------------------------------

    return {
        "sample_name":
            sample_name,

        "sample_path":
            str(
                image_path
            ),

        "line_count":
            len(
                line_metadata
            ),

        "word_count":
            len(
                complete_word_metadata
            ),

        "normalized_word_count":
            len(
                [
                    item
                    for item
                    in normalized_word_metadata
                    if item[
                        "status"
                    ]
                    == "success"
                ]
            ),

        "line_metadata":
            line_metadata,

        "word_metadata":
            complete_word_metadata,

        "normalized_word_metadata":
            normalized_word_metadata,
    }


# ============================================================
# FIND HANDWRITING IMAGES
# ============================================================

def find_handwriting_images(
    input_directory: Path,
) -> list:
    """
    Find supported handwriting images.
    """

    supported_extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ]

    image_paths = []

    for extension in (
        supported_extensions
    ):

        image_paths.extend(
            input_directory.glob(
                f"*{extension}"
            )
        )

        image_paths.extend(
            input_directory.glob(
                f"*{extension.upper()}"
            )
        )

    return sorted(
        image_paths
    )


# ============================================================
# SAVE JSON METADATA
# ============================================================

def save_metadata_json(
    metadata: list,
    output_path: Path,
) -> None:
    """
    Save dataset metadata as JSON.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# SAVE CSV METADATA
# ============================================================

def save_metadata_csv(
    metadata: list,
    output_path: Path,
) -> None:
    """
    Save normalized word metadata
    as CSV.
    """

    if not metadata:

        return

    fieldnames = (
        metadata[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=
                    fieldnames,
            )
        )

        writer.writeheader()

        writer.writerows(
            metadata
        )


# ============================================================
# PROCESS COMPLETE DATASET
# ============================================================

def build_generation_dataset(
    input_directory: Path,
    dataset_directory: Path,
) -> dict:
    """
    Build a complete handwriting
    generation dataset.

    Processes every handwriting page
    inside the input directory.
    """

    # --------------------------------------------------------
    # CHECK INPUT DIRECTORY
    # --------------------------------------------------------

    if not input_directory.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{input_directory}"
        )

    # --------------------------------------------------------
    # FIND HANDWRITING PAGES
    # --------------------------------------------------------

    image_paths = (
        find_handwriting_images(
            input_directory
        )
    )

    if not image_paths:

        raise FileNotFoundError(
            f"No handwriting images found in:\n"
            f"{input_directory}"
        )

    # --------------------------------------------------------
    # CREATE DATASET DIRECTORIES
    # --------------------------------------------------------

    dataset_directories = (
        create_dataset_directories(
            dataset_directory
        )
    )

    # --------------------------------------------------------
    # COMPLETE METADATA
    # --------------------------------------------------------

    complete_page_metadata = []

    complete_normalized_metadata = []

    failed_pages = []

    # --------------------------------------------------------
    # PROCESS EVERY PAGE
    # --------------------------------------------------------

    for image_path in (
        image_paths
    ):

        try:

            print(
                f"\nProcessing:"
                f" {image_path.name}"
            )

            page_results = (
                process_handwriting_page(
                    image_path=
                        image_path,

                    dataset_directories=
                        dataset_directories,
                )
            )

            complete_page_metadata.append(
                page_results
            )

            complete_normalized_metadata.extend(
                page_results[
                    "normalized_word_metadata"
                ]
            )

            print(
                f"Lines:"
                f" {page_results['line_count']}"
            )

            print(
                f"Words:"
                f" {page_results['word_count']}"
            )

            print(
                f"Normalized:"
                f" {page_results['normalized_word_count']}"
            )

        except Exception as error:

            failed_pages.append(
                {
                    "sample_name":
                        image_path.name,

                    "sample_path":
                        str(
                            image_path
                        ),

                    "error":
                        str(
                            error
                        ),
                }
            )

            print(
                f"\nFailed to process:"
                f" {image_path.name}"
            )

            print(
                f"Error:"
                f" {error}"
            )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    json_path = (
        dataset_directory
        / "metadata.json"
    )

    csv_path = (
        dataset_directory
        / "metadata.csv"
    )

    pages_json_path = (
        dataset_directory
        / "pages_metadata.json"
    )

    failed_pages_path = (
        dataset_directory
        / "failed_pages.json"
    )

    save_metadata_json(
        metadata=
            complete_normalized_metadata,

        output_path=
            json_path,
    )

    save_metadata_csv(
        metadata=
            complete_normalized_metadata,

        output_path=
            csv_path,
    )

    save_metadata_json(
        metadata=
            complete_page_metadata,

        output_path=
            pages_json_path,
    )

    save_metadata_json(
        metadata=
            failed_pages,

        output_path=
            failed_pages_path,
    )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    successful_words = (
        len(
            [
                item
                for item
                in complete_normalized_metadata
                if item[
                    "status"
                ]
                == "success"
            ]
        )
    )

    failed_words = (
        len(
            complete_normalized_metadata
        )
        - successful_words
    )

    return {
        "total_pages_found":
            len(
                image_paths
            ),

        "successfully_processed_pages":
            len(
                complete_page_metadata
            ),

        "failed_pages":
            len(
                failed_pages
            ),

        "total_words":
            len(
                complete_normalized_metadata
            ),

        "successfully_normalized_words":
            successful_words,

        "failed_words":
            failed_words,

        "dataset_directory":
            str(
                dataset_directory
            ),

        "metadata_json":
            str(
                json_path
            ),

        "metadata_csv":
            str(
                csv_path
            ),

        "pages_metadata":
            str(
                pages_json_path
            ),

        "failed_pages_metadata":
            str(
                failed_pages_path
            ),
    }