# ============================================================
# HANDWRITTEN WORD EXTRACTION
# Phase 6C
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONVERT IMAGE TO GRAYSCALE
# ============================================================

def convert_to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert an image to grayscale.
    """

    if image is None:

        raise ValueError(
            "Input image cannot be None."
        )

    if len(image.shape) == 2:

        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


# ============================================================
# PREPROCESS IMAGE FOR WORD EXTRACTION
# ============================================================

def preprocess_for_word_extraction(
    image: np.ndarray,
) -> np.ndarray:
    """
    Prepare a handwriting line image
    for word extraction.

    Returns a binary image where
    handwriting pixels are white.
    """

    gray_image = (
        convert_to_grayscale(
            image
        )
    )

    # --------------------------------------------------------
    # NOISE REDUCTION
    # --------------------------------------------------------

    blurred_image = cv2.GaussianBlur(
        gray_image,
        (3, 3),
        0,
    )

    # --------------------------------------------------------
    # ADAPTIVE THRESHOLDING
    # --------------------------------------------------------

    binary_image = (
        cv2.adaptiveThreshold(
            blurred_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )
    )

    # --------------------------------------------------------
    # REMOVE SMALL NOISE
    # --------------------------------------------------------

    kernel = np.ones(
        (2, 2),
        np.uint8,
    )

    cleaned_image = (
        cv2.morphologyEx(
            binary_image,
            cv2.MORPH_OPEN,
            kernel,
        )
    )

    return cleaned_image


# ============================================================
# CALCULATE VERTICAL PROJECTION
# ============================================================

def calculate_vertical_projection(
    binary_image: np.ndarray,
) -> np.ndarray:
    """
    Calculate the number of foreground
    handwriting pixels in every column.
    """

    if binary_image is None:

        raise ValueError(
            "Binary image cannot be None."
        )

    foreground_pixels = (
        binary_image > 0
    )

    vertical_projection = (
        np.sum(
            foreground_pixels,
            axis=0,
        )
    )

    return vertical_projection


# ============================================================
# FIND INK BOUNDARIES
# ============================================================

def find_ink_regions(
    vertical_projection: np.ndarray,
    minimum_pixels_per_column: int = 1,
) -> list:
    """
    Detect continuous regions containing
    handwriting pixels.
    """

    regions = []

    inside_region = False

    start_column = 0

    for column_index, pixel_count in enumerate(
        vertical_projection
    ):

        has_ink = (
            pixel_count
            >= minimum_pixels_per_column
        )

        # ----------------------------------------------------
        # START REGION
        # ----------------------------------------------------

        if (
            has_ink
            and not inside_region
        ):

            start_column = (
                column_index
            )

            inside_region = True

        # ----------------------------------------------------
        # END REGION
        # ----------------------------------------------------

        elif (
            not has_ink
            and inside_region
        ):

            end_column = (
                column_index
                - 1
            )

            regions.append(
                (
                    start_column,
                    end_column,
                )
            )

            inside_region = False

    # --------------------------------------------------------
    # HANDLE FINAL REGION
    # --------------------------------------------------------

    if inside_region:

        regions.append(
            (
                start_column,
                len(
                    vertical_projection
                )
                - 1,
            )
        )

    return regions


# ============================================================
# CALCULATE GAP SIZES
# ============================================================

def calculate_gap_sizes(
    ink_regions: list,
) -> list:
    """
    Calculate whitespace gaps between
    neighboring handwriting regions.
    """

    gap_sizes = []

    if len(
        ink_regions
    ) < 2:

        return gap_sizes

    for index in range(
        len(
            ink_regions
        )
        - 1
    ):

        current_end = (
            ink_regions[
                index
            ][1]
        )

        next_start = (
            ink_regions[
                index
                + 1
            ][0]
        )

        gap_size = (
            next_start
            - current_end
            - 1
        )

        gap_sizes.append(
            gap_size
        )

    return gap_sizes


# ============================================================
# ESTIMATE WORD GAP THRESHOLD
# ============================================================

def estimate_word_gap_threshold(
    gap_sizes: list,
    minimum_word_gap: int = 12,
) -> float:
    """
    Estimate a threshold that separates
    character gaps from word gaps.
    """

    if not gap_sizes:

        return float(
            minimum_word_gap
        )

    positive_gaps = [
        gap
        for gap in gap_sizes
        if gap > 0
    ]

    if not positive_gaps:

        return float(
            minimum_word_gap
        )

    median_gap = float(
        np.median(
            positive_gaps
        )
    )

    threshold = max(
        minimum_word_gap,
        median_gap * 3.0,
    )

    return float(
        threshold
    )


# ============================================================
# FIND WORD BOUNDARIES
# ============================================================

def find_word_boundaries(
    ink_regions: list,
    word_gap_threshold: float,
) -> list:
    """
    Group nearby handwriting regions
    into complete words.
    """

    if not ink_regions:

        return []

    word_regions = []

    current_start = (
        ink_regions[0][0]
    )

    current_end = (
        ink_regions[0][1]
    )

    for index in range(
        1,
        len(
            ink_regions
        )
    ):

        previous_end = (
            ink_regions[
                index
                - 1
            ][1]
        )

        current_region_start = (
            ink_regions[
                index
            ][0]
        )

        current_region_end = (
            ink_regions[
                index
            ][1]
        )

        gap_size = (
            current_region_start
            - previous_end
            - 1
        )

        # ----------------------------------------------------
        # NEW WORD
        # ----------------------------------------------------

        if (
            gap_size
            >= word_gap_threshold
        ):

            word_regions.append(
                (
                    current_start,
                    current_end,
                )
            )

            current_start = (
                current_region_start
            )

            current_end = (
                current_region_end
            )

        # ----------------------------------------------------
        # SAME WORD
        # ----------------------------------------------------

        else:

            current_end = (
                current_region_end
            )

    # --------------------------------------------------------
    # FINAL WORD
    # --------------------------------------------------------

    word_regions.append(
        (
            current_start,
            current_end,
        )
    )

    return word_regions


# ============================================================
# FILTER WORD REGIONS
# ============================================================

def filter_word_regions(
    word_regions: list,
    minimum_word_width: int = 8,
) -> list:
    """
    Remove extremely small regions
    that are unlikely to be words.
    """

    filtered_regions = []

    for start_column, end_column in (
        word_regions
    ):

        word_width = (
            end_column
            - start_column
            + 1
        )

        if (
            word_width
            >= minimum_word_width
        ):

            filtered_regions.append(
                (
                    start_column,
                    end_column,
                )
            )

    return filtered_regions


# ============================================================
# ADD WORD PADDING
# ============================================================

def add_word_padding(
    word_regions: list,
    image_width: int,
    padding: int = 5,
) -> list:
    """
    Add horizontal padding around
    extracted word regions.
    """

    padded_regions = []

    for start_column, end_column in (
        word_regions
    ):

        padded_start = max(
            0,
            start_column
            - padding,
        )

        padded_end = min(
            image_width
            - 1,
            end_column
            + padding,
        )

        padded_regions.append(
            (
                padded_start,
                padded_end,
            )
        )

    return padded_regions


# ============================================================
# EXTRACT WORD IMAGES
# ============================================================

def extract_word_images(
    line_image: np.ndarray,
    word_regions: list,
) -> list:
    """
    Extract individual word images
    from a handwritten line.
    """

    extracted_words = []

    for word_index, (
        start_column,
        end_column,
    ) in enumerate(
        word_regions
    ):

        word_image = (
            line_image[
                :,
                start_column:
                end_column
                + 1,
            ]
        )

        extracted_words.append(
            {
                "word_index": (
                    word_index
                ),

                "start_column": (
                    start_column
                ),

                "end_column": (
                    end_column
                ),

                "image": (
                    word_image
                ),
            }
        )

    return extracted_words


# ============================================================
# SAVE EXTRACTED WORDS
# ============================================================

def save_extracted_words(
    extracted_words: list,
    output_directory: Path,
    sample_name: str,
    line_index: int,
) -> list:
    """
    Save extracted handwriting words.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    word_metadata = []

    sample_stem = (
        Path(
            sample_name
        ).stem
    )

    for word_data in (
        extracted_words
    ):

        word_index = (
            word_data[
                "word_index"
            ]
        )

        filename = (
            f"{sample_stem}_"
            f"line_{line_index:03d}_"
            f"word_{word_index + 1:03d}.png"
        )

        output_path = (
            output_directory
            / filename
        )

        success = cv2.imwrite(
            str(
                output_path
            ),
            word_data[
                "image"
            ],
        )

        if not success:

            raise RuntimeError(
                f"Unable to save word: "
                f"{filename}"
            )

        height, width = (
            word_data[
                "image"
            ].shape[:2]
        )

        word_metadata.append(
            {
                "sample_name": (
                    sample_name
                ),

                "line_index": (
                    line_index
                ),

                "word_index": (
                    word_index
                    + 1
                ),

                "filename": (
                    filename
                ),

                "path": (
                    str(
                        output_path
                    )
                ),

                "start_column": (
                    word_data[
                        "start_column"
                    ]
                ),

                "end_column": (
                    word_data[
                        "end_column"
                    ]
                ),

                "width": (
                    width
                ),

                "height": (
                    height
                ),
            }
        )

    return word_metadata


# ============================================================
# COMPLETE WORD EXTRACTION PIPELINE
# ============================================================

def extract_handwritten_words(
    line_image: np.ndarray,
    minimum_pixels_per_column: int = 1,
    minimum_word_gap: int = 12,
    minimum_word_width: int = 8,
    padding: int = 5,
) -> dict:
    """
    Complete handwritten word extraction
    pipeline.
    """

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    binary_image = (
        preprocess_for_word_extraction(
            line_image
        )
    )

    # --------------------------------------------------------
    # VERTICAL PROJECTION
    # --------------------------------------------------------

    vertical_projection = (
        calculate_vertical_projection(
            binary_image
        )
    )

    # --------------------------------------------------------
    # FIND RAW INK REGIONS
    # --------------------------------------------------------

    ink_regions = (
        find_ink_regions(
            vertical_projection=
                vertical_projection,

            minimum_pixels_per_column=
                minimum_pixels_per_column,
        )
    )

    # --------------------------------------------------------
    # CALCULATE GAPS
    # --------------------------------------------------------

    gap_sizes = (
        calculate_gap_sizes(
            ink_regions
        )
    )

    # --------------------------------------------------------
    # ESTIMATE WORD GAP
    # --------------------------------------------------------

    word_gap_threshold = (
        estimate_word_gap_threshold(
            gap_sizes=
                gap_sizes,

            minimum_word_gap=
                minimum_word_gap,
        )
    )

    # --------------------------------------------------------
    # FIND WORD BOUNDARIES
    # --------------------------------------------------------

    word_regions = (
        find_word_boundaries(
            ink_regions=
                ink_regions,

            word_gap_threshold=
                word_gap_threshold,
        )
    )

    # --------------------------------------------------------
    # FILTER SMALL REGIONS
    # --------------------------------------------------------

    filtered_regions = (
        filter_word_regions(
            word_regions=
                word_regions,

            minimum_word_width=
                minimum_word_width,
        )
    )

    # --------------------------------------------------------
    # ADD PADDING
    # --------------------------------------------------------

    image_width = (
        line_image.shape[1]
    )

    padded_regions = (
        add_word_padding(
            word_regions=
                filtered_regions,

            image_width=
                image_width,

            padding=
                padding,
        )
    )

    # --------------------------------------------------------
    # EXTRACT WORD IMAGES
    # --------------------------------------------------------

    extracted_words = (
        extract_word_images(
            line_image=
                line_image,

            word_regions=
                padded_regions,
        )
    )

    return {
        "binary_image": (
            binary_image
        ),

        "vertical_projection": (
            vertical_projection
        ),

        "ink_regions": (
            ink_regions
        ),

        "gap_sizes": (
            gap_sizes
        ),

        "word_gap_threshold": (
            word_gap_threshold
        ),

        "word_regions": (
            padded_regions
        ),

        "extracted_words": (
            extracted_words
        ),

        "word_count": (
            len(
                extracted_words
            )
        ),
    }