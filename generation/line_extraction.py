# ============================================================
# HANDWRITTEN LINE EXTRACTION
# Phase 6B
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
# PREPROCESS IMAGE FOR LINE EXTRACTION
# ============================================================

def preprocess_for_line_extraction(
    image: np.ndarray,
) -> np.ndarray:
    """
    Prepare a handwriting image for
    horizontal line extraction.

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
# CALCULATE HORIZONTAL PROJECTION
# ============================================================

def calculate_horizontal_projection(
    binary_image: np.ndarray,
) -> np.ndarray:
    """
    Calculate the number of foreground
    handwriting pixels in every row.
    """

    if binary_image is None:

        raise ValueError(
            "Binary image cannot be None."
        )

    foreground_pixels = (
        binary_image > 0
    )

    horizontal_projection = (
        np.sum(
            foreground_pixels,
            axis=1,
        )
    )

    return horizontal_projection


# ============================================================
# FIND RAW LINE REGIONS
# ============================================================

def find_raw_line_regions(
    horizontal_projection: np.ndarray,
    minimum_pixels_per_row: int = 3,
) -> list:
    """
    Detect continuous horizontal regions
    containing handwriting pixels.
    """

    line_regions = []

    inside_line = False

    start_row = 0

    for row_index, pixel_count in enumerate(
        horizontal_projection
    ):

        has_handwriting = (
            pixel_count
            >= minimum_pixels_per_row
        )

        # ----------------------------------------------------
        # START NEW REGION
        # ----------------------------------------------------

        if (
            has_handwriting
            and not inside_line
        ):

            start_row = (
                row_index
            )

            inside_line = True

        # ----------------------------------------------------
        # END REGION
        # ----------------------------------------------------

        elif (
            not has_handwriting
            and inside_line
        ):

            end_row = (
                row_index
                - 1
            )

            line_regions.append(
                (
                    start_row,
                    end_row,
                )
            )

            inside_line = False

    # --------------------------------------------------------
    # HANDLE LAST REGION
    # --------------------------------------------------------

    if inside_line:

        line_regions.append(
            (
                start_row,
                len(
                    horizontal_projection
                )
                - 1,
            )
        )

    return line_regions


# ============================================================
# MERGE NEARBY LINE REGIONS
# ============================================================

def merge_nearby_line_regions(
    line_regions: list,
    maximum_gap: int = 15,
) -> list:
    """
    Merge nearby handwriting regions.

    This helps combine characters,
    dots and punctuation that may
    be separated vertically.
    """

    if not line_regions:

        return []

    merged_regions = []

    current_start, current_end = (
        line_regions[0]
    )

    for start_row, end_row in (
        line_regions[1:]
    ):

        gap = (
            start_row
            - current_end
            - 1
        )

        if gap <= maximum_gap:

            current_end = (
                max(
                    current_end,
                    end_row,
                )
            )

        else:

            merged_regions.append(
                (
                    current_start,
                    current_end,
                )
            )

            current_start = (
                start_row
            )

            current_end = (
                end_row
            )

    merged_regions.append(
        (
            current_start,
            current_end,
        )
    )

    return merged_regions


# ============================================================
# FILTER SMALL LINE REGIONS
# ============================================================

def filter_line_regions(
    line_regions: list,
    minimum_line_height: int = 10,
) -> list:
    """
    Remove extremely small regions that
    are unlikely to represent text lines.
    """

    filtered_regions = []

    for start_row, end_row in (
        line_regions
    ):

        line_height = (
            end_row
            - start_row
            + 1
        )

        if (
            line_height
            >= minimum_line_height
        ):

            filtered_regions.append(
                (
                    start_row,
                    end_row,
                )
            )

    return filtered_regions


# ============================================================
# ADD PADDING TO LINE REGIONS
# ============================================================

def add_line_padding(
    line_regions: list,
    image_height: int,
    padding: int = 8,
) -> list:
    """
    Add vertical padding around
    detected handwriting lines.
    """

    padded_regions = []

    for start_row, end_row in (
        line_regions
    ):

        padded_start = max(
            0,
            start_row
            - padding,
        )

        padded_end = min(
            image_height
            - 1,
            end_row
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
# EXTRACT LINE IMAGES
# ============================================================

def extract_line_images(
    image: np.ndarray,
    line_regions: list,
) -> list:
    """
    Extract individual handwriting
    line images from the original image.
    """

    extracted_lines = []

    for line_index, (
        start_row,
        end_row,
    ) in enumerate(
        line_regions
    ):

        line_image = (
            image[
                start_row:
                end_row
                + 1,
                :
            ]
        )

        extracted_lines.append(
            {
                "line_index": (
                    line_index
                ),

                "start_row": (
                    start_row
                ),

                "end_row": (
                    end_row
                ),

                "image": (
                    line_image
                ),
            }
        )

    return extracted_lines


# ============================================================
# SAVE EXTRACTED LINES
# ============================================================

def save_extracted_lines(
    extracted_lines: list,
    output_directory: Path,
    sample_name: str,
) -> list:
    """
    Save extracted handwriting lines.

    Returns metadata for each line.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    line_metadata = []

    sample_stem = (
        Path(
            sample_name
        ).stem
    )

    for line_data in (
        extracted_lines
    ):

        line_index = (
            line_data[
                "line_index"
            ]
        )

        filename = (
            f"{sample_stem}_"
            f"line_{line_index + 1:03d}.png"
        )

        output_path = (
            output_directory
            / filename
        )

        success = cv2.imwrite(
            str(
                output_path
            ),
            line_data[
                "image"
            ],
        )

        if not success:

            raise RuntimeError(
                f"Unable to save line: "
                f"{filename}"
            )

        height, width = (
            line_data[
                "image"
            ].shape[:2]
        )

        line_metadata.append(
            {
                "sample_name": (
                    sample_name
                ),

                "line_index": (
                    line_index
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

                "start_row": (
                    line_data[
                        "start_row"
                    ]
                ),

                "end_row": (
                    line_data[
                        "end_row"
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

    return line_metadata


# ============================================================
# COMPLETE LINE EXTRACTION PIPELINE
# ============================================================

def extract_handwritten_lines(
    image: np.ndarray,
    minimum_pixels_per_row: int = 3,
    maximum_gap: int = 15,
    minimum_line_height: int = 10,
    padding: int = 8,
) -> dict:
    """
    Complete handwritten line extraction
    pipeline.

    Returns:

    - binary image
    - horizontal projection
    - detected line regions
    - extracted line images
    """

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    binary_image = (
        preprocess_for_line_extraction(
            image
        )
    )

    # --------------------------------------------------------
    # HORIZONTAL PROJECTION
    # --------------------------------------------------------

    horizontal_projection = (
        calculate_horizontal_projection(
            binary_image
        )
    )

    # --------------------------------------------------------
    # RAW REGIONS
    # --------------------------------------------------------

    raw_regions = (
        find_raw_line_regions(
            horizontal_projection=
                horizontal_projection,

            minimum_pixels_per_row=
                minimum_pixels_per_row,
        )
    )

    # --------------------------------------------------------
    # MERGE REGIONS
    # --------------------------------------------------------

    merged_regions = (
        merge_nearby_line_regions(
            line_regions=
                raw_regions,

            maximum_gap=
                maximum_gap,
        )
    )

    # --------------------------------------------------------
    # FILTER SMALL REGIONS
    # --------------------------------------------------------

    filtered_regions = (
        filter_line_regions(
            line_regions=
                merged_regions,

            minimum_line_height=
                minimum_line_height,
        )
    )

    # --------------------------------------------------------
    # ADD PADDING
    # --------------------------------------------------------

    image_height = (
        image.shape[0]
    )

    padded_regions = (
        add_line_padding(
            line_regions=
                filtered_regions,

            image_height=
                image_height,

            padding=
                padding,
        )
    )

    # --------------------------------------------------------
    # EXTRACT LINES
    # --------------------------------------------------------

    extracted_lines = (
        extract_line_images(
            image=image,

            line_regions=
                padded_regions,
        )
    )

    return {
        "binary_image": (
            binary_image
        ),

        "horizontal_projection": (
            horizontal_projection
        ),

        "raw_regions": (
            raw_regions
        ),

        "merged_regions": (
            merged_regions
        ),

        "line_regions": (
            padded_regions
        ),

        "extracted_lines": (
            extracted_lines
        ),

        "line_count": (
            len(
                extracted_lines
            )
        ),
    }