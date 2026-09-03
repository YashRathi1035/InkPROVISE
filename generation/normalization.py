# ============================================================
# HANDWRITING NORMALIZATION
# Phase 6D
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
# CREATE BINARY HANDWRITING IMAGE
# ============================================================

def create_binary_handwriting_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert a handwriting image into a
    binary image where handwriting pixels
    are white.
    """

    gray_image = (
        convert_to_grayscale(
            image
        )
    )

    blurred_image = (
        cv2.GaussianBlur(
            gray_image,
            (3, 3),
            0,
        )
    )

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

    return binary_image


# ============================================================
# FIND HANDWRITING BOUNDING BOX
# ============================================================

def find_handwriting_bounding_box(
    binary_image: np.ndarray,
) -> tuple:
    """
    Find the bounding box containing
    the handwriting pixels.

    Returns:

    x
    y
    width
    height
    """

    foreground_points = (
        cv2.findNonZero(
            binary_image
        )
    )

    if foreground_points is None:

        raise ValueError(
            "No handwriting pixels detected."
        )

    x, y, width, height = (
        cv2.boundingRect(
            foreground_points
        )
    )

    return (
        x,
        y,
        width,
        height,
    )


# ============================================================
# CROP HANDWRITING REGION
# ============================================================

def crop_handwriting_region(
    image: np.ndarray,
    bounding_box: tuple,
    padding: int = 5,
) -> np.ndarray:
    """
    Crop the image around the handwriting
    bounding box while preserving padding.
    """

    x, y, width, height = (
        bounding_box
    )

    image_height, image_width = (
        image.shape[:2]
    )

    x_start = max(
        0,
        x - padding,
    )

    y_start = max(
        0,
        y - padding,
    )

    x_end = min(
        image_width,
        x + width + padding,
    )

    y_end = min(
        image_height,
        y + height + padding,
    )

    cropped_image = (
        image[
            y_start:y_end,
            x_start:x_end,
        ]
    )

    return cropped_image


# ============================================================
# RESIZE WORD BY HEIGHT
# ============================================================

def resize_word_by_height(
    image: np.ndarray,
    target_height: int = 96,
) -> np.ndarray:
    """
    Resize handwriting while preserving
    its aspect ratio.
    """

    current_height, current_width = (
        image.shape[:2]
    )

    if current_height == 0:

        raise ValueError(
            "Image height cannot be zero."
        )

    scale = (
        target_height
        / current_height
    )

    target_width = max(
        1,
        int(
            current_width
            * scale
        ),
    )

    resized_image = (
        cv2.resize(
            image,
            (
                target_width,
                target_height,
            ),
            interpolation=
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_CUBIC,
        )
    )

    return resized_image


# ============================================================
# ADD CANVAS PADDING
# ============================================================

def add_canvas_padding(
    image: np.ndarray,
    vertical_padding: int = 12,
    horizontal_padding: int = 16,
) -> np.ndarray:
    """
    Add consistent white padding around
    normalized handwriting.
    """

    padded_image = (
        cv2.copyMakeBorder(
            image,
            vertical_padding,
            vertical_padding,
            horizontal_padding,
            horizontal_padding,
            borderType=
                cv2.BORDER_CONSTANT,
            value=(
                255,
                255,
                255,
            ),
        )
    )

    return padded_image


# ============================================================
# ESTIMATE BASELINE POSITION
# ============================================================

def estimate_baseline(
    binary_image: np.ndarray,
) -> int:
    """
    Estimate the handwriting baseline
    using the horizontal ink projection.

    The baseline is approximated as the
    lower region with the strongest
    handwriting density.
    """

    foreground = (
        binary_image > 0
    )

    horizontal_projection = (
        np.sum(
            foreground,
            axis=1,
        )
    )

    if (
        horizontal_projection.size
        == 0
    ):

        return 0

    image_height = (
        binary_image.shape[0]
    )

    # Focus on lower 60% of the word.

    search_start = int(
        image_height * 0.40
    )

    lower_projection = (
        horizontal_projection[
            search_start:
        ]
    )

    if lower_projection.size == 0:

        return int(
            np.argmax(
                horizontal_projection
            )
        )

    baseline_offset = int(
        np.argmax(
            lower_projection
        )
    )

    baseline = (
        search_start
        + baseline_offset
    )

    return baseline


# ============================================================
# ALIGN WORD TO BASELINE
# ============================================================

def align_word_to_baseline(
    image: np.ndarray,
    target_baseline_ratio: float = 0.75,
) -> tuple:
    """
    Vertically align handwriting using
    the estimated baseline.

    Returns:

    aligned_image
    estimated_baseline
    vertical_shift
    """

    binary_image = (
        create_binary_handwriting_image(
            image
        )
    )

    estimated_baseline = (
        estimate_baseline(
            binary_image
        )
    )

    image_height = (
        image.shape[0]
    )

    target_baseline = int(
        image_height
        * target_baseline_ratio
    )

    vertical_shift = (
        target_baseline
        - estimated_baseline
    )

    translation_matrix = np.float32(
        [
            [
                1,
                0,
                0,
            ],
            [
                0,
                1,
                vertical_shift,
            ],
        ]
    )

    aligned_image = (
        cv2.warpAffine(
            image,
            translation_matrix,
            (
                image.shape[1],
                image.shape[0],
            ),
            flags=
                cv2.INTER_CUBIC,
            borderMode=
                cv2.BORDER_CONSTANT,
            borderValue=(
                255,
                255,
                255,
            ),
        )
    )

    return (
        aligned_image,
        estimated_baseline,
        vertical_shift,
    )


# ============================================================
# NORMALIZE HANDWRITING WORD
# ============================================================

def normalize_handwriting_word(
    word_image: np.ndarray,
    crop_padding: int = 5,
    target_height: int = 96,
    vertical_padding: int = 12,
    horizontal_padding: int = 16,
    target_baseline_ratio: float = 0.75,
) -> dict:
    """
    Complete handwriting normalization
    pipeline for one word.
    """

    # --------------------------------------------------------
    # CREATE BINARY IMAGE
    # --------------------------------------------------------

    binary_image = (
        create_binary_handwriting_image(
            word_image
        )
    )

    # --------------------------------------------------------
    # FIND HANDWRITING
    # --------------------------------------------------------

    bounding_box = (
        find_handwriting_bounding_box(
            binary_image
        )
    )

    # --------------------------------------------------------
    # CROP WORD
    # --------------------------------------------------------

    cropped_image = (
        crop_handwriting_region(
            image=
                word_image,

            bounding_box=
                bounding_box,

            padding=
                crop_padding,
        )
    )

    # --------------------------------------------------------
    # BASELINE ALIGNMENT
    # --------------------------------------------------------

    (
        aligned_image,
        estimated_baseline,
        vertical_shift,
    ) = (
        align_word_to_baseline(
            image=
                cropped_image,

            target_baseline_ratio=
                target_baseline_ratio,
        )
    )

    # --------------------------------------------------------
    # HEIGHT NORMALIZATION
    # --------------------------------------------------------

    resized_image = (
        resize_word_by_height(
            image=
                aligned_image,

            target_height=
                target_height,
        )
    )

    # --------------------------------------------------------
    # FINAL PADDING
    # --------------------------------------------------------

    normalized_image = (
        add_canvas_padding(
            image=
                resized_image,

            vertical_padding=
                vertical_padding,

            horizontal_padding=
                horizontal_padding,
        )
    )

    return {
        "original_image": (
            word_image
        ),

        "binary_image": (
            binary_image
        ),

        "bounding_box": (
            bounding_box
        ),

        "cropped_image": (
            cropped_image
        ),

        "aligned_image": (
            aligned_image
        ),

        "resized_image": (
            resized_image
        ),

        "normalized_image": (
            normalized_image
        ),

        "estimated_baseline": (
            estimated_baseline
        ),

        "vertical_shift": (
            vertical_shift
        ),

        "target_height": (
            target_height
        ),
    }


# ============================================================
# SAVE NORMALIZED WORD
# ============================================================

def save_normalized_word(
    normalized_image: np.ndarray,
    output_directory: Path,
    original_filename: str,
) -> dict:
    """
    Save a normalized handwriting word.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{Path(original_filename).stem}"
        f"_normalized.png"
    )

    output_path = (
        output_directory
        / filename
    )

    success = (
        cv2.imwrite(
            str(
                output_path
            ),
            normalized_image,
        )
    )

    if not success:

        raise RuntimeError(
            "Unable to save normalized word."
        )

    height, width = (
        normalized_image.shape[:2]
    )

    return {
        "filename": (
            filename
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
    }