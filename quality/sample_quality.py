# ============================================================
# HANDWRITING SAMPLE QUALITY ANALYSIS
# Phase 5A
# ============================================================


import cv2
import numpy as np


# ============================================================
# IMAGE SHARPNESS
# ============================================================

def calculate_sharpness(
    image: np.ndarray,
) -> float:
    """
    Calculate image sharpness using
    the variance of the Laplacian.
    """

    if image is None:

        return 0.0

    if len(image.shape) == 3:

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    else:

        gray_image = image

    laplacian = cv2.Laplacian(
        gray_image,
        cv2.CV_64F,
    )

    sharpness = (
        laplacian.var()
    )

    return float(
        sharpness
    )


# ============================================================
# IMAGE BRIGHTNESS
# ============================================================

def calculate_brightness(
    image: np.ndarray,
) -> float:
    """
    Calculate average image brightness.
    """

    if image is None:

        return 0.0

    if len(image.shape) == 3:

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    else:

        gray_image = image

    brightness = (
        np.mean(
            gray_image
        )
    )

    return float(
        brightness
    )


# ============================================================
# IMAGE CONTRAST
# ============================================================

def calculate_contrast(
    image: np.ndarray,
) -> float:
    """
    Calculate image contrast using
    standard deviation.
    """

    if image is None:

        return 0.0

    if len(image.shape) == 3:

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    else:

        gray_image = image

    contrast = (
        np.std(
            gray_image
        )
    )

    return float(
        contrast
    )


# ============================================================
# IMAGE RESOLUTION
# ============================================================

def calculate_resolution_score(
    image: np.ndarray,
) -> float:
    """
    Calculate resolution quality score.
    """

    if image is None:

        return 0.0

    height, width = (
        image.shape[:2]
    )

    total_pixels = (
        width * height
    )

    # Approximately 2 megapixels
    # is considered excellent.

    target_pixels = (
        2_000_000
    )

    resolution_score = min(
        total_pixels
        / target_pixels,
        1.0,
    )

    return float(
        resolution_score
    )


# ============================================================
# HANDWRITING COVERAGE
# ============================================================

def calculate_handwriting_coverage(
    binary_image: np.ndarray,
) -> float:
    """
    Calculate the percentage of the image
    occupied by handwriting.
    """

    if binary_image is None:

        return 0.0

    if len(
        binary_image.shape
    ) == 3:

        gray_image = cv2.cvtColor(
            binary_image,
            cv2.COLOR_RGB2GRAY,
        )

    else:

        gray_image = binary_image

    # Handwriting pixels are assumed
    # to be dark.

    handwriting_pixels = np.sum(
        gray_image < 128
    )

    total_pixels = (
        gray_image.size
    )

    if total_pixels == 0:

        return 0.0

    coverage = (
        handwriting_pixels
        / total_pixels
    )

    return float(
        coverage
    )


# ============================================================
# NORMALIZE SHARPNESS SCORE
# ============================================================

def normalize_sharpness_score(
    sharpness: float,
) -> float:
    """
    Convert sharpness value into
    a score between 0 and 1.
    """

    target_sharpness = (
        300.0
    )

    score = min(
        sharpness
        / target_sharpness,
        1.0,
    )

    return float(
        max(
            0.0,
            score,
        )
    )


# ============================================================
# NORMALIZE BRIGHTNESS SCORE
# ============================================================

def normalize_brightness_score(
    brightness: float,
) -> float:
    """
    Images with moderate-to-high brightness
    receive higher scores.
    """

    ideal_brightness = (
        200.0
    )

    difference = abs(
        brightness
        - ideal_brightness
    )

    score = (
        1.0
        - difference
        / ideal_brightness
    )

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    return float(
        score
    )


# ============================================================
# NORMALIZE CONTRAST SCORE
# ============================================================

def normalize_contrast_score(
    contrast: float,
) -> float:
    """
    Convert contrast into
    a score between 0 and 1.
    """

    target_contrast = (
        70.0
    )

    score = min(
        contrast
        / target_contrast,
        1.0,
    )

    return float(
        max(
            0.0,
            score,
        )
    )


# ============================================================
# NORMALIZE COVERAGE SCORE
# ============================================================

def normalize_coverage_score(
    coverage: float,
) -> float:
    """
    Moderate handwriting coverage is ideal.
    """

    ideal_coverage = (
        0.12
    )

    difference = abs(
        coverage
        - ideal_coverage
    )

    score = (
        1.0
        - difference
        / ideal_coverage
    )

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    return float(
        score
    )


# ============================================================
# COMPLETE QUALITY ANALYSIS
# ============================================================

def analyze_sample_quality(
    original_image: np.ndarray,
    binary_image: np.ndarray,
) -> dict:
    """
    Perform complete handwriting
    sample quality analysis.
    """

    sharpness = (
        calculate_sharpness(
            original_image
        )
    )

    brightness = (
        calculate_brightness(
            original_image
        )
    )

    contrast = (
        calculate_contrast(
            original_image
        )
    )

    resolution_score = (
        calculate_resolution_score(
            original_image
        )
    )

    coverage = (
        calculate_handwriting_coverage(
            binary_image
        )
    )

    # --------------------------------------------------------
    # NORMALIZED SCORES
    # --------------------------------------------------------

    sharpness_score = (
        normalize_sharpness_score(
            sharpness
        )
    )

    brightness_score = (
        normalize_brightness_score(
            brightness
        )
    )

    contrast_score = (
        normalize_contrast_score(
            contrast
        )
    )

    coverage_score = (
        normalize_coverage_score(
            coverage
        )
    )

    # --------------------------------------------------------
    # FINAL QUALITY SCORE
    # --------------------------------------------------------

    quality_score = (
        0.30
        * sharpness_score

        + 0.20
        * brightness_score

        + 0.15
        * contrast_score

        + 0.20
        * resolution_score

        + 0.15
        * coverage_score
    )

    quality_score = (
        quality_score
        * 100
    )

    # --------------------------------------------------------
    # QUALITY LABEL
    # --------------------------------------------------------

    if quality_score >= 85:

        quality_label = (
            "Excellent"
        )

    elif quality_score >= 70:

        quality_label = (
            "Good"
        )

    elif quality_score >= 50:

        quality_label = (
            "Fair"
        )

    else:

        quality_label = (
            "Poor"
        )

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "resolution_score": (
            resolution_score
        ),
        "handwriting_coverage": (
            coverage
        ),
        "sharpness_score": (
            sharpness_score
        ),
        "brightness_score": (
            brightness_score
        ),
        "contrast_score": (
            contrast_score
        ),
        "coverage_score": (
            coverage_score
        ),
        "quality_score": (
            quality_score
        ),
        "quality_label": (
            quality_label
        ),
    }