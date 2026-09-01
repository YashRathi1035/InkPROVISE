import cv2
import numpy as np


# ============================================================
# BASIC IMAGE STATISTICS
# ============================================================

def calculate_ink_density(
    binary: np.ndarray,
) -> float:
    """
    Calculate the percentage of the image occupied
    by handwriting pixels.

    Binary image:
        Black  = handwriting
        White  = background
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    total_pixels = binary.size

    if total_pixels == 0:
        return 0.0

    ink_pixels = np.sum(
        binary < 128
    )

    density = (
        ink_pixels
        / total_pixels
    )

    return float(density)

# ============================================================
# HANDWRITING BOUNDING BOX
# ============================================================

def calculate_handwriting_bbox(
    binary: np.ndarray,
) -> dict:
    """
    Calculate the bounding box of the handwriting.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    foreground = (
        binary < 128
    ).astype(np.uint8)

    coords = cv2.findNonZero(
        foreground
    )

    if coords is None:

        return {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "area": 0,
        }

    x, y, width, height = (
        cv2.boundingRect(coords)
    )

    return {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "area": int(width * height),
    }

# ============================================================
# PAGE UTILIZATION
# ============================================================

def calculate_page_utilization(
    binary: np.ndarray,
) -> float:
    """
    Calculate how much of the page is occupied
    by handwriting.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    bbox = calculate_handwriting_bbox(
        binary
    )

    page_area = binary.shape[0] * binary.shape[1]

    if page_area == 0:
        return 0.0

    utilization = (
        bbox["area"]
        / page_area
    )

    return float(utilization)

# ============================================================
# MARGIN FEATURES
# ============================================================

def calculate_margins(
    binary: np.ndarray,
) -> dict:
    """
    Calculate approximate handwriting margins.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    bbox = calculate_handwriting_bbox(
        binary
    )

    height, width = binary.shape[:2]

    left_margin = bbox["x"]

    top_margin = bbox["y"]

    right_margin = (
        width
        - (
            bbox["x"]
            + bbox["width"]
        )
    )

    bottom_margin = (
        height
        - (
            bbox["y"]
            + bbox["height"]
        )
    )

    return {
        "left_margin": float(
            left_margin
        ),
        "right_margin": float(
            right_margin
        ),
        "top_margin": float(
            top_margin
        ),
        "bottom_margin": float(
            bottom_margin
        ),
    }

# ============================================================
# LINE HEIGHT FEATURES
# ============================================================

def calculate_line_features(
    lines: list[np.ndarray],
) -> dict:
    """
    Calculate statistical features from detected
    handwriting lines.
    """

    if not lines:

        return {
            "number_of_lines": 0,
            "mean_line_height": 0.0,
            "std_line_height": 0.0,
            "min_line_height": 0.0,
            "max_line_height": 0.0,
        }

    heights = [
        line.shape[0]
        for line in lines
    ]

    return {
        "number_of_lines": int(
            len(lines)
        ),
        "mean_line_height": float(
            np.mean(heights)
        ),
        "std_line_height": float(
            np.std(heights)
        ),
        "min_line_height": float(
            np.min(heights)
        ),
        "max_line_height": float(
            np.max(heights)
        ),
    }

# ============================================================
# LINE SPACING
# ============================================================

def calculate_line_spacing(
    lines: list[np.ndarray],
) -> dict:
    """
    Estimate vertical spacing between consecutive
    handwriting lines.
    """

    if len(lines) < 2:

        return {
            "mean_line_spacing": 0.0,
            "std_line_spacing": 0.0,
            "min_line_spacing": 0.0,
            "max_line_spacing": 0.0,
        }

    positions = []

    current_y = 0

    for line in lines:

        height = line.shape[0]

        positions.append(
            (
                current_y,
                current_y + height,
            )
        )

        current_y += height

    spacings = []

    for index in range(
        1,
        len(positions),
    ):

        previous_bottom = (
            positions[index - 1][1]
        )

        current_top = (
            positions[index][0]
        )

        spacing = (
            current_top
            - previous_bottom
        )

        spacings.append(
            spacing
        )

    if not spacings:

        return {
            "mean_line_spacing": 0.0,
            "std_line_spacing": 0.0,
            "min_line_spacing": 0.0,
            "max_line_spacing": 0.0,
        }

    return {
        "mean_line_spacing": float(
            np.mean(spacings)
        ),
        "std_line_spacing": float(
            np.std(spacings)
        ),
        "min_line_spacing": float(
            np.min(spacings)
        ),
        "max_line_spacing": float(
            np.max(spacings)
        ),
    }

# ============================================================
# WORD FEATURES
# ============================================================

def calculate_word_features(
    words_by_line: list[list[np.ndarray]],
) -> dict:
    """
    Calculate statistical features from segmented words.
    """

    all_words = []

    for words in words_by_line:

        all_words.extend(
            words
        )

    if not all_words:

        return {
            "number_of_words": 0,
            "mean_word_width": 0.0,
            "std_word_width": 0.0,
            "mean_word_height": 0.0,
            "std_word_height": 0.0,
        }

    widths = [
        word.shape[1]
        for word in all_words
    ]

    heights = [
        word.shape[0]
        for word in all_words
    ]

    return {
        "number_of_words": int(
            len(all_words)
        ),
        "mean_word_width": float(
            np.mean(widths)
        ),
        "std_word_width": float(
            np.std(widths)
        ),
        "mean_word_height": float(
            np.mean(heights)
        ),
        "std_word_height": float(
            np.std(heights)
        ),
    }

# ============================================================
# STROKE THICKNESS
# ============================================================

def estimate_stroke_thickness(
    binary: np.ndarray,
) -> float:
    """
    Estimate average handwriting stroke thickness.

    Uses a distance transform on the foreground.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    foreground = (
        binary < 128
    ).astype(np.uint8)

    if np.sum(foreground) == 0:
        return 0.0

    distance = cv2.distanceTransform(
        foreground,
        cv2.DIST_L2,
        5,
    )

    values = distance[
        foreground > 0
    ]

    if len(values) == 0:
        return 0.0

    # Distance represents approximate radius.
    # Multiplying by 2 gives approximate stroke width.
    stroke_width = (
        np.mean(values) * 2
    )

    return float(
        stroke_width
    )

# ============================================================
# CONNECTED COMPONENT FEATURES
# ============================================================

def calculate_connected_components(
    binary: np.ndarray,
) -> dict:
    """
    Analyze connected handwriting components.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    foreground = (
        binary < 128
    ).astype(np.uint8)

    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )
    )

    if number_labels <= 1:

        return {
            "number_of_components": 0,
            "mean_component_area": 0.0,
            "std_component_area": 0.0,
        }

    areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    return {
        "number_of_components": int(
            len(areas)
        ),
        "mean_component_area": float(
            np.mean(areas)
        ),
        "std_component_area": float(
            np.std(areas)
        ),
    }

# ============================================================
# SLANT ESTIMATION
# ============================================================

def estimate_slant(
    binary: np.ndarray,
) -> float:
    """
    Estimate global handwriting slant angle in degrees.

    Positive values indicate a forward/right slant.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    foreground = (
        binary < 128
    ).astype(np.uint8)

    height, width = (
        foreground.shape
    )

    if height == 0 or width == 0:
        return 0.0

    angles = []

    # Analyze vertical strips.
    for x in range(
        0,
        width,
        max(1, width // 100),
    ):

        column = foreground[
            :,
            x
        ]

        ys = np.where(
            column > 0
        )[0]

        if len(ys) < 2:
            continue

        y_min = np.min(ys)
        y_max = np.max(ys)

        if y_max == y_min:
            continue

        # Basic local estimate.
        angle = np.degrees(
            np.arctan2(
                x,
                y_max - y_min,
            )
        )

        angles.append(
            angle
        )

    if not angles:
        return 0.0

    return float(
        np.median(angles)
    )

# ============================================================
# COMPLETE HANDWRITING FEATURE VECTOR
# ============================================================

def extract_handwriting_features(
    binary: np.ndarray,
    lines: list[np.ndarray] | None = None,
    words_by_line: list[list[np.ndarray]] | None = None,
) -> dict:
    """
    Extract a complete set of handwriting features.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    if lines is None:
        lines = []

    if words_by_line is None:
        words_by_line = []

    bbox = calculate_handwriting_bbox(
        binary
    )

    margins = calculate_margins(
        binary
    )

    line_features = (
        calculate_line_features(
            lines
        )
    )

    line_spacing = (
        calculate_line_spacing(
            lines
        )
    )

    word_features = (
        calculate_word_features(
            words_by_line
        )
    )

    components = (
        calculate_connected_components(
            binary
        )
    )

    features = {

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        "image_width": float(
            binary.shape[1]
        ),

        "image_height": float(
            binary.shape[0]
        ),

        "ink_density": (
            calculate_ink_density(
                binary
            )
        ),

        # ----------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------

        "handwriting_width": float(
            bbox["width"]
        ),

        "handwriting_height": float(
            bbox["height"]
        ),

        "handwriting_area": float(
            bbox["area"]
        ),

        "page_utilization": (
            calculate_page_utilization(
                binary
            )
        ),

        # ----------------------------------------------------
        # MARGINS
        # ----------------------------------------------------

        "left_margin": margins[
            "left_margin"
        ],

        "right_margin": margins[
            "right_margin"
        ],

        "top_margin": margins[
            "top_margin"
        ],

        "bottom_margin": margins[
            "bottom_margin"
        ],

        # ----------------------------------------------------
        # LINES
        # ----------------------------------------------------

        **line_features,

        **line_spacing,

        # ----------------------------------------------------
        # WORDS
        # ----------------------------------------------------

        **word_features,

        # ----------------------------------------------------
        # STROKE
        # ----------------------------------------------------

        "estimated_stroke_thickness": (
            estimate_stroke_thickness(
                binary
            )
        ),

        # ----------------------------------------------------
        # SLANT
        # ----------------------------------------------------

        "estimated_slant": (
            estimate_slant(
                binary
            )
        ),

        # ----------------------------------------------------
        # CONNECTED COMPONENTS
        # ----------------------------------------------------

        **components,
    }

    return features