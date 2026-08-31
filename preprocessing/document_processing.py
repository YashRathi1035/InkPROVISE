import cv2
import numpy as np


# ============================================================
# FIND DOCUMENT CONTOUR
# ============================================================

def find_document_contour(
    image: np.ndarray,
) -> np.ndarray | None:
    """
    Detect the largest four-sided contour,
    assumed to be the handwritten page.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.20:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approximation) == 4:

            return approximation

    return None

# ============================================================
# ORDER DOCUMENT POINTS
# ============================================================

def order_document_points(
    points: np.ndarray,
) -> np.ndarray:
    """
    Order four document corners as:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    points = points.reshape(4, 2)

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    sums = points.sum(axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    differences = np.diff(
        points,
        axis=1,
    )

    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered

# ============================================================
# FOUR POINT PERSPECTIVE TRANSFORM
# ============================================================

def perspective_correct(
    image: np.ndarray,
    document_contour: np.ndarray,
) -> np.ndarray:
    """
    Correct perspective distortion of a detected document.
    """

    points = order_document_points(
        document_contour
    )

    top_left, top_right = points[0], points[1]
    bottom_right, bottom_left = points[2], points[3]

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    max_width = int(
        max(
            width_top,
            width_bottom,
        )
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    max_height = int(
        max(
            height_left,
            height_right,
        )
    )

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination,
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
    )

    return corrected

# ============================================================
# BACKGROUND NORMALIZATION
# ============================================================

def normalize_background(
    grayscale: np.ndarray,
    kernel_size: int = 31,
) -> np.ndarray:
    """
    Normalize uneven background illumination.
    """

    if grayscale is None:
        raise ValueError(
            "Grayscale image cannot be None."
        )

    if kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be odd."
        )

    background = cv2.morphologyEx(
        grayscale,
        cv2.MORPH_CLOSE,
        np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        ),
    )

    normalized = cv2.divide(
        grayscale,
        background,
        scale=255,
    )

    return normalized

# ============================================================
# DESKEW IMAGE
# ============================================================

def deskew_image(
    binary: np.ndarray,
) -> np.ndarray:
    """
    Correct the dominant skew angle of handwriting.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    coords = np.column_stack(
        np.where(binary < 128)
    )

    if len(coords) < 10:
        return binary.copy()

    angle = cv2.minAreaRect(
        coords.astype(np.float32)
    )[-1]

    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle

    height, width = binary.shape[:2]

    center = (
        width // 2,
        height // 2,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    deskewed = cv2.warpAffine(
        binary,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    return deskewed

# ============================================================
# EXTRACT HANDWRITING REGION
# ============================================================

def extract_handwriting_region(
    binary: np.ndarray,
) -> np.ndarray:
    """
    Crop the image around the handwriting.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    foreground = cv2.bitwise_not(
        binary
    )

    coords = cv2.findNonZero(
        foreground
    )

    if coords is None:
        return binary.copy()

    x, y, width, height = cv2.boundingRect(
        coords
    )

    padding = 20

    x1 = max(
        0,
        x - padding,
    )

    y1 = max(
        0,
        y - padding,
    )

    x2 = min(
        binary.shape[1],
        x + width + padding,
    )

    y2 = min(
        binary.shape[0],
        y + height + padding,
    )

    return binary[
        y1:y2,
        x1:x2,
    ]

# ============================================================
# LINE SEGMENTATION
# ============================================================

def segment_lines(
    binary: np.ndarray,
    minimum_height: int = 5,
) -> list[np.ndarray]:
    """
    Segment handwritten lines using horizontal
    projection analysis.
    """

    if binary is None:
        raise ValueError(
            "Binary image cannot be None."
        )

    # Black handwriting = 1
    foreground = (
        binary < 128
    ).astype(np.uint8)

    horizontal_projection = (
        foreground.sum(axis=1)
    )

    lines = []

    inside_line = False
    start = 0

    for y, value in enumerate(
        horizontal_projection
    ):

        if value > 0 and not inside_line:

            start = y
            inside_line = True

        elif (
            value == 0
            and inside_line
        ):

            end = y

            if (
                end - start
                >= minimum_height
            ):

                line = binary[
                    start:end,
                    :
                ]

                lines.append(line)

            inside_line = False

    if inside_line:

        end = len(
            horizontal_projection
        )

        if (
            end - start
            >= minimum_height
        ):

            lines.append(
                binary[
                    start:end,
                    :
                ]
            )

    return lines

# ============================================================
# WORD SEGMENTATION
# ============================================================

def segment_words(
    line: np.ndarray,
    minimum_width: int = 3,
) -> list[np.ndarray]:
    """
    Segment words from a single handwritten line.
    """

    if line is None:
        raise ValueError(
            "Line cannot be None."
        )

    foreground = (
        line < 128
    ).astype(np.uint8)

    vertical_projection = (
        foreground.sum(axis=0)
    )

    words = []

    inside_word = False
    start = 0

    for x, value in enumerate(
        vertical_projection
    ):

        if value > 0 and not inside_word:

            start = x
            inside_word = True

        elif (
            value == 0
            and inside_word
        ):

            end = x

            if (
                end - start
                >= minimum_width
            ):

                word = line[
                    :,
                    start:end
                ]

                words.append(word)

            inside_word = False

    if inside_word:

        end = len(
            vertical_projection
        )

        if (
            end - start
            >= minimum_width
        ):

            words.append(
                line[
                    :,
                    start:end
                ]
            )

    return words