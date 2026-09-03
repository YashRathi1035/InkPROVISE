# ============================================================
# HANDWRITING OCR
# Phase 7A
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import cv2
import numpy as np
import easyocr


# ============================================================
# OCR READER CACHE
# ============================================================

_OCR_READER = None


# ============================================================
# GET OCR READER
# ============================================================

def get_ocr_reader(
    languages: list = None,
    gpu: bool = False,
):
    """
    Create or reuse an EasyOCR reader.

    The reader is cached because loading
    the OCR model repeatedly is slow.
    """

    global _OCR_READER

    if languages is None:

        languages = [
            "en"
        ]

    if _OCR_READER is None:

        _OCR_READER = (
            easyocr.Reader(
                languages,
                gpu=gpu,
            )
        )

    return _OCR_READER


# ============================================================
# VALIDATE IMAGE
# ============================================================

def validate_image(
    image: np.ndarray,
) -> None:
    """
    Validate an image before OCR.
    """

    if image is None:

        raise ValueError(
            "Input image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):

        raise TypeError(
            "Input image must be a NumPy array."
        )

    if image.size == 0:

        raise ValueError(
            "Input image is empty."
        )


# ============================================================
# PREPROCESS FOR OCR
# ============================================================

def preprocess_for_ocr(
    image: np.ndarray,
) -> np.ndarray:
    """
    Perform light preprocessing before OCR.

    We intentionally avoid aggressive
    preprocessing because it can damage
    handwriting strokes.
    """

    validate_image(
        image
    )

    if len(
        image.shape
    ) == 3:

        gray_image = (
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        )

    else:

        gray_image = (
            image.copy()
        )

    # --------------------------------------------------------
    # LIGHT DENOISING
    # --------------------------------------------------------

    denoised_image = (
        cv2.GaussianBlur(
            gray_image,
            (3, 3),
            0,
        )
    )

    # --------------------------------------------------------
    # CONTRAST ENHANCEMENT
    # --------------------------------------------------------

    enhanced_image = (
        cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(
            denoised_image
        )
    )

    return enhanced_image


# ============================================================
# RUN OCR
# ============================================================

def run_handwriting_ocr(
    image: np.ndarray,
    languages: list = None,
    gpu: bool = False,
) -> list:
    """
    Run OCR on a handwriting image.

    Returns OCR detections containing:

    - bounding box
    - recognized text
    - confidence
    """

    validate_image(
        image
    )

    processed_image = (
        preprocess_for_ocr(
            image
        )
    )

    reader = (
        get_ocr_reader(
            languages=languages,
            gpu=gpu,
        )
    )

    results = (
        reader.readtext(
            processed_image,
            detail=1,
            paragraph=False,
        )
    )

    detections = []

    for result in results:

        bounding_box = (
            result[0]
        )

        text = (
            result[1]
        )

        confidence = (
            float(
                result[2]
            )
        )

        detections.append(
            {
                "bounding_box":
                    bounding_box,

                "text":
                    text,

                "confidence":
                    confidence,
            }
        )

    return detections


# ============================================================
# SORT OCR DETECTIONS
# ============================================================

def sort_ocr_detections(
    detections: list,
) -> list:
    """
    Sort OCR detections from top to bottom
    and left to right.
    """

    def get_position(
        detection,
    ):

        bounding_box = (
            detection[
                "bounding_box"
            ]
        )

        x_values = [
            point[0]
            for point
            in bounding_box
        ]

        y_values = [
            point[1]
            for point
            in bounding_box
        ]

        return (
            min(
                y_values
            ),
            min(
                x_values
            ),
        )

    sorted_detections = sorted(
        detections,
        key=get_position,
    )

    return sorted_detections


# ============================================================
# EXTRACT COMPLETE TEXT
# ============================================================

def extract_complete_text(
    detections: list,
) -> str:
    """
    Combine OCR detections into text.
    """

    sorted_detections = (
        sort_ocr_detections(
            detections
        )
    )

    text_parts = [
        detection[
            "text"
        ]
        for detection
        in sorted_detections
    ]

    complete_text = (
        " ".join(
            text_parts
        )
    )

    return complete_text.strip()


# ============================================================
# CALCULATE OCR STATISTICS
# ============================================================

def calculate_ocr_statistics(
    detections: list,
) -> dict:
    """
    Calculate basic OCR statistics.
    """

    if not detections:

        return {
            "detection_count": 0,
            "average_confidence": 0.0,
            "minimum_confidence": 0.0,
            "maximum_confidence": 0.0,
        }

    confidences = [
        detection[
            "confidence"
        ]
        for detection
        in detections
    ]

    return {
        "detection_count":
            len(
                detections
            ),

        "average_confidence":
            float(
                np.mean(
                    confidences
                )
            ),

        "minimum_confidence":
            float(
                np.min(
                    confidences
                )
            ),

        "maximum_confidence":
            float(
                np.max(
                    confidences
                )
            ),
    }


# ============================================================
# COMPLETE OCR PIPELINE
# ============================================================

def analyze_handwriting_with_ocr(
    image: np.ndarray,
    languages: list = None,
    gpu: bool = False,
) -> dict:
    """
    Complete handwriting OCR pipeline.
    """

    detections = (
        run_handwriting_ocr(
            image=image,
            languages=languages,
            gpu=gpu,
        )
    )

    sorted_detections = (
        sort_ocr_detections(
            detections
        )
    )

    complete_text = (
        extract_complete_text(
            sorted_detections
        )
    )

    statistics = (
        calculate_ocr_statistics(
            sorted_detections
        )
    )

    return {
        "detections":
            sorted_detections,

        "complete_text":
            complete_text,

        "statistics":
            statistics,
    }


# ============================================================
# LOAD IMAGE FROM PATH
# ============================================================

def load_image(
    image_path: Path,
) -> np.ndarray:
    """
    Load an image from disk.
    """

    image = (
        cv2.imread(
            str(
                image_path
            )
        )
    )

    if image is None:

        raise ValueError(
            f"Unable to load image:\n"
            f"{image_path}"
        )

    return image


# ============================================================
# ANALYZE IMAGE PATH
# ============================================================

def analyze_handwriting_image(
    image_path: Path,
    languages: list = None,
    gpu: bool = False,
) -> dict:
    """
    Load a handwriting image and
    run the complete OCR pipeline.
    """

    image = (
        load_image(
            image_path
        )
    )

    results = (
        analyze_handwriting_with_ocr(
            image=image,
            languages=languages,
            gpu=gpu,
        )
    )

    results[
        "image_path"
    ] = (
        str(
            image_path
        )
    )

    results[
        "image_name"
    ] = (
        image_path.name
    )

    return results

# ============================================================
# OCR LINE EXTRACTION
# Phase 7B
# ============================================================


# ============================================================
# GET DETECTION BOUNDING BOX VALUES
# ============================================================

def get_detection_box_values(
    detection: dict,
) -> dict:
    """
    Calculate useful bounding box values
    for an OCR detection.
    """

    bounding_box = (
        detection[
            "bounding_box"
        ]
    )

    x_values = [
        point[0]
        for point
        in bounding_box
    ]

    y_values = [
        point[1]
        for point
        in bounding_box
    ]

    x_min = float(
        min(
            x_values
        )
    )

    x_max = float(
        max(
            x_values
        )
    )

    y_min = float(
        min(
            y_values
        )
    )

    y_max = float(
        max(
            y_values
        )
    )

    width = (
        x_max
        - x_min
    )

    height = (
        y_max
        - y_min
    )

    center_x = (
        x_min
        + width / 2
    )

    center_y = (
        y_min
        + height / 2
    )

    return {
        "x_min":
            x_min,

        "x_max":
            x_max,

        "y_min":
            y_min,

        "y_max":
            y_max,

        "width":
            width,

        "height":
            height,

        "center_x":
            center_x,

        "center_y":
            center_y,
    }


# ============================================================
# CALCULATE AVERAGE DETECTION HEIGHT
# ============================================================

def calculate_average_detection_height(
    detections: list,
) -> float:
    """
    Calculate average OCR detection height.
    """

    if not detections:

        return 0.0

    heights = []

    for detection in detections:

        box_values = (
            get_detection_box_values(
                detection
            )
        )

        heights.append(
            box_values[
                "height"
            ]
        )

    return float(
        np.mean(
            heights
        )
    )


# ============================================================
# SORT DETECTIONS LEFT TO RIGHT
# ============================================================

def sort_detections_left_to_right(
    detections: list,
) -> list:
    """
    Sort detections from left to right.
    """

    return sorted(
        detections,
        key=lambda detection:
            get_detection_box_values(
                detection
            )[
                "x_min"
            ],
    )


# ============================================================
# GROUP OCR DETECTIONS INTO LINES
# ============================================================

def group_detections_into_lines(
    detections: list,
    line_threshold_ratio: float = 0.70,
) -> list:
    """
    Group OCR detections into text lines
    based on vertical position.

    line_threshold_ratio controls how much
    vertical distance is allowed between
    detections belonging to the same line.
    """

    if not detections:

        return []

    average_height = (
        calculate_average_detection_height(
            detections
        )
    )

    if average_height <= 0:

        average_height = 20.0

    vertical_threshold = (
        average_height
        * line_threshold_ratio
    )

    # --------------------------------------------------------
    # SORT TOP TO BOTTOM
    # --------------------------------------------------------

    sorted_detections = sorted(
        detections,
        key=lambda detection:
            get_detection_box_values(
                detection
            )[
                "center_y"
            ],
    )

    lines = []

    # --------------------------------------------------------
    # CREATE FIRST LINE
    # --------------------------------------------------------

    current_line = [
        sorted_detections[
            0
        ]
    ]

    current_line_centers = [
        get_detection_box_values(
            sorted_detections[
                0
            ]
        )[
            "center_y"
        ]
    ]

    # --------------------------------------------------------
    # PROCESS REMAINING DETECTIONS
    # --------------------------------------------------------

    for detection in (
        sorted_detections[
            1:
        ]
    ):

        detection_center_y = (
            get_detection_box_values(
                detection
            )[
                "center_y"
            ]
        )

        current_line_center_y = float(
            np.mean(
                current_line_centers
            )
        )

        vertical_distance = abs(
            detection_center_y
            - current_line_center_y
        )

        # ----------------------------------------------------
        # SAME LINE
        # ----------------------------------------------------

        if (
            vertical_distance
            <= vertical_threshold
        ):

            current_line.append(
                detection
            )

            current_line_centers.append(
                detection_center_y
            )

        # ----------------------------------------------------
        # NEW LINE
        # ----------------------------------------------------

        else:

            lines.append(
                current_line
            )

            current_line = [
                detection
            ]

            current_line_centers = [
                detection_center_y
            ]

    # --------------------------------------------------------
    # ADD FINAL LINE
    # --------------------------------------------------------

    if current_line:

        lines.append(
            current_line
        )

    return lines


# ============================================================
# CREATE LINE BOUNDING BOX
# ============================================================

def create_line_bounding_box(
    detections: list,
) -> list:
    """
    Create one bounding box covering
    all detections in a line.
    """

    if not detections:

        return []

    x_values = []
    y_values = []

    for detection in detections:

        bounding_box = (
            detection[
                "bounding_box"
            ]
        )

        for point in bounding_box:

            x_values.append(
                point[
                    0
                ]
            )

            y_values.append(
                point[
                    1
                ]
            )

    x_min = float(
        min(
            x_values
        )
    )

    x_max = float(
        max(
            x_values
        )
    )

    y_min = float(
        min(
            y_values
        )
    )

    y_max = float(
        max(
            y_values
        )
    )

    return [
        [
            x_min,
            y_min,
        ],
        [
            x_max,
            y_min,
        ],
        [
            x_max,
            y_max,
        ],
        [
            x_min,
            y_max,
        ],
    ]


# ============================================================
# CALCULATE LINE CONFIDENCE
# ============================================================

def calculate_line_confidence(
    detections: list,
) -> float:
    """
    Calculate average OCR confidence
    for one line.
    """

    if not detections:

        return 0.0

    confidences = [
        detection[
            "confidence"
        ]
        for detection
        in detections
    ]

    return float(
        np.mean(
            confidences
        )
    )


# ============================================================
# CREATE STRUCTURED OCR LINES
# ============================================================

def create_structured_ocr_lines(
    detections: list,
    line_threshold_ratio: float = 0.70,
) -> list:
    """
    Convert OCR detections into
    structured text lines.
    """

    grouped_lines = (
        group_detections_into_lines(
            detections=
                detections,

            line_threshold_ratio=
                line_threshold_ratio,
        )
    )

    structured_lines = []

    for line_index, line_detections in enumerate(
        grouped_lines,
        start=1,
    ):

        # ----------------------------------------------------
        # SORT LEFT TO RIGHT
        # ----------------------------------------------------

        sorted_line_detections = (
            sort_detections_left_to_right(
                line_detections
            )
        )

        # ----------------------------------------------------
        # CREATE LINE TEXT
        # ----------------------------------------------------

        line_text_parts = [
            detection[
                "text"
            ]
            for detection
            in sorted_line_detections
        ]

        line_text = (
            " ".join(
                line_text_parts
            ).strip()
        )

        # ----------------------------------------------------
        # CREATE LINE BOUNDING BOX
        # ----------------------------------------------------

        line_bounding_box = (
            create_line_bounding_box(
                sorted_line_detections
            )
        )

        # ----------------------------------------------------
        # CREATE LINE DATA
        # ----------------------------------------------------

        line_data = {
            "line_index":
                line_index,

            "text":
                line_text,

            "bounding_box":
                line_bounding_box,

            "detections":
                sorted_line_detections,

            "detection_count":
                len(
                    sorted_line_detections
                ),

            "average_confidence":
                calculate_line_confidence(
                    sorted_line_detections
                ),
        }

        structured_lines.append(
            line_data
        )

    return structured_lines


# ============================================================
# ANALYZE HANDWRITING OCR LINES
# ============================================================

def analyze_handwriting_ocr_lines(
    image: np.ndarray,
    languages: list = None,
    gpu: bool = False,
    line_threshold_ratio: float = 0.70,
) -> dict:
    """
    Complete OCR line extraction pipeline.
    """

    # --------------------------------------------------------
    # RUN OCR
    # --------------------------------------------------------

    ocr_results = (
        analyze_handwriting_with_ocr(
            image=image,
            languages=languages,
            gpu=gpu,
        )
    )

    detections = (
        ocr_results[
            "detections"
        ]
    )

    # --------------------------------------------------------
    # CREATE STRUCTURED LINES
    # --------------------------------------------------------

    structured_lines = (
        create_structured_ocr_lines(
            detections=
                detections,

            line_threshold_ratio=
                line_threshold_ratio,
        )
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "complete_text":
            ocr_results[
                "complete_text"
            ],

        "detections":
            detections,

        "statistics":
            ocr_results[
                "statistics"
            ],

        "lines":
            structured_lines,

        "line_count":
            len(
                structured_lines
            ),
    }


# ============================================================
# ANALYZE HANDWRITING IMAGE OCR LINES
# ============================================================

def analyze_handwriting_image_lines(
    image_path: Path,
    languages: list = None,
    gpu: bool = False,
    line_threshold_ratio: float = 0.70,
) -> dict:
    """
    Load an image and extract
    structured OCR lines.
    """

    image = (
        load_image(
            image_path
        )
    )

    results = (
        analyze_handwriting_ocr_lines(
            image=image,
            languages=languages,
            gpu=gpu,
            line_threshold_ratio=
                line_threshold_ratio,
        )
    )

    results[
        "image_path"
    ] = (
        str(
            image_path
        )
    )

    results[
        "image_name"
    ] = (
        image_path.name
    )

    return results