# ============================================================
# WORD LEVEL IMAGE AND TEXT ALIGNMENT
# Phase 7D
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# GET BOUNDING RECTANGLE
# ============================================================

def bounding_box_to_rectangle(
    bounding_box: list,
) -> dict:
    """
    Convert a polygon bounding box into
    a rectangular bounding box.
    """

    if not bounding_box:

        raise ValueError(
            "Bounding box is empty."
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

    x_min = int(
        min(
            x_values
        )
    )

    x_max = int(
        max(
            x_values
        )
    )

    y_min = int(
        min(
            y_values
        )
    )

    y_max = int(
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

    if width <= 0:

        raise ValueError(
            "Bounding box width must be positive."
        )

    if height <= 0:

        raise ValueError(
            "Bounding box height must be positive."
        )

    return {
        "x": x_min,
        "y": y_min,
        "width": width,
        "height": height,
    }


# ============================================================
# ADD PADDING TO RECTANGLE
# ============================================================

def add_padding_to_rectangle(
    rectangle: dict,
    image_width: int,
    image_height: int,
    padding: int = 5,
) -> dict:
    """
    Add padding around a rectangle while
    keeping it inside image boundaries.
    """

    x = (
        rectangle["x"]
        - padding
    )

    y = (
        rectangle["y"]
        - padding
    )

    width = (
        rectangle["width"]
        + (
            padding
            * 2
        )
    )

    height = (
        rectangle["height"]
        + (
            padding
            * 2
        )
    )

    # --------------------------------------------------------
    # KEEP INSIDE IMAGE
    # --------------------------------------------------------

    x = max(
        0,
        x,
    )

    y = max(
        0,
        y,
    )

    x_end = min(
        image_width,
        x
        + width,
    )

    y_end = min(
        image_height,
        y
        + height,
    )

    width = (
        x_end
        - x
    )

    height = (
        y_end
        - y
    )

    if width <= 0:

        raise ValueError(
            "Invalid padded width."
        )

    if height <= 0:

        raise ValueError(
            "Invalid padded height."
        )

    return {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
    }


# ============================================================
# CROP WORD FROM IMAGE
# ============================================================

def crop_word_from_image(
    image: np.ndarray,
    bounding_box: list,
    padding: int = 5,
) -> tuple:
    """
    Crop a word image using its OCR
    bounding box.

    Returns:
        word_image
        rectangle
    """

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    if image.size == 0:

        raise ValueError(
            "Image is empty."
        )

    image_height, image_width = (
        image.shape[:2]
    )

    rectangle = (
        bounding_box_to_rectangle(
            bounding_box
        )
    )

    padded_rectangle = (
        add_padding_to_rectangle(
            rectangle=rectangle,
            image_width=image_width,
            image_height=image_height,
            padding=padding,
        )
    )

    x = (
        padded_rectangle["x"]
    )

    y = (
        padded_rectangle["y"]
    )

    width = (
        padded_rectangle["width"]
    )

    height = (
        padded_rectangle["height"]
    )

    word_image = image[
        y:y + height,
        x:x + width,
    ].copy()

    if word_image.size == 0:

        raise ValueError(
            "Extracted word image is empty."
        )

    return (
        word_image,
        padded_rectangle,
    )


# ============================================================
# GET WORD DETECTIONS FROM OCR LINES
# ============================================================

def get_word_detections(
    ocr_lines: list,
) -> list:
    """
    Extract individual word detections
    from structured OCR lines.
    """

    word_detections = []

    for line in ocr_lines:

        line_index = (
            line.get(
                "line_index",
                None,
            )
        )

        detections = (
            line.get(
                "detections",
                [],
            )
        )

        for detection_index, detection in enumerate(
            detections,
            start=1,
        ):

            text = str(
                detection.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:

                continue

            bounding_box = (
                detection.get(
                    "bounding_box",
                    None,
                )
            )

            if bounding_box is None:

                continue

            word_data = {
                "line_index":
                    line_index,

                "word_index":
                    detection_index,

                "text":
                    text,

                "confidence":
                    float(
                        detection.get(
                            "confidence",
                            0.0,
                        )
                    ),

                "bounding_box":
                    bounding_box,
            }

            word_detections.append(
                word_data
            )

    return word_detections


# ============================================================
# CREATE WORD IMAGE PAIRS
# ============================================================

def create_word_image_pairs(
    image: np.ndarray,
    ocr_lines: list,
    minimum_confidence: float = 0.0,
    padding: int = 5,
) -> list:
    """
    Create word image and text pairs
    using OCR detections.
    """

    word_detections = (
        get_word_detections(
            ocr_lines
        )
    )

    word_pairs = []

    for word_data in word_detections:

        confidence = (
            word_data[
                "confidence"
            ]
        )

        if (
            confidence
            < minimum_confidence
        ):

            continue

        try:

            word_image, rectangle = (
                crop_word_from_image(
                    image=image,
                    bounding_box=
                        word_data[
                            "bounding_box"
                        ],
                    padding=
                        padding,
                )
            )

        except ValueError:

            continue

        word_pair = {
            "line_index":
                word_data[
                    "line_index"
                ],

            "word_index":
                word_data[
                    "word_index"
                ],

            "text":
                word_data[
                    "text"
                ],

            "confidence":
                confidence,

            "bounding_box":
                rectangle,

            "image":
                word_image,
        }

        word_pairs.append(
            word_pair
        )

    return word_pairs


# ============================================================
# CLEAN TEXT FOR FILENAME
# ============================================================

def clean_text_for_filename(
    text: str,
) -> str:
    """
    Convert text into a safe filename.
    """

    cleaned_text = (
        text
        .strip()
        .lower()
    )

    allowed_characters = []

    for character in cleaned_text:

        if (
            character.isalnum()
            or character
            in [
                "_",
                "-",
            ]
        ):

            allowed_characters.append(
                character
            )

        elif character.isspace():

            allowed_characters.append(
                "_"
            )

    cleaned_text = (
        "".join(
            allowed_characters
        )
    )

    cleaned_text = (
        cleaned_text.strip(
            "_"
        )
    )

    if not cleaned_text:

        cleaned_text = (
            "unknown_word"
        )

    return cleaned_text


# ============================================================
# SAVE WORD IMAGE PAIRS
# ============================================================

def save_word_image_pairs(
    word_pairs: list,
    output_directory: Path,
    image_prefix: str = "word",
) -> list:
    """
    Save extracted word images.

    Returns metadata without NumPy arrays.
    """

    output_directory = (
        Path(
            output_directory
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_pairs = []

    for global_index, pair in enumerate(
        word_pairs,
        start=1,
    ):

        text = (
            pair[
                "text"
            ]
        )

        safe_text = (
            clean_text_for_filename(
                text
            )
        )

        line_index = (
            pair[
                "line_index"
            ]
        )

        word_index = (
            pair[
                "word_index"
            ]
        )

        filename = (
            f"{image_prefix}"
            f"_line_{line_index:03d}"
            f"_word_{word_index:03d}"
            f"_{safe_text}"
            f"_{global_index:04d}.png"
        )

        output_path = (
            output_directory
            / filename
        )

        word_image = (
            pair[
                "image"
            ]
        )

        success = (
            cv2.imwrite(
                str(
                    output_path
                ),
                word_image,
            )
        )

        if not success:

            raise RuntimeError(
                f"Unable to save word image: "
                f"{output_path}"
            )

        rectangle = (
            pair[
                "bounding_box"
            ]
        )

        saved_pair = {
            "global_index":
                global_index,

            "line_index":
                line_index,

            "word_index":
                word_index,

            "text":
                text,

            "confidence":
                float(
                    pair[
                        "confidence"
                    ]
                ),

            "bounding_box":
                rectangle,

            "filename":
                filename,

            "path":
                str(
                    output_path
                ),

            "width":
                int(
                    word_image.shape[
                        1
                    ]
                ),

            "height":
                int(
                    word_image.shape[
                        0
                    ]
                ),
        }

        saved_pairs.append(
            saved_pair
        )

    return saved_pairs


# ============================================================
# SAVE WORD PAIR METADATA
# ============================================================

def save_word_pair_metadata(
    word_pairs: list,
    output_path: Path,
) -> Path:
    """
    Save word image-text metadata
    as JSON.
    """

    output_path = (
        Path(
            output_path
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            word_pairs,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path