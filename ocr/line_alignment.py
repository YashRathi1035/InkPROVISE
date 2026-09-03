# ============================================================
# CV LINE AND OCR LINE ALIGNMENT
# Phase 7C
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json

import numpy as np


# ============================================================
# VALIDATE RECTANGULAR BOUNDING BOX
# ============================================================

def validate_bounding_box(
    bounding_box: dict,
) -> None:
    """
    Validate a rectangular bounding box.
    """

    required_keys = [
        "x",
        "y",
        "width",
        "height",
    ]

    for key in required_keys:

        if key not in bounding_box:

            raise ValueError(
                f"Bounding box is missing '{key}'."
            )

    if bounding_box["width"] <= 0:

        raise ValueError(
            "Bounding box width must be positive."
        )

    if bounding_box["height"] <= 0:

        raise ValueError(
            "Bounding box height must be positive."
        )


# ============================================================
# CONVERT OCR POLYGON TO RECTANGLE
# ============================================================

def ocr_box_to_rectangle(
    bounding_box: list,
) -> dict:
    """
    Convert EasyOCR polygon coordinates
    into a rectangular bounding box.
    """

    if not bounding_box:

        raise ValueError(
            "OCR bounding box is empty."
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
        min(x_values)
    )

    x_max = float(
        max(x_values)
    )

    y_min = float(
        min(y_values)
    )

    y_max = float(
        max(y_values)
    )

    rectangle = {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }

    validate_bounding_box(
        rectangle
    )

    return rectangle


# ============================================================
# GET BOX CENTER
# ============================================================

def get_box_center(
    bounding_box: dict,
) -> tuple:
    """
    Return center coordinates of a box.
    """

    validate_bounding_box(
        bounding_box
    )

    center_x = (
        bounding_box["x"]
        + bounding_box["width"] / 2
    )

    center_y = (
        bounding_box["y"]
        + bounding_box["height"] / 2
    )

    return (
        center_x,
        center_y,
    )


# ============================================================
# GET CV LINE BOUNDING BOX
# ============================================================

def get_cv_line_bounding_box(
    line_data: dict,
    image_width: int,
) -> dict:
    """
    Convert Phase 6B extracted line metadata
    into a rectangular bounding box.

    Phase 6B format:

    {
        "line_index": ...,
        "start_row": ...,
        "end_row": ...,
        "image": ...
    }

    The extracted line spans the full image width.
    """

    required_keys = [
        "start_row",
        "end_row",
    ]

    for key in required_keys:

        if key not in line_data:

            raise ValueError(
                f"CV line data is missing '{key}'."
            )

    start_row = float(
        line_data[
            "start_row"
        ]
    )

    end_row = float(
        line_data[
            "end_row"
        ]
    )

    if end_row < start_row:

        raise ValueError(
            "end_row cannot be smaller than start_row."
        )

    bounding_box = {
        "x": 0.0,
        "y": start_row,
        "width": float(
            image_width
        ),
        "height": (
            end_row
            - start_row
            + 1
        ),
    }

    validate_bounding_box(
        bounding_box
    )

    return bounding_box


# ============================================================
# GET OCR LINE BOUNDING BOX
# ============================================================

def get_ocr_line_bounding_box(
    ocr_line: dict,
) -> dict:
    """
    Convert OCR line polygon into
    rectangular bounding box.
    """

    if "bounding_box" not in ocr_line:

        raise ValueError(
            "OCR line is missing bounding_box."
        )

    return (
        ocr_box_to_rectangle(
            ocr_line[
                "bounding_box"
            ]
        )
    )


# ============================================================
# CALCULATE VERTICAL OVERLAP
# ============================================================

def calculate_vertical_overlap(
    box_a: dict,
    box_b: dict,
) -> float:
    """
    Calculate normalized vertical overlap.
    """

    validate_bounding_box(
        box_a
    )

    validate_bounding_box(
        box_b
    )

    a_top = (
        box_a["y"]
    )

    a_bottom = (
        box_a["y"]
        + box_a["height"]
    )

    b_top = (
        box_b["y"]
    )

    b_bottom = (
        box_b["y"]
        + box_b["height"]
    )

    overlap_top = max(
        a_top,
        b_top,
    )

    overlap_bottom = min(
        a_bottom,
        b_bottom,
    )

    overlap = max(
        0.0,
        overlap_bottom
        - overlap_top,
    )

    minimum_height = min(
        box_a["height"],
        box_b["height"],
    )

    if minimum_height <= 0:

        return 0.0

    return float(
        overlap
        / minimum_height
    )


# ============================================================
# CALCULATE NORMALIZED CENTER DISTANCE
# ============================================================

def calculate_normalized_center_distance(
    box_a: dict,
    box_b: dict,
) -> float:
    """
    Calculate normalized vertical distance
    between the centers of two boxes.
    """

    _, center_a_y = (
        get_box_center(
            box_a
        )
    )

    _, center_b_y = (
        get_box_center(
            box_b
        )
    )

    distance = abs(
        center_a_y
        - center_b_y
    )

    average_height = (
        box_a["height"]
        + box_b["height"]
    ) / 2

    if average_height <= 0:

        return 1.0

    return float(
        distance
        / average_height
    )


# ============================================================
# CALCULATE ALIGNMENT SCORE
# ============================================================

def calculate_alignment_score(
    cv_box: dict,
    ocr_box: dict,
) -> dict:
    """
    Calculate spatial alignment score.
    """

    vertical_overlap = (
        calculate_vertical_overlap(
            cv_box,
            ocr_box,
        )
    )

    normalized_distance = (
        calculate_normalized_center_distance(
            cv_box,
            ocr_box,
        )
    )

    distance_score = max(
        0.0,
        1.0
        - normalized_distance,
    )

    alignment_score = (
        0.70
        * vertical_overlap
        +
        0.30
        * distance_score
    )

    return {
        "vertical_overlap":
            float(
                vertical_overlap
            ),

        "normalized_center_distance":
            float(
                normalized_distance
            ),

        "alignment_score":
            float(
                alignment_score
            ),
    }


# ============================================================
# FIND BEST OCR MATCH
# ============================================================

def find_best_ocr_match(
    cv_line: dict,
    ocr_lines: list,
    image_width: int,
) -> dict | None:
    """
    Find the best OCR line match
    for a CV extracted line.
    """

    if not ocr_lines:

        return None

    cv_box = (
        get_cv_line_bounding_box(
            line_data=cv_line,
            image_width=image_width,
        )
    )

    best_match = None

    best_score = -1.0

    for ocr_line in ocr_lines:

        ocr_box = (
            get_ocr_line_bounding_box(
                ocr_line
            )
        )

        score_data = (
            calculate_alignment_score(
                cv_box,
                ocr_box,
            )
        )

        alignment_score = (
            score_data[
                "alignment_score"
            ]
        )

        if alignment_score > best_score:

            best_score = (
                alignment_score
            )

            best_match = {
                "ocr_line":
                    ocr_line,

                "score_data":
                    score_data,

                "cv_box":
                    cv_box,

                "ocr_box":
                    ocr_box,
            }

    return best_match


# ============================================================
# ALIGN CV LINES WITH OCR LINES
# ============================================================

def align_cv_and_ocr_lines(
    cv_lines: list,
    ocr_lines: list,
    image_width: int,
    minimum_alignment_score: float = 0.30,
) -> list:
    """
    Align Phase 6B extracted lines
    with Phase 7B OCR lines.

    Each OCR line can only be used once.
    """

    alignments = []

    used_ocr_indices = set()

    for cv_position, cv_line in enumerate(
        cv_lines,
        start=1,
    ):

        available_ocr_lines = []

        for ocr_position, ocr_line in enumerate(
            ocr_lines,
            start=1,
        ):

            if (
                ocr_position
                not in used_ocr_indices
            ):

                available_ocr_lines.append(
                    (
                        ocr_position,
                        ocr_line,
                    )
                )

        if not available_ocr_lines:

            break

        available_lines_only = [
            item[1]
            for item
            in available_ocr_lines
        ]

        best_match = (
            find_best_ocr_match(
                cv_line=cv_line,
                ocr_lines=
                    available_lines_only,
                image_width=
                    image_width,
            )
        )

        if best_match is None:

            continue

        matched_ocr_line = (
            best_match[
                "ocr_line"
            ]
        )

        score_data = (
            best_match[
                "score_data"
            ]
        )

        alignment_score = (
            score_data[
                "alignment_score"
            ]
        )

        matched_ocr_index = None

        for (
            original_index,
            original_line,
        ) in available_ocr_lines:

            if (
                original_line
                is matched_ocr_line
            ):

                matched_ocr_index = (
                    original_index
                )

                break

        is_matched = (
            alignment_score
            >= minimum_alignment_score
        )

        alignment = {
            "cv_line_index":
                cv_line.get(
                    "line_index",
                    cv_position,
                ),

            "cv_line":
                cv_line,

            "matched":
                is_matched,

            "alignment_score":
                float(
                    alignment_score
                ),

            "vertical_overlap":
                score_data[
                    "vertical_overlap"
                ],

            "normalized_center_distance":
                score_data[
                    "normalized_center_distance"
                ],
        }

        if is_matched:

            alignment.update(
                {
                    "ocr_line_index":
                        matched_ocr_line.get(
                            "line_index",
                            matched_ocr_index,
                        ),

                    "ocr_line":
                        matched_ocr_line,

                    "text":
                        matched_ocr_line[
                            "text"
                        ],

                    "cv_bounding_box":
                        best_match[
                            "cv_box"
                        ],

                    "ocr_bounding_box":
                        best_match[
                            "ocr_box"
                        ],
                }
            )

            used_ocr_indices.add(
                matched_ocr_index
            )

        else:

            alignment.update(
                {
                    "ocr_line_index":
                        None,

                    "ocr_line":
                        None,

                    "text":
                        "",

                    "cv_bounding_box":
                        best_match[
                            "cv_box"
                        ],

                    "ocr_bounding_box":
                        best_match[
                            "ocr_box"
                        ],
                }
            )

        alignments.append(
            alignment
        )

    return alignments


# ============================================================
# CREATE LINE TRAINING PAIRS
# ============================================================

def create_line_training_pairs(
    alignments: list,
) -> list:
    """
    Create image-text metadata pairs
    from successful line alignments.
    """

    training_pairs = []

    for alignment in alignments:

        if not alignment[
            "matched"
        ]:

            continue

        text = (
            alignment[
                "text"
            ].strip()
        )

        if not text:

            continue

        cv_line = (
            alignment[
                "cv_line"
            ]
        )

        training_pair = {
            "cv_line_index":
                alignment[
                    "cv_line_index"
                ],

            "ocr_line_index":
                alignment[
                    "ocr_line_index"
                ],

            "text":
                text,

            "alignment_score":
                alignment[
                    "alignment_score"
                ],

            "vertical_overlap":
                alignment[
                    "vertical_overlap"
                ],

            "normalized_center_distance":
                alignment[
                    "normalized_center_distance"
                ],

            "start_row":
                cv_line[
                    "start_row"
                ],

            "end_row":
                cv_line[
                    "end_row"
                ],
        }

        # Optional metadata if lines
        # were already saved by Phase 6B.

        if "path" in cv_line:

            training_pair[
                "image_path"
            ] = (
                cv_line[
                    "path"
                ]
            )

        if "filename" in cv_line:

            training_pair[
                "filename"
            ] = (
                cv_line[
                    "filename"
                ]
            )

        training_pairs.append(
            training_pair
        )

    return training_pairs


# ============================================================
# SAVE TRAINING PAIRS
# ============================================================

def save_training_pairs(
    training_pairs: list,
    output_path: Path,
) -> Path:
    """
    Save training pairs as JSON.
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
            training_pairs,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path