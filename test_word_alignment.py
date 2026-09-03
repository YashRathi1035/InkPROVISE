# ============================================================
# TEST WORD LEVEL ALIGNMENT
# Phase 7D
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import cv2

from ocr.handwriting_ocr import (
    analyze_handwriting_image_lines,
)

from ocr.word_alignment import (
    create_word_image_pairs,
    save_word_image_pairs,
    save_word_pair_metadata,
)


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parent
)


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = (
    BASE_DIR
    / "test_handwriting.jpg"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "word_alignment_test"
)


WORDS_DIR = (
    OUTPUT_DIR
    / "words"
)


METADATA_PATH = (
    OUTPUT_DIR
    / "word_training_pairs.json"
)


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"\nTest image not found:\n"
        f"{IMAGE_PATH}\n"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(
        IMAGE_PATH
    )
)


if image is None:

    raise ValueError(
        "Unable to load test image."
    )


# ============================================================
# DISPLAY HEADER
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PHASE 7D - WORD LEVEL IMAGE AND TEXT ALIGNMENT"
)

print(
    "=" * 70
)


# ============================================================
# STEP 1 - OCR LINE ANALYSIS
# ============================================================

print(
    "\n[1/4] Running OCR..."
)


ocr_results = (
    analyze_handwriting_image_lines(
        image_path=
            IMAGE_PATH,

        languages=[
            "en"
        ],

        gpu=False,

        line_threshold_ratio=
            0.70,
    )
)


ocr_lines = (
    ocr_results[
        "lines"
    ]
)


print(
    f"OCR Lines Detected: "
    f"{len(ocr_lines)}"
)


# ============================================================
# STEP 2 - CREATE WORD PAIRS
# ============================================================

print(
    "\n[2/4] Extracting word images..."
)


word_pairs = (
    create_word_image_pairs(
        image=image,

        ocr_lines=
            ocr_lines,

        minimum_confidence=
            0.0,

        padding=
            5,
    )
)


print(
    f"Word Pairs Created: "
    f"{len(word_pairs)}"
)


# ============================================================
# STEP 3 - SAVE WORD IMAGES
# ============================================================

print(
    "\n[3/4] Saving word images..."
)


saved_word_pairs = (
    save_word_image_pairs(
        word_pairs=
            word_pairs,

        output_directory=
            WORDS_DIR,

        image_prefix=
            "handwriting",
    )
)


print(
    f"Word Images Saved: "
    f"{len(saved_word_pairs)}"
)


# ============================================================
# STEP 4 - SAVE METADATA
# ============================================================

print(
    "\n[4/4] Saving word metadata..."
)


saved_metadata_path = (
    save_word_pair_metadata(
        word_pairs=
            saved_word_pairs,

        output_path=
            METADATA_PATH,
    )
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "WORD ALIGNMENT RESULTS"
)

print(
    "=" * 70
)


for pair in saved_word_pairs:

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"Line Index: "
        f"{pair['line_index']}"
    )

    print(
        f"Word Index: "
        f"{pair['word_index']}"
    )

    print(
        f"Text: "
        f"{pair['text']}"
    )

    print(
        f"Confidence: "
        f"{pair['confidence']:.4f}"
    )

    print(
        f"Image: "
        f"{pair['path']}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PHASE 7D COMPLETE"
)

print(
    "=" * 70
)


print(
    f"\nTotal Word Images: "
    f"{len(saved_word_pairs)}"
)


print(
    f"\nWord Images Directory:\n"
    f"{WORDS_DIR}"
)


print(
    f"\nMetadata File:\n"
    f"{saved_metadata_path}"
)