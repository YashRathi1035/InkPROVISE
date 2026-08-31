from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
UPLOADS_DIR = DATA_DIR / "uploads"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLES_DIR,
    UPLOADS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_NAME = "Personalized Handwriting AI"

APP_VERSION = "0.1.0"


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_IMAGE_TYPES = [
    "png",
    "jpg",
    "jpeg",
    "webp",
]


# ============================================================
# HANDWRITING PROFILE SETTINGS
# ============================================================

MIN_PROFILE_SAMPLES = 3
MAX_PROFILE_SAMPLES = 10


# ============================================================
# IMAGE SETTINGS
# ============================================================

MAX_IMAGE_WIDTH = 4000
MAX_IMAGE_HEIGHT = 4000