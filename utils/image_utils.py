from io import BytesIO

import numpy as np

from PIL import Image


def uploaded_file_to_numpy(
    uploaded_file,
) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile into
    an RGB NumPy array.
    """

    image_bytes = uploaded_file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    return np.array(image)


def get_image_dimensions(
    image: np.ndarray,
) -> tuple[int, int]:
    """
    Return image width and height.
    """

    height, width = image.shape[:2]

    return width, height


def validate_image(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[bool, str]:
    """
    Validate an image against maximum dimensions.
    """

    if image is None:

        return False, "Image is empty."

    if image.ndim not in [2, 3]:

        return False, "Invalid image format."

    height, width = image.shape[:2]

    if width > max_width:

        return (
            False,
            f"Image width exceeds {max_width} pixels.",
        )

    if height > max_height:

        return (
            False,
            f"Image height exceeds {max_height} pixels.",
        )

    return True, "Image is valid."


def image_to_pil(
    image: np.ndarray,
) -> Image.Image:
    """
    Convert a NumPy image into a PIL Image.
    """

    return Image.fromarray(image)