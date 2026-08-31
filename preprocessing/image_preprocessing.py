import cv2
import numpy as np

def resize_image(
    image: np.ndarray,
    max_width: int = 4000,
    max_height: int = 4000,
) -> np.ndarray:
    """
    Resize an image if it exceeds the maximum
    allowed dimensions while preserving aspect ratio.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    height, width = image.shape[:2]

    if width <= max_width and height <= max_height:
        return image.copy()

    width_scale = max_width / width
    height_scale = max_height / height

    scale = min(
        width_scale,
        height_scale,
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized

def convert_to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert an RGB/BGR image to grayscale.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    if len(image.shape) == 2:
        return image.copy()

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

def remove_noise(
    grayscale: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Reduce high-frequency image noise using Gaussian blur.
    """

    if grayscale is None:
        raise ValueError(
            "Grayscale image cannot be None."
        )

    if kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be an odd number."
        )

    return cv2.GaussianBlur(
        grayscale,
        (kernel_size, kernel_size),
        0,
    )

def enhance_contrast(
    grayscale: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhance local contrast using CLAHE.
    """

    if grayscale is None:
        raise ValueError(
            "Grayscale image cannot be None."
        )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    enhanced = clahe.apply(grayscale)

    return enhanced

def apply_binary_threshold(
    grayscale: np.ndarray,
    threshold: int = 127,
) -> np.ndarray:
    """
    Convert grayscale image into a binary image
    using a fixed threshold.
    """

    if grayscale is None:
        raise ValueError(
            "Grayscale image cannot be None."
        )

    _, binary = cv2.threshold(
        grayscale,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    return binary

def apply_otsu_threshold(
    grayscale: np.ndarray,
) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding.
    """

    if grayscale is None:
        raise ValueError(
            "Grayscale image cannot be None."
        )

    _, binary = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary

def apply_adaptive_threshold(
    grayscale: np.ndarray,
    block_size: int = 21,
    c: int = 10,
) -> np.ndarray:
    """
    Apply adaptive Gaussian thresholding.
    """

    if grayscale is None:
        raise ValueError(
            "Grayscale image cannot be None."
        )

    if block_size % 2 == 0:
        raise ValueError(
            "Block size must be an odd number."
        )

    binary = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )

    return binary

def morphological_opening(
    binary: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Remove small isolated noise using morphological opening.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    return cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

def morphological_closing(
    binary: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Close small gaps and connect nearby foreground regions.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )