import numpy as np


def apply_color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> np.ndarray:
    """Apply random brightness/contrast/saturation/hue jitter to a (C, H, W) uint8 image."""
    img = image.astype(np.float32) / 255.0
    brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
    img = np.clip(img * brightness_factor, 0, 1)

    contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
    mean = img.mean(axis=(1, 2), keepdims=True)
    img = np.clip((img - mean) * contrast_factor + mean, 0, 1)

    if img.shape[0] == 3:
        saturation_factor = 1.0 + np.random.uniform(-saturation, saturation)
        gray = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        img = np.clip(gray + (img - gray) * saturation_factor, 0, 1)

    if img.shape[0] == 3:
        hue_shift = np.random.uniform(-hue, hue)
        if abs(hue_shift) > 0.01:
            shift_amount = hue_shift * 0.5
            r, g, b = img[0], img[1], img[2]
            img = np.stack(
                [
                    np.clip(r + shift_amount * (g - b), 0, 1),
                    np.clip(g + shift_amount * (b - r), 0, 1),
                    np.clip(b + shift_amount * (r - g), 0, 1),
                ]
            )

    return (img * 255).astype(np.uint8)
