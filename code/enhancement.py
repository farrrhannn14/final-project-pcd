"""E · Gaussian, CLAHE, dan black-hat pada resolusi asli."""


import cv2
import numpy as np


PREPROCESSING = {
    "blur": 5,
    "sigma": 2.5,
    "clip": 1.0,
    "blackhat": 15,
    "dilate": 3,
    "area": 30,
    "canny_low": 0.5,
    "canny_high": 1.0,
    "max_side": None,
}


def enhance_image(image, config):
    blurred = cv2.GaussianBlur(image, (config["blur"],) * 2, sigmaX=config["sigma"], sigmaY=config["sigma"])
    clahe = cv2.createCLAHE(clipLimit=config["clip"], tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    kernel = np.ones((config["blackhat"],) * 2, dtype=np.uint8)
    enhanced = cv2.morphologyEx(equalized, cv2.MORPH_BLACKHAT, kernel)
    return blurred, equalized, enhanced
