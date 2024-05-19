import numpy as np
from numpy import ndarray
import cv2 as cv

def autoCannyEdgeDetection(image: ndarray, sigma: float) -> ndarray:
    """The 1 parameter Canny edge detection.

    Args:
        image (ndarray): the image.
        sigma (ndarray): free parameter for the Auto-Canny edge detection.

    Returns:
        ndarray: binarized image with detected edges.
    """

    # Preprocess the image with a bilateral filter first.
    preprocessed = cv.bilateralFilter(image, 5, 50, 50)

    # Convert to grayscale.
    gray = cv.cvtColor(preprocessed, cv.COLOR_BGR2GRAY)

    # Automatic canny edge detection.
    median = np.median(gray)
    lower = max(0, int(1.0-sigma)*median)
    upper = min(255, int(1.0+sigma)*median)

    return cv.Canny(gray, threshold1=lower, threshold2=upper)

def cannyEdgeDetection(image: ndarray, lower: float=10,
        upper: float=100) -> ndarray:
    """Detect edges on the image using canny edge detection.

    Args:
        image (ndarray): image.
        lower (float): lower threshold of the Canny edge detection.
        upper (float): upper threshold of the Canny edge detection.

    Returns:
        ndarray: binarized image with detected edges.
    """

    # Preprocess the image with a bilateral filter first.
    preprocessed = cv.bilateralFilter(image, 5, 50, 50)

    # Convert to grayscale.
    gray = cv.cvtColor(preprocessed, cv.COLOR_BGR2GRAY)

    return cv.Canny(gray, threshold1=lower, threshold2=upper)

def contourDetection(binaryImg: ndarray, lengthFactor: float=0.1) -> ndarray:
    """Detect contours of the binary image.

    Args:
        binaryImg (ndarray): binary image.

    Returns:
        ndarray: contour points (N,2) of the image.
    """

    # Find contours.
    contours, hierarchy = cv.findContours(
        binaryImg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE
    )
    contours = [np.array([i.ravel() for i in j]) for j in contours]

    # Filter by length.
    length = [cv.arcLength(i.astype(np.float32), False) for i in contours]
    mask = length > lengthFactor*np.mean(length)
    filtered = [contours[i] for i,j in enumerate(mask) if j]

    return np.concatenate(filtered)