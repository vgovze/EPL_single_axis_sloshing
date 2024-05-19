from typing import Tuple

import numpy as np
from numpy import ndarray
import cv2 as cv
from scipy.spatial import KDTree

from myModules.utils import parametric2dLineFit

class gridPars():
    def __init__(self) -> None:
        pass

def findPoints(image: ndarray, minCircularity=0.85, minInertiaRatio=0.85,
        minConvexity=0.85) -> ndarray:
    """Find points in an image.

    Args:
        image (ndarray): the uint8 numpy image array (grayscale).
        minCircularity (float, optional): area to perimeter**2 ratio.
            1 for circle. Defaults to 0.85.
        minInertiaRatio (float, optional): measure of elongation of the blob.
            1 for circle. Defaults to 0.85.
        minConvexity (float, optional): blob area to area of it's convex hull
            ratio. 1 for circle. Defaults to 0.85.

    Raises:
        TypeError: if image is not of np.uint8 data type.
        ValueError: if the image is not 2D.

    Returns:
        ndarray: points (N,2) in image coordinates.
    """

    if (image.dtype != np.uint8)\
        and (image.dtype != np.float32):
        raise TypeError("Expected a uint8 or float32 image data type.")

    if (image.ndim != 2):
        raise ValueError("Expected a 2D image.")

    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByColor = False
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True
    params.minCircularity = minCircularity
    params.minConvexity = minConvexity
    params.minInertiaRatio = minInertiaRatio

    detector = cv.SimpleBlobDetector_create(params)
    ptsReco = detector.detect(image)

    return np.array([i.pt for i in ptsReco])

def findCorners(image: ndarray, qualityLevel=0.1, minDistance=20) -> ndarray:
    """Find corners of an image array.

    Args:
        image (ndarray): the image array.
        qualityLevel (float, optional): Corner quality level. Corners with
            a quality below qualityLevel*maxCornerQuality are rejected.
            Defaults to 0.1.
        minDistance (int, optional): minimum euclidian distance between
            corners. Defaults to 20.

    Raises:
        TypeError: if the image array is not of proper dtype.
        ValueError: if the image array is not 2-dimensional.

    Returns:
        ndarray: corners (N,2) in image coordinates.
    """

    if (image.ndim != 2):
        raise ValueError("Expected a 2D image.")

    if (image.dtype != np.uint8)\
        and (image.dtype != np.float32):
        raise TypeError("Expected a uint8 or float32 image data type.")

    corners = cv.goodFeaturesToTrack(
        image, maxCorners=0, qualityLevel=qualityLevel,
        minDistance=minDistance
    )

    # Use both eps and max iter criteria (first arg)
    # max 100 iterations (second arg)
    # accuracy 1e-7 pixel (third arg)
    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-7)

    # The half of the window size for refining corners.
    winSize = (int(minDistance/2),)*2
    corners = cv.cornerSubPix(image, corners, winSize, (-1,-1), CRITERIA)

    return np.array([i.ravel() for i in corners])

def mapSquare2dGrid(inPts: ndarray, gridSpacing: float = 1.)\
        -> Tuple[ndarray,ndarray]:
    """Assign an array of cartesian point coordinates with a given grid spacing
    to the given array of input points which are assumed to still represent a
    clean or slightly distorted uniform cartesian grid. It is assumed that a
    every point internal to the grid has a right, bottom, left and top point
    at approximately the same relative positions.

    The function will fail to return some (or all) of the corresponding grid
    points, if the internal points are not connected (they form isolated
    patches).

    Args:
        inPts (ndarray): the input points to assign a grid to of shape (N,2).
        gridSpacing (int): the spacing between the points. Defaults to 1.

    Raises:
        ValueError: if the points are not of shape (N,2).

    Returns:
        Tuple[ndarray,ndarray]: grid points (M,2) and the corresponding
            input points (M,2), where the grid positions could be determined.
    """

    ERR_MSG = "Expected an array of shape (N,2)."
    if inPts.ndim != 2:
        raise ValueError(ERR_MSG)

    if inPts.shape[1] != 2:
        raise ValueError(ERR_MSG)

    # Construct a KDTree for nearest neighbor querying.
    tree = KDTree(inPts)
    neigh = tree.query(inPts, 5)[1] # Use 5 nearest neighbors (central + 4).

    # Approximately align the grid points with the axes to make
    # the sort more robust.
    k1, k2, *c = parametric2dLineFit(inPts[:,0], inPts[:,1])
    alpha = np.arctan2(k2, k1)
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]])

    # Axes are reordered alphabetically in einsum (i - 0, j - 1)
    ptsAligned = np.einsum("jk,ik", R, inPts)

    # Sort the neighbors according to their orientation.
    # (center, right, bottom, left, top).
    sortedNeigh = neigh.copy()

    cp = []
    cosAngle = []
    for i,n in enumerate(neigh):
        pts = ptsAligned[n]-ptsAligned[i]

        left, right = np.argsort(pts[:,0])[[0,-1]]
        bottom, top = np.argsort(pts[:,1])[[0,-1]]
        sortedNeigh[i,1:] = n[[right, bottom, left, top]]

        # Calculate stencil orthogonality.
        a, b = pts[right]-pts[left], pts[top]-pts[bottom]
        a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
        cosAngle.append(a@b)

        # Calculate the stencil centerpoint.
        cp.append(np.linalg.norm(np.sum(inPts[i]-inPts[n[1:]], axis=0)))

    # Filter points.
    medianCp = np.median(cp)

    nodes = set([i for i,j in enumerate(inPts)])
    for i,j in enumerate(inPts):
        if cp[i] > 5*medianCp or cosAngle[i] > 0.05:
            nodes.remove(i)
    nodes = list(nodes)

    # Assign the corresponding positions to the points.
    outPts = np.zeros_like(inPts, dtype="float")
    isKnown = np.zeros(inPts.shape[0], dtype="bool")

    # The positions of neighboring points relative to the central point.
    REL_POSITION = gridSpacing*np.array([[0,0], [1,0], [0,-1], [-1,0], [0,1]])

    # The position of the central internal point is known (0,0).
    isKnown[nodes[0]] = True

    # Assign positions to the interior nodes. (cellular-automata)
    nIters = 0
    k = 1
    while k > 0 and nIters < inPts.shape[0]:
        k = 0
        for i in nodes:
            if isKnown[i]: continue

            n = np.argmax(isKnown[sortedNeigh[i,:]])
            if n == 0: continue

            neighIdx = sortedNeigh[i, n]
            outPts[i,:] = outPts[neighIdx,:] - REL_POSITION[n]
            isKnown[i] = True
            k += 1

        nIters += 1

    # Assign positions to the boundary nodes. (improper cellular-automata)
    nIters = 0
    k = 1
    while k > 0 and nIters < inPts.shape[0]:
        k = 0
        for i in nodes:
            n = np.argmin(isKnown[sortedNeigh[i,:]])
            if n == 0: continue

            neighIdx = sortedNeigh[i, n]
            outPts[neighIdx,:] = outPts[i,:] + REL_POSITION[n]
            isKnown[neighIdx] = True
            k += 1

        nIters += 1

    # Rotate the found points back.
    outPts = np.einsum("jk,ik", np.linalg.inv(R), outPts)

    return outPts[isKnown], inPts[isKnown]