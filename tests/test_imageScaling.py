import numpy as np
import numpy.testing as npt
import cv2 as cv
import pytest
import matplotlib.pyplot as plt

from context import myModules
from myModules import imageScaling as imgs

def test_mapSquare2dGrid():
    u,v = np.mgrid[0:10,0:10]
    pts = np.array((u.flatten(), v.flatten())).T

    # Rotate pts for 45 degrees.
    alpha = np.pi/4
    R = np.array([[np.cos(alpha), -np.sin(alpha)], \
                  [np.sin(alpha), np.cos(alpha)]])
    pts = np.einsum("jk,ik", R, pts)

    with pytest.raises(ValueError) as err:
        imgs.mapSquare2dGrid(pts.T)

    pts_wo_corners = np.delete(pts, [0,9,-10,-1], axis=0)
    a = 0.2

    outPts, inPts = imgs.mapSquare2dGrid(pts, a)

    npt.assert_almost_equal(inPts, pts_wo_corners)

    outPts -= outPts[0,:]
    inPts -= inPts[0,:]
    assert npt.assert_almost_equal(outPts, a*inPts) == None

    with pytest.raises(ValueError) as e:
        imgs.mapSquare2dGrid(np.array((u.flatten())))

def test_findCorners():
    # Draw a simple binary checkerboard image.
    img = np.zeros((100,100), dtype="uint8")
    for i in range(100):
        for j in range(100):
            a = int(i/20)%2
            b = int(j/20)%2
            if (a==0 and b==1) or (a==1 and b==0):
                img[i,j]=255

    # Expected corner positions.
    u,v = np.mgrid[1:5, 1:5]*20-0.5
    pts = np.array((u.flatten(), v.flatten())).T

    corners = imgs.findCorners(img, minDistance=10)

    # Sort by x and y.
    ind = np.lexsort((pts[:,0], pts[:,1]))
    pts = pts[ind,:]

    ind = np.lexsort((corners[:,0], corners[:,1]))
    corners = corners[ind,:]

    npt.assert_almost_equal(corners, pts)

    with pytest.raises(ValueError) as err:
        imgs.findCorners(img[:,0])

    with pytest.raises(TypeError) as err:
        imgs.findCorners(img.astype(np.uint16))

def test_findPoints():
    # Draw points.
    u,v = np.mgrid[10:500:10, 10:500:10]
    pts = np.array((u.flatten(), v.flatten())).T

    img = np.zeros((500,500), dtype="uint8")
    for i in pts:
        cv.circle(img, tuple(i[::-1]), 3, 255, -1)

    # Find the points.
    points = imgs.findPoints(img, 0.01, 0.01, 0.01)

    # Sort points.
    pts = pts[np.lexsort((pts[:,0], pts[:,1])),:]
    points = points[np.lexsort((points[:,0], points[:,1])),:]

    npt.assert_almost_equal(points, pts)

    with pytest.raises(TypeError) as err:
        imgs.findPoints(img.astype(np.uint16))

    with pytest.raises(ValueError) as err:
        imgs.findPoints(img[:,0])

if __name__ == "__main__":
    pass