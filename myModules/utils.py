import numpy as np
from numpy import ndarray
from numba import jit

def parametric2dLineFit(x:ndarray, y:ndarray) -> ndarray:
    """Return the coefficients of the geometric fit of a line to data points
    x and y. The fitted curves are defined as
        x = k_1 * t + c_1 and
        y = k_2 * t + c_2
    and are both parametrized in the range t = [0,1].

    Args:
        x (ndarray): input 1D real array representing x points.
        y (ndarray): input 1D real array representing y points.

    Returns:
        ndarray: coefficients of the fit in order (k1, k2, c1, c2).
    """
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("Expected x and y to be 1-dimensional.")

    if y.size != x.size:
        raise ValueError("Expected x and y to be of equal size.")

    varX, varY = np.var(x), np.var(y)

    if varX > varY:
        idx = np.argsort(x)
    else:
        idx = np.argsort(y)

    # Parametrization variable.
    t = np.linspace(0,1,x.size)

    k1, c1 = np.polyfit(t, x[idx], deg=1)
    k2, c2 = np.polyfit(t, y[idx], deg=1)

    return np.array([k1, k2, c1, c2])

@jit(nopython=True)
def findAbsMaxDiffIdx(arr: ndarray) -> int:
    """Find the index at which the forward difference between two elements of
    the moving maximum of the absolute value of input array is the largest.

    Args:
        arr (ndarray): the input 1D array.

    Returns:
        int: the index of the largest difference.
    """
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")

    movingMax = np.abs(arr)
    for i in range(1,movingMax.size):
        movingMax[i] = max(movingMax[i], movingMax[i-1])

    return np.argmax(np.diff(movingMax))

@jit(nopython=True)
def findChangeStartIdx(arr: ndarray, n: int) -> int:
    """Find the index at which the array starts consistently changing
    the value. The function assumes that the array begins with values
    oscillating around 0. The index at which motion starts is the one
    after which n elements in the array have the same sign.

    Args:
        arr (ndarray): the input 1D array.
        n (int): number of elements in a sequence having the same sign.

    Raises:
        ValueError: if the input is not a 1D array.

    Returns:
        int: the change start index.
    """
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")

    signPrev = 1
    k = 0
    for i,j in enumerate(arr):
        sign = np.sign(j)
        if sign == signPrev:
            k += 1
        else:
            signPrev = sign
            k = 0
        if k == n: break
    return i-n