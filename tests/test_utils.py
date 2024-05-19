import numpy as np
import numpy.testing as npt
import pytest

from context import myModules
from myModules import utils as ut

def test_parametric2dLineFit():
    N = 1000
    k1, k2, c1, c2 = 0.1, -5.23, -2, 1

    t = np.linspace(0,1,N)
    idx = np.arange(N)
    np.random.shuffle(idx)
    x, y = (k1*t+c1)[idx], (k2*t+c2)[idx]

    pytest.raises(ValueError, ut.parametric2dLineFit, x, np.ones((2,2)))
    pytest.raises(ValueError, ut.parametric2dLineFit, x, x[:2])

    K1, K2, C1, C2 = ut.parametric2dLineFit(x,y)

    # The points where lines cross x and y axis should be almost equal.
    tx, ty = -c1/k1, -c2/k2
    cross0 = k1*ty+c1, k2*tx+c2
    tX, tY = -C1/K1, -C2/K2
    cross1 = K1*tY+C1, K2*tX+C2

    npt.assert_allclose(cross1, cross0)

def test_findAbsMaxDiffIdx():
    arr = np.array([0,1,2,1,3,0,1,7,0,5,10,12,10])

    pytest.raises(ValueError, ut.findAbsMaxDiffIdx, np.ones((2,2)))

    assert ut.findAbsMaxDiffIdx(arr) == 6

def test_findMotionStartIdx():
    arr = np.array([-0.1,0,-1,0,0,0,1,1,1,1,1,1,1])

    pytest.raises(ValueError, ut.findChangeStartIdx, np.ones((2,2)), 4)

    assert ut.findChangeStartIdx(arr, 4) == 6

if __name__ == "__main__":
    pass