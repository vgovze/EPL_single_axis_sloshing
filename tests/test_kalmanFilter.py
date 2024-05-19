import unittest
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from context import myModules
from myModules.kalmanFilter import kalmanFilter, kinematic1dFilter

class test_kalmanFilter(unittest.TestCase):

    def test_kalmanFilter(self):

        # Test if initialization is ok.
        N = 5
        I = np.identity(N)
        filterInit = kalmanFilter(N)
        npt.assert_almost_equal(filterInit.getState(), np.zeros(N))
        npt.assert_almost_equal(filterInit.getControlModel(), I)
        npt.assert_almost_equal(filterInit.getStateCov(), I)
        npt.assert_almost_equal(filterInit.getPredictionCov(), I)
        npt.assert_almost_equal(filterInit.getPredictionModel(), I)
        npt.assert_almost_equal(filterInit.getObservationModel(), I)
        npt.assert_almost_equal(filterInit.getObservationCov(), I)

        # Test faulty inputs.
        test = np.array([1])
        with self.assertRaises(ValueError):
            filterInit.setControlModel(test)

        with self.assertRaises(ValueError):
            filterInit.setState(test)

        with self.assertRaises(ValueError):
            filterInit.setStateCov(test)

        with self.assertRaises(ValueError):
            filterInit.setPredictionCov(test)

        with self.assertRaises(ValueError):
            filterInit.setPredictionModel(test)

        with self.assertRaises(ValueError):
            filterInit.setObservationModel(test)

        with self.assertRaises(ValueError):
            filterInit.setObservationCov(test)

        # Test the Kalman Filter based on the model:
        # p1 = p0 + v0*dt + a0*dt**2/2
        # v1 =    + v0    + a0*dt
        # a1 =            + a0

        dt = 1 # time step.
        N = 3  # state space dimension.
        filter = kalmanFilter(N)

        # Initial state.
        x0 = np.ones(N)
        filter.setState(x0)
        npt.assert_almost_equal(x0, filter.getState())

        # Initial covariance.
        P0 = np.zeros((N,N))
        filter.setStateCov(P0)
        npt.assert_almost_equal(P0, filter.getStateCov())

        # State transition model.
        F = np.array([[1, dt, dt**2/2],
                      [0, 1,  dt     ],
                      [0, 0,  1      ]])
        filter.setPredictionModel(F)
        npt.assert_almost_equal(F, filter.getPredictionModel())

        # Control-input model.
        B = np.array([[0.5, 1,    2],
                      [3,   2,    1],
                      [0.2, 0.9, 10]])
        filter.setControlModel(B)
        npt.assert_almost_equal(B, filter.getControlModel())

        # Prediction covariance.
        Q = 3*np.identity(N)
        filter.setPredictionCov(Q)
        npt.assert_almost_equal(Q, filter.getPredictionCov())

        # Observation model
        H = np.array([[0.5, 1, 1],
                      [0, 1, 3],
                      [0, 0, 1]])
        filter.setObservationModel(H)
        npt.assert_almost_equal(H, filter.getObservationModel())

        # Observation covariance.
        R = 5*np.identity(N)
        filter.setObservationCov(R)
        npt.assert_almost_equal(R, filter.getObservationCov())

        # Control vector.
        c = np.array([1,1,-1])

        filter1 = deepcopy(filter)

        # True states list.
        x = []

        # Kalman Filter results.
        xFilter = []
        xFilter1 = []

        xcurr = x0
        for i in range(1,100):

            xcurr = F@xcurr + B@c
            x.append(xcurr)

            filter.step(H@xcurr, c)
            filter1.step(H@xcurr+np.random.normal(N), c)

            xFilter.append(filter.getState())
            xFilter1.append(filter1.getState())

        x = np.array(x).flatten()
        xFilter = np.array(xFilter).flatten()
        xFilter1 = np.array(xFilter1).flatten()

        # Assert x and xFilter are equal.
        npt.assert_almost_equal(x, xFilter)

        # Assert x and xFilter1 are not equal.
        npt.assert_raises(AssertionError, npt.assert_almost_equal, x, xFilter1)

    def test_kinematic1dFilter(self):

        time = np.linspace(0,2*np.pi,1000)
        displ = np.sin(time)
        vel = np.cos(time)
        acc = -np.sin(time)

        noiseStd = np.ones(3)*0.001
        measurementsFull = np.array((time, displ, vel, acc)).T
        measurements = measurementsFull.copy()

        # Test faulty inputs:
        with self.assertRaises(ValueError) as err:
            kinematic1dFilter(measurements[:,1:], noiseStd)

        with self.assertRaises(ValueError) as err:
            kinematic1dFilter(measurements, measurements)

        with self.assertRaises(ValueError) as err:
            kinematic1dFilter(measurements, noiseStd[1:])

        with self.assertRaises(ValueError) as err:
            kinematic1dFilter(time, noiseStd)

        with self.assertRaises(ValueError) as err:
            kinematic1dFilter(
                np.array((time[::-1], displ, vel, acc)).T, noiseStd
            )

        # Keep every 3rd element from the input array.
        for i in range(3):
            measurements[:,i+1] = np.where(
                np.arange(i+1,1001+i)%3 == 0, measurements[:,i+1], np.nan
            )

        filtered, covariances = kinematic1dFilter(measurements, noiseStd)

        npt.assert_allclose(
            filtered[1:,:], measurementsFull[1:,:], rtol=0, atol=2e-2
        )

if __name__ == "__main__":
    unittest.main()