from typing import Tuple

import numpy as np
from numpy import ndarray

class kalmanFilter():

    def __init__(self, N: int) -> None:
        """Instantiate a Kalman filter object with a given dimension of state
        space. By default, all models and covariances are set to be identity
        matrices.

        Args:
            N (int): state space dimension.
        """
        self._N = N # state space dimensions.

        # Default initialization of the model.
        self._x: ndarray = np.zeros(N)
        self._P: ndarray = np.identity(N)
        self._F: ndarray = np.identity(N)
        self._B: ndarray = np.identity(N)
        self._Q: ndarray = np.identity(N)
        self._H: ndarray = np.identity(N)
        self._R: ndarray = np.identity(N)

        # Operating parameters of the model.
        self._K: ndarray = np.identity(N)

    def setState(self, x: ndarray) -> None:
        """Set the state vector.

        Args:
            x (ndarray): state vector of shape (N,).
        """
        if x.shape != (self._N,):
            raise ValueError("Expected input of shape (N,)")
        self._x = x

    def getState(self) -> ndarray:
        """Get the state vector of the current state.

        Returns:
            ndarray: state vector of shape (N,).
        """
        return(self._x)

    def setStateCov(self, P: ndarray) -> None:
        """Set the state covariance matrix.

        Args:
            P (ndarray): state covariance matrix of shape (N,N).
        """
        if P.shape != (self._N,)*2:
            raise ValueError("Expected input of shape (N,)")
        self._P = P

    def getStateCov(self) -> ndarray:
        """Get the state covariance matrix of the current state.

        Returns:
            ndarray: state covariance matrix of shape (N,N).
        """
        return(self._P)

    def setPredictionModel(self, F: ndarray) -> None:
        """Set the prediction model matrix.

        Args:
            F (ndarray): prediction model matrix of shape (N,N).
        """
        if F.shape != (self._N,)*2:
            raise ValueError("Expected input of shape (N,)")
        self._F = F

    def getPredictionModel(self) -> ndarray:
        """Get the prediction model matrix.

        Returns:
            ndarray: prediction model matrix of shape (N,N).
        """
        return(self._F)

    def setControlModel(self, B: ndarray) -> None:
        """Set the control model matrix.

        Args:
            B (ndarray): control-input model matrix of shape (N,N).
        """
        if B.shape != (self._N,)*2:
            raise ValueError("Expected input of shape (N,)")
        self._B = B

    def getControlModel(self) -> ndarray:
        """Get the control model matrix.

        Returns:
            ndarray: control-input model matrix of shape (N,N).
        """
        return(self._B)

    def setPredictionCov(self, Q: ndarray) -> None:
        """Set the prediction covariance matrix.

        Args:
            Q (ndarray): prediction covariance matrix of shape (N,N).
        """
        if Q.shape != (self._N,)*2:
            raise ValueError("Expected input of shape (N,)")
        self._Q = Q

    def getPredictionCov(self) -> ndarray:
        """Get the prediction covariance matrix.

        Returns:
            ndarray: prediction covariance matrix of shape (N,N).
        """
        return(self._Q)

    def setObservationModel(self, H: ndarray) -> None:
        """Set the observation model matrix.

        Args:
            H (ndarray): observation model matrix of shape (N,N).
        """
        if H.shape != (self._N,)*2:
            raise ValueError("Expected input of shape (N,)")
        self._H = H

    def getObservationModel(self) -> ndarray:
        """Get the observation model matrix.

        Args:
            ndarray: observation model matrix of shape (N,N).
        """
        return(self._H)

    def setObservationCov(self, R: ndarray) -> None:
        """Get the observation covariance matrix.

        Args:
            R (ndarray): observation covariance matrix of shape (N,N).
        """
        if R.shape != (self._N,)*2:
            raise ValueError("Expected input of shape (N,)")
        self._R = R

    def getObservationCov(self) -> ndarray:
        """Set the observation covariance matrix.

        Args:
            ndarray: observation covariance matrix of shape (N,N).
        """
        return(self._R)

    def getKalmanGain(self) -> ndarray:
        """Get the Kalman gain matrix.

        Args:
            ndarray: the Kalman gain matrix.
        """
        return(self._K)

    def step(self, z: ndarray, c: ndarray) -> None:
        """The Kalman filter step.

        Variables: F = F(t), x = x(t), B = B(t), c = c(t), P = P(t), Q = Q(t),
        xp = xp(t+1), Pp = Pp(t+1), H = H(t+1), z = z(t+1), R = R(t+1),
        K = K(t+1)

        Predict:
        xp = F @ x + B @ c
        Pp = F @ P @ F.T + Q

        Calculate Kalman gain:
        K = Pp @ H.T @ (H @ Pp @ H.T + R)^(-1)

        Update:
        x = xp + K @ (z - H @ xp)
        P = Pp - K @ H @ Pp

        Args:
            z (ndarray): measurement vector at t+1.
            c (ndarray): control vector at t.
        """

        # Do the prediction.
        xp = self._F@self._x + self._B@c
        Pp = self._F@self._P@self._F.T + self._Q

        # The Kalman gain.
        self._K: ndarray = Pp@self._H.T@np.linalg.inv(
            self._H@Pp@self._H.T + self._R
        )

        # Update the state.
        self._x = xp + self._K@(z-self._H@xp)
        self._P = Pp - self._K@self._H@Pp

def kinematic1dFilter(measurements: ndarray, noiseStd: ndarray) \
        -> Tuple[ndarray, ndarray]:
    """Filter the kinematic data using the Kalman filter and return the
    filtered data.

    Args:
        measurements (ndarray): the measurement array of dimension (N,4),
            with (N,1) being time, (N,2) displacement, (N,3) velocity and
            (N,4) acceleration. It is assumed that where the measurement is
            not available, the value is np.nan.

        noiseStd (ndarray): the measurement noise standard deviation of
            dimension 3 (for displacement, velocity and acceleration). For
            unknown data, give a large value.

    Returns:
        Tuple[ndarray, ndarray]: filtered measurements, covariance matrices.
    """
    if measurements.ndim != 2 or noiseStd.ndim != 1:
        raise ValueError("Expected a 2D measurements and 1D noise array")

    if measurements.shape[1] != 4 or noiseStd.size != 3:
        raise ValueError(
            "Expected 4 columns in measurements and 3 noiseStd elements."
        )

    if np.any(np.diff(measurements[:,0]) < 0):
        raise ValueError("Expected ascending sorted time array.")

    # Set up the Kalman Filter.
    NSTATES = 3

    # Use default initialization for most arrays.
    filter = kalmanFilter(NSTATES)

    R = np.identity(NSTATES)
    np.fill_diagonal(R, noiseStd**2)
    filter.setObservationCov(R)

    filtered = measurements.copy()
    init = measurements[0,1:]
    filtered[0,1:] = np.where(np.isnan(init), 0, init)
    covariance = np.zeros((measurements.shape[0], NSTATES, NSTATES))

    for i in range(1,measurements.shape[0]):

        # Get the time step.
        dt = measurements[i,0]-measurements[i-1,0]

        # The prediction model depends on dt.
        filter.setPredictionModel(
            np.array(
                [[1, dt, 0.5*dt*dt],
                 [0, 1,         dt],
                 [0, 0,          1]]
            )
        )

        # The prediction is noisy due to unmodeled forces (accelerations),
        # vibrations.
        Q = np.identity(NSTATES)
        Q[0,0] = 0.5*dt*dt
        Q[1,1] = dt
        filter.setPredictionCov(noiseStd[2]**2*Q)

        # Get the measurements.
        z = measurements[i,1:]
        np.fill_diagonal(R, np.where(np.isnan(z), 1e10, noiseStd**2))
        z = np.where(np.isnan(z), 0, z)

        filter.setObservationCov(R)

        filter.step(z, np.zeros(3))

        filtered[i,1:] = filter.getState()
        covariance[i,:,:] = filter.getStateCov()

    return filtered, covariance