import os
from typing import Tuple, Callable
import multiprocessing as mp

import numpy as np
from numpy import ndarray
import pandas as pd
import cv2 as cv
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from myModules.physicalDataIO import physicalDataIO
from myModules.stevalMKI209V1K_IO import stevalMKI209V1K_IO
from myModules.physicalDataLabels import label
from myModules.utils import findAbsMaxDiffIdx
import myModules.imageScaling as imgs
from myModules.edgeDetection import contourDetection

###############################################################################
# DISCLAIMER:
# This module is very specific to the setup we have in the lab as of 28.6.2023.

# This data processing methodology is described in the signalProcessing.ipynb,
# although there may be some small changes.
###############################################################################

l = label()

# Preprocess the data from the displacement transducer.
def preprocessDisplacementData(path: str, calConstant: float, \
        steadyTime: list[float]) -> Tuple[pd.DataFrame, float]:
    """Process the data from the displacement transducer gathered using the
    LabVIEW program with a constant sampling frequency. Zero the measurements
    and calculate steady state std.

    Args:
        path (str): path to the LabVIEW raw data.
        calConstant (str): the displacement transducer calibration constant.
        steadyTime (list[float]): the time interval after synchronization
            in which to calculate the noise of the reading.

    Returns:
        pd.DataFrame: DataFrame containing time and zeroed position based
            on the steadyTime interval.
        float: The standard deviation of steady state position.
    """
    global l

    displReader = physicalDataIO()
    displReader.read(path)
    dataDispl = displReader.getData()

    # Convert the units to avoid hassle in processing.
    dataDispl[l.pos] = dataDispl["voltage"]*calConstant

    # Find the synchronization event.
    tSyncIdx = findAbsMaxDiffIdx(np.gradient(np.gradient(dataDispl[l.pos])))

    # Synchronize the time of the data.
    tStart = (dataDispl[l.time][tSyncIdx+1]+dataDispl[l.time][tSyncIdx])/2
    dataDispl[l.time] -= tStart

    # Get the measurement statistics.
    mask = (dataDispl[l.time] > steadyTime[0])\
         & (dataDispl[l.time] < steadyTime[1])
    mean = dataDispl[l.pos].loc[mask].mean()
    noise = dataDispl[l.pos].loc[mask].std()

    # Zero the measurements.
    dataDispl[l.pos] -= mean

    return dataDispl[[l.time, l.pos]].copy(), noise

def preprocessAccelerometerData(path: str, fs: float, calAngle: float,
        steadyTime: list[float]) -> Tuple[pd.DataFrame, ndarray]:
    """Process the data from the accelerometer with the given constant sampling
    frequency. Calculate only the projection in the direction of movement.
    Zero the measurements and calculate steady state std.

    Args:
        path (str): path to the accelerometer data.
        fs (float): sampling frequency of the accelerometer.
        calAngle (float): the angle with which to rotate the data to
            align with the accelerometer +x axis. In radians!
        steadyTime (list[float]): the time interval after synchronization
            in which to calculate the noise of the reading.

    Returns:
        pd.DataFrame: DataFrame containing time and accelerations, with
            column X.acc being aligned with the direction of movement. The
            accelerations are zeroed based on the steadyTime interval.
        ndarray: the standard deviation of the acceleration readings for
            X.acc and Y.acc (after alignment) in steady state time.
    """
    global l

    # Get the raw accelerometer data.
    accReader = stevalMKI209V1K_IO(fs)
    accReader.read(path)
    dataAccRaw = accReader.getData()
    dataAcc = dataAccRaw.copy()

    # Align the acceleration in the direction of movement to the x axis.
    dataAcc["X.acc"] = dataAccRaw["X.acc"]*np.cos(calAngle)\
                     - dataAccRaw["Y.acc"]*np.sin(calAngle)
    dataAcc["Y.acc"] = dataAccRaw["X.acc"]*np.sin(calAngle)\
                     + dataAccRaw["Y.acc"]*np.cos(calAngle)

    # Find the synchronization index.
    tSyncIdx = min(
        findAbsMaxDiffIdx(dataAcc["X.acc"].to_numpy()),
        findAbsMaxDiffIdx(dataAcc["Y.acc"].to_numpy())
    )

    # Synchronize the time of the data. Take the mean.
    tStart = (dataAcc[l.time][tSyncIdx+1]+dataAcc[l.time][tSyncIdx])/2
    dataAcc[l.time] -= tStart

    # Zero the measurements.
    mask = (dataAcc[l.time] > steadyTime[0]) \
         & (dataAcc[l.time] < steadyTime[1])
    mean = dataAcc[["X.acc", "Y.acc"]].loc[mask].mean()
    dataAcc[["X.acc", "Y.acc"]] -= mean

    # Calculate the noise.
    noiseX = dataAcc["X.acc"][mask].std()
    noiseY = dataAcc["Y.acc"][mask].std()

    return dataAcc, np.array((noiseX, noiseY))

def mergeTimeDataFrames(dataFrameList: list[pd.DataFrame]) \
        -> pd.DataFrame:
    """Merge the dataframes in the dataFrameList with respect to time and
    combine the data at the same times.

    Args:
        dataFrameList (list[pd.DataFrame]): list of dataframes with a
        time column.

    Returns:
        pd.DataFrame: combined dataframe without duplicate times.
    """
    global l

    # Merge dataframes.
    data = pd.concat(
        dataFrameList, ignore_index=True
    ).sort_values(l.time).reset_index(drop=True)

    # Get the duplicate times.
    mask = data[l.time].diff() < 1e-10
    duplicates = data[mask]
    data = data[~mask]

    # Combine data on these times.
    duplicates.index -= 1 # First shift index to match.
    data = data.combine_first(duplicates).reset_index(drop=True)

    return data

def getVideoFpsAndNframes(path: str) -> Tuple[int, int]:
    """Return the video framerate and the number of frames.

    Args:
        path (str): path to the video.

    Returns:
        Tuple[int, int]: the fps and number of frames.
    """
    cap = cv.VideoCapture(path)
    fps = cap.get(cv.CAP_PROP_FPS)
    nFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    cap.release()

    return int(fps), int(nFrames)

def getVideoIntensitySum(path: str, iStart: int, iEnd: int) -> ndarray:
    """For frame numbers iStart <= i < iEnd calculate the sum the
    values of the pixels and return an array of values for every frame.

    Args:
        path (str): video path.
        iStart (int): starting frame.
        iEnd (int): ending frame.

    Returns:
        ndarray: sum of pixel values for every frame in the given range.
    """
    cap = cv.VideoCapture(path)
    cap.set(cv.CAP_PROP_POS_FRAMES, iStart)

    intensity = []
    i = iStart
    while True:
        if i >= iEnd: break
        ret, frame = cap.read()
        if not ret: break
        intensity.append(np.sum(frame))
        i += 1
    cap.release()

    return np.array(intensity)

class ParallelVideoProcessor():
    """The parallel video processor class providing image processing methods
    that can be run embarassingly parallel."""
    SCALING_FILE = "video_scalingMap.txt"
    SCALING_IMAGE = "video_scalingImg.pdf"

    def __init__(self) -> None:
        self.imgPts = None
        self.gridPts = None
        self.X = None
        self.Y = None

    def _checkIfScalingPresent(self) -> None:
        """Sanity check."""
        if type(self.X) == type(None):
            print(
                "Image scaling not computed. Run calculateImageScaling first!"
            )
        return None

    def calculateImageScaling(self, grayImg: ndarray, saveDir: str,
            gridSpacing: float, method: Callable[..., ndarray],
            args: tuple=(), kwargs: dict={}) -> None:
        """Calculate the scaling coefficients for the video.

        Args:
            grayImg (ndarray): grayscale image of the grid.
            gridSpacing (float): the physical grid point spacing in mm.
            method (Callable): the function to use to determine grid point
                locations in image coordinates. Must accept grayscale 2D
                matrix as the first argument.
            args (tuple, optional): positional arguments to the method
                function. Defaults to 0 length tuple.
            kwargs (dict, optional): keyword arguments to the method function.
                Defaults to 0 length dictionary.
        """

        # Detect the points.
        uv = method(grayImg, *args, **kwargs)

        # Reconstruct the grid.
        xy, uvAct = imgs.mapSquare2dGrid(uv, gridSpacing)
        self.imgPts = uvAct
        self.gridPts = xy

        # Save the grid (normalized).
        shiftXY = np.array((np.min(xy[:,0]), np.min(xy[:,1])))
        xy = xy - shiftXY
        xyNorm = xy/np.max(xy[:,1])

        shiftUV = np.array((np.min(uvAct[:,0]), np.min(uvAct[:,1])))
        uvNorm = uvAct - shiftUV
        uvNorm = uvNorm/np.max(uvNorm[:,1])

        xyNorm = xyNorm
        fig, ax = plt.subplots(1,1,figsize=(10,8))
        ax.set(xlabel=r"$\tilde{x}$", ylabel=r"$\tilde{y}$")
        ax.axis("equal")
        ax.scatter(
            xyNorm[:,0], xyNorm[:,1], s = 0.1, color="k", label="xyGrid"
        )
        ax.scatter(
            uvNorm[:,0], uvNorm[:,1], s = 0.1, color="r", label="imageGrid"
        )
        ax.legend()

        savePath = os.path.join(saveDir, self.SCALING_IMAGE)
        print(f"Saving the obtained grid to: \"{savePath}\"")
        fig.savefig(savePath)

        # Interpolate the x-y coordinates.
        U, V = np.mgrid[0:grayImg.shape[1], 0:grayImg.shape[0]]

        # Flip the y direction of the xy to get the proper video
        # frame coordinate correspondence.
        xy[:,1] *= -1
        res = griddata(uvAct, xy, (U,V), method="linear")

        # Set the pixel to coordinate map for x and y.
        self.X, self.Y = res[:,:,0], res[:,:,1]

    def saveImageScaling(self, saveDir: str) -> None:
        """Save the image scaling map (u,v) -> (x,y) in a .txt file.

        Args:
            saveDir (str): the save directory.
        """

        self._checkIfScalingPresent()

        data = pd.DataFrame(
            {"U": self.imgPts[:,0], "V": self.imgPts[:,1],
            "X": self.gridPts[:,0], "Y": self.gridPts[:,1]}
        )

        io = physicalDataIO()
        io.setData(data, ["px"]*2+["mm"]*2)

        saveFile = os.path.join(saveDir, self.SCALING_FILE)
        print(f"Saving the scaling to \"{saveFile}\"")
        io.write(saveFile)

    def edgeDetection(self, vidPath: str, saveDir: str,
            iStart: int, iEnd: int, method: Callable[..., ndarray],
            args=(), kwargs={}, tStart: float=0.0,
            saveTxtStep: int=1, saveFigStep: int=1) -> None:
        """Detect the edges using the Canny edge detection and save them
        to a file.

        Args:
            vidPath (str):  path to the video.
            saveDir (str):  path to the save directory.
            iStart (int):   starting frame.
            iEnd (int):     ending frame (not inclusive).
            method (Callable): the function to use to calculate the edges.
                Must accept the image ndarray as the first argument.
            args (tuple, optional): positional arguments to the method
                function. Defaults to 0 length tuple.
            kwargs (dict, optional): keyword arguments to the method function.
                Defaults to 0 length dictionary.
            tStart (float): the start time of the synchronized video.
            saveTxtStep (int, optional): store every i-th frame detected edge
                xy coordinates. Defaults to 1.
            saveFigStep (int, optional): store every i-th frame image of
                detected edge x-y coordinates and grayscale image with
                highlighted edges. Defaults to 1.
        """

        pid = os.getpid()
        print(f"Process {pid} started work.")
        self._checkIfScalingPresent()

        io = physicalDataIO()
        cap = cv.VideoCapture(vidPath)
        cap.set(cv.CAP_PROP_POS_FRAMES, iStart)
        fps = cap.get(cv.CAP_PROP_FPS)

        # Prepare the figure.
        fig, ax = plt.subplots(1,1,figsize=(16,9))
        fig2, ax2 = plt.subplots(1,1,figsize=(10,8))

        i = iStart-1
        while i < iEnd:
            i += 1

            print(
                f"Current progress of process {pid}: "
                f"{100*(i-iStart)/(iEnd-iStart):3.2f} %"
            )

            if i%saveTxtStep != 0 and i%saveFigStep != 0: continue

            ret, frame = cap.read()
            if not ret: break

            # Binarize the image with edges.
            edges = method(frame, *args, **kwargs)

            # Extract detected edge x and y coordinates.
            contourPts = contourDetection(edges)

            I, J = contourPts[:,0], contourPts[:,1]
            xs, ys = self.X[I, J], self.Y[I, J]

            # Calculate time.
            time = i/fps - tStart # Calculate time.

            # Write data.
            if i%saveTxtStep == 0:
                io.setData(pd.DataFrame({"x":xs, "y":ys}), units=["mm", "mm"])
                io.write(os.path.join(saveDir, f"{time:3.3f}s_xy.txt"))

            # Store image.
            if i%saveFigStep == 0:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                ax.cla()
                ax.set(
                    xlabel="u / px", ylabel="v / px", title=f"{time:3.3f} s"
                )
                ax.axis('equal')
                ax.imshow(edges, cmap="viridis", interpolation="none")
                mask = np.ma.masked_where(edges > 0, gray)
                ax.imshow(mask, cmap="gray", interpolation="none")
                fig.savefig(os.path.join(saveDir, f"{time:3.3f}s_img.png"))

                # Store edge position.
                ax2.cla()
                ax2.set(
                    xlabel="x / mm", ylabel="y / mm", title=f"{time:3.3f} s"
                )
                ax2.axis("equal")
                ax2.scatter(xs, ys, s=0.1, color="k")
                fig2.savefig(os.path.join(saveDir, f"{time:3.3f}s_xy.png"))

        plt.close("all")
        print(f"Process {pid} done.")