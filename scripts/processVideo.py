import os
import multiprocessing as mp
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

from context import myModules, CONFIGPATH
from myModules.imageScaling import findPoints, findCorners
from myModules.edgeDetection import cannyEdgeDetection, autoCannyEdgeDetection
from myModules.processingHelpers import ParallelVideoProcessor, \
    getVideoIntensitySum, getVideoFpsAndNframes
from myModules.physicalDataIO import physicalDataIO
from myModules.physicalDataLabels import label
from myModules.utils import findAbsMaxDiffIdx, findChangeStartIdx

plt.style.use(os.path.join(CONFIGPATH, "report.mplstyle"))

l = label()

if __name__ == "__main__":

# PARAMETERS
###############################################################################
    # Video processing interval after motion start.
    tInt = 15

    # Grid spacing constant in mm
    gridSpacing = 2

    # Grid detection method, its arguments and optional arguments.
    gridMethod = findCorners
    gridKwargs = {"minDistance":13}

    # Input directories.
    rawDataDir = r"L:\Documents\LAB\data\raw\sloshing_2_mm_3"
    procDataDir = r"L:\Documents\LAB\data\processed\sloshing_2_mm_3"
    gridPath = r"C:\Users\Viktor.Govze\FILES\final\glass\sloshing_3_mm_1\grid.MP4"

    # Number of parallel processes to use (limited by the number of cores).
    nProc = 4

    # Edge detection parameters.
    # edgeMethod = cannyEdgeDetection
    # edgeKwargs = {"lower": 10, "upper": 30}

    edgeMethod = autoCannyEdgeDetection
    edgeKwargs = {"sigma": 0.33}

# INPUT
###############################################################################
    # Check if inputs exist.
    a = False
    if not os.path.exists(gridPath):
        a = True
        print(f"{gridPath} does not exist.")

    if not os.path.exists(rawDataDir):
        a = True
        print(f"{rawDataDir} does not exist.")

    if not os.path.exists(procDataDir):
        a = True
        print(f"{procDataDir} does not exist.")

    if a == True:
        print("Terminating program.")
        exit(0)

    # Instantiate the video processor.
    proc = ParallelVideoProcessor()

    # Instantiate the IO.
    io = physicalDataIO()

    # Find grid scaling.
    # Read the first frame of the video.
    cap = cv.VideoCapture(gridPath)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read frame from the video.")
    grid = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    proc.calculateImageScaling(
        grid, procDataDir, gridSpacing, gridMethod, kwargs=gridKwargs
    )

    while True:
        res = input(
            "Check the image scaling. Is the grid as expected? [y/n]: "
        )
        if res == "y":
            break
        elif res == "n":
            print("Terminating program.")
            exit(0)

    proc.saveImageScaling(procDataDir)

    # Make the list of all raw video paths.
    videoPaths = {}

    # The directory to store processed video files.
    procVideoDir = {}
    reVid = re.compile("f?([0-9]\.[0-9])Hz.*\.mp4", re.IGNORECASE)

    for i in os.listdir(rawDataDir):
        match = reVid.search(i)
        if match is None: continue

        freq = match.group(1)
        videoPaths[freq] = os.path.join(rawDataDir, i)

        dirPath = os.path.join(procDataDir, i[:i.rfind(".")])
        procVideoDir[freq] = dirPath
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

    # The corresponding processed data paths for each video.
    procDataPaths = {}
    reDat = re.compile("f?([0-9]\.[0-9]).*\.txt", re.IGNORECASE)

    for i in os.listdir(procDataDir):
        match = reDat.search(i)
        if match is None: continue

        procDataPaths[match.group(1)] = os.path.join(procDataDir, i)

    # Check if data has been processed.
    if videoPaths.keys() != procDataPaths.keys():
        print(f"Processed data paths are not available in {procDataDir}")
        print("Terminating program.")
        exit(0)

# PROCESSING
###############################################################################

    for freq in videoPaths:
        if float(freq) < 0.0 or float(freq) > 2.7: continue

        print(f"Processing video: {videoPaths[freq]}")

        # Prepare the argument list for parallel execution.
        fps, nFrames = getVideoFpsAndNframes(videoPaths[freq])
        framesPerProc = int(nFrames/nProc)

        arglist = [
            (videoPaths[freq], j*framesPerProc, (j+1)*framesPerProc)
            for j in range(nProc)
        ]

        intensity = mp.Pool(nProc).starmap(getVideoIntensitySum, arglist)
        intensity = np.concatenate(intensity)
        intensityNorm = intensity/np.max(intensity)

        syncIdx = findAbsMaxDiffIdx(np.gradient(np.gradient(intensityNorm)))
        tStart = (2*syncIdx+1)/(2*fps) # Take the mean time.

        print("Video synchronizes with the measurements approximately at time "
              f"{tStart:3.3f} s.")

        # Save the intensity plot and the detected sync time.
        fig, ax = plt.subplots(1,1,figsize=(10,8))
        mask = np.arange(nFrames)[syncIdx-100:syncIdx+100]
        ax.set(xlabel = "$t$ / s", ylabel = "normalized intensity")
        ax.plot((np.arange(nFrames)/fps)[mask],intensityNorm[mask])
        ax.vlines(
            tStart,
            np.min(intensityNorm[mask]),
            np.max(intensityNorm[mask]), color="r", lw=0.4,
            label="synchronization"
        )
        ax.legend()
        fig.savefig(
            os.path.join(procVideoDir[freq], f"{freq}Hz_videoSync.pdf")
        )

        # Calculate the start and end frame time to process. Process tInt
        # seconds of the video after motion starts.
        io.read(procDataPaths[freq])
        df = io.getData()
        endTime = df[l.time].iloc[-1]
        indexMotionStart = findChangeStartIdx(df[l.pos].to_numpy(), 200)
        motionStartTime = df[l.time].iloc[indexMotionStart]

        # Set the range of frames to process.
        endFrame = min(nFrames, int((tStart+motionStartTime+tInt)*fps))
        startFrame = max(0, endFrame - int(tInt*fps))

        # Save the video properties.
        with open(os.path.join(
            procVideoDir[freq], f"{freq}Hz_videoProperties.txt"
            ), "w") as f:
            f.write("Motion start time / second: "
                    f"{tStart+motionStartTime:3.3f}\n")
            f.write(f"Sync time / second: {tStart: 3.3f}\n")
            f.write("Sync frame interval [start, end] / frame: "
                    f"[{syncIdx}, {syncIdx+1}]\n")

        # Plot where the motion starts.
        ax.cla()
        ax.set(xlabel="$t$ / s", ylabel="velocity / m/s")
        mask = (df[l.time] > motionStartTime-1)\
               & (df[l.time] < motionStartTime+1)
        ax.plot(df[l.time][mask], df[l.pos][mask])
        ax.vlines(motionStartTime,
            df[l.pos][mask].min(),
            df[l.pos][mask].max(), color="r", lw=0.4,
            label="motion start"
        )
        ax.legend()
        fig.savefig(
            os.path.join(procVideoDir[freq], f"{freq}Hz_motionStart.pdf")
        )

        # Prepare the argument list for parallel execution.
        framesPerProc = int((endFrame-startFrame)/nProc)
        arglist = [
            [videoPaths[freq], procVideoDir[freq],
            startFrame + j*framesPerProc, 
            startFrame + (j+1)*framesPerProc,
            edgeMethod, (), edgeKwargs,
            tStart, 1, 3]
            for j in range(nProc)
        ]

        # Make the last process process the residual frames.
        arglist[-1][3] = endFrame

        # Parallely execute the edge detection in the video.
        mp.Pool(nProc).starmap(proc.edgeDetection, arglist)