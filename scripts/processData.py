import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from context import myModules, CONFIGPATH
from myModules.physicalDataIO import physicalDataIO
from myModules.physicalDataLabels import label
from myModules.kalmanFilter import kinematic1dFilter

from myModules.processingHelpers import preprocessAccelerometerData, \
    preprocessDisplacementData, mergeTimeDataFrames

plt.style.use(os.path.join(CONFIGPATH, "report.mplstyle"))


l = label()

if __name__ == "__main__":

# PARAMETERS
###############################################################################
    # Displacement transducer calibration constant / m/V
    calDT: float = 0.0228274

    # Accelerometer calibration angle / degrees
    # The angle with which to rotate the acceleration data to align with the
    # direction of movement.
    calAngle: float = (180-89.713)*np.pi/180

    # The time interval after the sync event to get noise statistics.
    statsDt = [0.2, 10]

# INPUT
###############################################################################
    rawDataDir = r"C:\Users\Viktor.Govze\FILES\final\glass\sloshing_3_mm_2"
    procDataDir = r"C:\Users\Viktor.Govze\FILES\results\sloshing_3_mm_2"

    if not os.path.exists(procDataDir):
        os.makedirs(procDataDir)

    # Make the list of all data paths.
    dataPaths = []

    # Find all possible frequencies of the measurement.
    freq = list(set(
        [j.split("f")[1].split("H")[0] for j in os.listdir(rawDataDir) \
         if ("f" in j) and (".txt" in j)]
    ))

    # Make all frequency paths.
    for f in freq:
        displ = f"f{f}Hz_displ.txt"
        acc = f"f{f}Hz_acc.txt"
        dataPaths.append(
            [os.path.join(rawDataDir, displ),
             os.path.join(rawDataDir, acc)]
        )

# PROCESSING
###############################################################################
    for i, pathList in enumerate(dataPaths):
        print("Processing frequency: ", freq[i])

        displData, accData = pathList

        dfAcc, noiseAcc = preprocessAccelerometerData(
            accData, fs=1/0.0012, calAngle=calAngle, steadyTime=statsDt
        )
        maskAcc = (dfAcc[l.time] < 0.05) & (dfAcc[l.time] > -0.05)

        dfDispl, noiseDispl = preprocessDisplacementData(
            displData, calConstant=calDT, steadyTime=statsDt
        )
        maskDispl = (dfDispl[l.time] < 0.05) & (dfDispl[l.time] > -0.05)

        # Save the figure of the sync event in the dataset.
        fig, axs = plt.subplots(1, 2, figsize=(15,8))
        ax1, ax2 = axs
        ax1.set(xlabel="$t$ / s", ylabel="position / m")
        ax2.set(xlabel="$t$ / s", ylabel="acceleration / m/s")
        ax1.plot(dfDispl[l.time][maskDispl], dfDispl[l.pos][maskDispl])
        ax1.vlines(
            0, dfDispl[l.pos][maskDispl].min(),
            dfDispl[l.pos][maskDispl].max(), color="r", lw=0.2,
            label="synchronization"
        )
        ax1.legend()
        ax2.plot(dfAcc[l.time][maskAcc], dfAcc["X.acc"][maskAcc])
        ax2.vlines(
            0, dfAcc["X.acc"][maskAcc].min(), dfAcc["X.acc"][maskAcc].max(),
            color="r", lw=0.2, label="synchronization"
        )
        ax2.legend()
        fig.savefig(
            os.path.join(procDataDir, f"{freq[i]}Hz_kinematicSync.pdf")
        )

        # Get the sampling time intervals.
        dtDispl = dfDispl[l.time].iloc[1]-dfDispl[l.time].iloc[0]
        dtAcc = dfAcc[l.time].iloc[1]-dfAcc[l.time].iloc[0]

        # Drop data outside common time.
        tMin, tMax = statsDt[0],\
            min(dfDispl[l.time].iloc[-1], dfAcc[l.time].iloc[-1])

        mask = (dfDispl[l.time] > tMin) & (dfDispl[l.time] < tMax)
        dfDispl = dfDispl[mask].reset_index(drop=True)

        mask = (dfAcc[l.time] > tMin) & (dfAcc[l.time] < tMax)
        dfAcc = dfAcc[mask].reset_index(drop=True)

        merged = mergeTimeDataFrames([dfDispl, dfAcc])
        merged.rename(columns={"X.acc":l.acc}, inplace=True)
        merged[l.vel] = np.nan*merged[l.acc]

        filtered, covariance = kinematic1dFilter(
            merged[[l.time, l.pos, l.vel, l.acc]].to_numpy(),
            np.array([noiseDispl, 1e10, noiseAcc[0]])
        )
        units = ["s", "m", "ms-1", "ms-2"]

        # Interpolate data to even sampling intervals.
        time = dfDispl[l.time].to_numpy()
        if dtAcc < dtDispl:
            time = dfAcc[l.time].to_numpy()

        processed = pd.DataFrame(time, columns=[l.time])
        for k,j in enumerate([l.pos, l.vel, l.acc], start=1):
            processed[j] = np.interp(time, filtered[:,0], filtered[:,k])

# OUTPUT
###############################################################################

        # Write the data to a file.
        print("Writing data to a file.")
        dataIO = physicalDataIO()
        dataIO.setData(processed, units)
        dataIO.write(os.path.join(procDataDir, f"{freq[i]}Hz.txt"))

        # Plot the last <x> seconds of the processed data.
        shift = 4
        print("Plotting the results.")
        mask = (dfDispl[l.time] > tMax-shift) & (dfDispl[l.time] < tMax)
        dfDispl = dfDispl[mask].reset_index(drop=True)

        mask = (dfAcc[l.time] > tMax-shift) & (dfAcc[l.time] < tMax)
        dfAcc = dfAcc[mask].reset_index(drop=True)

        mask = (processed[l.time] > tMax-shift) & (processed[l.time] < tMax)
        processed = processed[mask].reset_index(drop=True)

        fig, axs = plt.subplots(1,3, figsize=(15,6))
        pos, vel, acc = axs

        pos.set(xlabel = "time / s", ylabel = "position / m")
        vel.set(xlabel = "time / s", ylabel = "velocity / ms-1")
        acc.set(xlabel = "time / s", ylabel = "acceleration / ms-2")

        acc.plot(dfAcc[l.time], dfAcc["X.acc"], "k", label="measured", lw=0.5)
        acc.plot(processed[l.time], processed[l.acc], label="filtered", lw=0.5)
        acc.plot(
            processed[l.time].iloc[:-1],
            np.diff(processed[l.vel])/np.diff(processed[l.time]),
            label="acc_from_vel", alpha=0.7, lw=0.2
        )
        acc.legend()

        vel.plot(processed[l.time], processed[l.vel], label="filtered", lw=0.5)
        vel.plot(
            processed[l.time].iloc[:-1],
            np.diff(processed[l.pos])/np.diff(processed[l.time]),
            label="vel_from_displ", alpha=0.7, lw=0.2)
        vel.legend()

        pos.plot(
            dfDispl[l.time], dfDispl[l.pos], "k", label="measured", lw=0.5
        )
        pos.plot(processed[l.time], processed[l.pos], label="filtered", lw=0.5)
        pos.legend()

        path = os.path.join(procDataDir, f"{freq[i]}Hz.pdf")
        fig.savefig(path)
        plt.close("all")