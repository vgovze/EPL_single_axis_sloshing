import os
import re

import numpy as np

from context import myModules
from myModules.utils import parametric2dLineFit
from myModules.stevalMKI209V1K_IO import stevalMKI209V1K_IO as acc_io

inputDir = r"L:\Documents\LAB\data\raw\accelerometerAngleCalibration"
outputDir = r"L:\Documents\LAB\data\processed\accelerometerAngleCalibration"

regXp = re.compile("[0-9]+.txt")
io = acc_io(1/0.0012)

f = open(os.path.join(outputDir, "calibrationReport.txt"), "w")
f.write("file: angle / degrees\n")

angles = []
for file in os.listdir(inputDir):
    if regXp.match(file) is None: continue
    print(f"Processing file \"{file}\"")

    io.read(os.path.join(inputDir, file))
    data = io.getData()

    k1, k2, *c = parametric2dLineFit(
        data["X.acc"].to_numpy(),
        data["Y.acc"].to_numpy()
    )

    angle = np.arctan2(k2, k1)*180/np.pi
    angles.append(angle)
    f.write(f"{file}: {angle:7.6f}\n")

f.write("\n")
f.write("Statistics\n")
angles = np.array(angles)
f.write(f"mean: {np.mean(angles):7.6f}\n")
f.write(f"std: {np.std(angles):7.6f}")
f.close()