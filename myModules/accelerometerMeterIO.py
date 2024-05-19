from itertools import islice

import pandas as pd

from myModules.physicalDataLabels import label
from myModules.physicalDataIO import physicalDataIO

# AccelorometerMeterData file manipulator.
class accelerometerMeterIO(physicalDataIO):
    """This class provides subroutines to read the output file of the 
    graph function of the Accelorometer Meter Android application by keuwlsoft,
    version 1.50.
    """

    LINE0 = "Keuwl Accelerometer Data File\n"
    LINE4 = "Time (s), X (m/s2), Y (m/s2), Z (m/s2), " \
            "R (m/s2), Theta (deg), Phi (deg)\n"

    def __init__(self) -> None:
        super(accelerometerMeterIO, self).__init__()

        l = label()
        self._header = [ \
            l.time, \
            l.combine("X", l.acc), \
            l.combine("Y", l.acc), \
            l.combine("Z", l.acc)
        ]

        self._units = ["s", "ms-2", "ms-2", "ms-2"]

    def read(self, path: str) -> None:
        """Read the accelerometerMeter file.

        Args:
            path (str): the path to the file.
        """

        # Check if the file is valid.
        with open(path, "r") as file:
            head = list(islice(file, 4))

        assert head[0] == self.LINE0, "Unrecognisable line 0."
        assert head[-1] == self.LINE4, "Unrecognisable line 4."

        df = pd.read_csv(
            path, sep=",", header=3, skip_blank_lines=False
        )
        df = df.drop(df.columns[4:], axis=1)

        df.columns = self._header
        self._dtypes = [i.name for i in df.dtypes]
        self._df = df

        self._assertConformity()