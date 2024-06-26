import pandas as pd
import numpy as np

from myModules.physicalDataIO import physicalDataIO
from myModules.physicalDataLabels import label
from myModules.constants import standardG

class stevalMKI209V1K_IO(physicalDataIO):
    """This class provides methods to read the output file of the mems
    based accelerometer evaluation board from STElectronics. The output is
    generated by the STElectronics Unico-gui.
    """

    LINE0 = "STEVAL-MKI209V1K (IIS2ICLX)\n"
    LINE1 = "\n"
    mg = 0.001*standardG # milli g's to ms-2

    def __init__(self, fs: int) -> None:
        """The file output does not provide time, so sampling frequency
        must be given.

        Args:
            fs (int): sampling frequency in Hz.
        """
        super(stevalMKI209V1K_IO, self).__init__()

        # Set the sampling frequency of the accelerometer.
        self._fs: int = fs

        l = label()
        self._header = [\
            l.time,
            l.combine("X", l.acc),
            l.combine("Y", l.acc),
            l.temp
        ]

        self._units = ["s", "ms-2", "ms-2", "degC"]

    def read(self, path: str) -> None:
        """Read the stevalMKI209V1K output.

        Args:
            path (str): path to the data file.
        """

        # Check if the file is valid.
        with open(path, "r") as file:
            assert file.readline() == self.LINE0, "Unrecognizable line 0."
            assert file.readline() == self.LINE1, "Unrecognizable line 1."
            head = [i.strip() for i in file.readline().strip().split("\t")]

        df = pd.read_csv(
            path, sep="\s+", skiprows=[0,1,2], skip_blank_lines=False,
        )

        df.columns = head

        # Extract acceleration specific columns.
        df = df[["A_X [mg]", "A_Y [mg]", "TEMP [C]"]]
        df.columns = self._header[1:]

        # Turn mg to ms-2.
        for i in self._header[1:3]:
            df[i] = df[i]*self.mg

        # Create the time array.
        df.insert(
            0, self._header[0],
            np.arange(0, (df.shape[0]-0.5)/self._fs, 1/self._fs)
        )

        self._df = df