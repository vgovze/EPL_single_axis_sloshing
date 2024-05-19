import pandas as pd
import numpy as np

# Standard signal output file.
# line 0 - header
# line 1 - units
# line 2 - data types
# line 3 and onwards - actual data

class physicalDataIO():
    # The estabilished file separator.
    SEP = ","

    def __init__(self) -> None:
        self._header: list[str] = []
        self._units: list[str] = []
        self._dtypes: list[np.dtype] = []
        self._df: pd.DataFrame = pd.DataFrame()

    def getHeader(self) -> list[str]:
        """Return the header of the data.

        Returns:
            list[str]: names of the header entries.
        """
        return self._header

    def getUnits(self) -> list[str]:
        """Return the units of the data.

        Returns:
            list[str]: units of the data.
        """
        return self._units

    def getDtypes(self) -> list[str]:
        """Return the datatypes of the data.

        Returns:
            list[str]: datatypes of the data.
        """
        return self._dtypes

    def getData(self) -> pd.DataFrame:
        """Return the dataframe containing the data.

        Returns:
            pd.DataFrame: the pandas dataframe.
        """
        # To prevent modifying the classes dataframe, return a copy.
        return self._df.copy()

    def setData(self, df: pd.DataFrame, units: list[str]) -> None:
        """Set the dataframe containing data. The header and dtypes are
        automatically inferred.

        Args:
            df (pd.DataFrame): the dataframe to store.
            units (list[str]): units of the corresponding columns.
        """
        # Store a copy of the dataframe.
        self._df = df.copy()

        # Infer the header and the dtypes from the dataframe.
        self._header = list(self._df.columns)
        self._units = units
        self._dtypes = [i.name for i in self._df.dtypes]

    def read(self, path: str) -> None:
        """Read the physicalData file.

        Args:
            path (str): path to the data file.

        Raises:
            TypeError: in case if the given data types do not match.
        """
        split: list[str] = lambda x : x.split("\n")[0].split(self.SEP)
        strip: list[str] = lambda x: [j.strip() for j in x]

        with open(path, "r") as file:
            self._header = strip(split(file.readline()))
            self._units  = strip(split(file.readline()))
            self._dtypes = strip(split(file.readline()))

        self._df = pd.read_csv(path, sep=self.SEP, skiprows=[1,2])
        self._df.columns = self._header

        self._assertConformity()

        # Cast the columns to proper data types.
        for i,j in enumerate(self._header):
            try:
                self._df[j] = np.array(self._df[j], dtype=self._dtypes[i])
            except:
                s: str = f"Data of column \"{j}\" can not be cast " \
                    f"into array of dtype \"{self._dtypes[i]}\""
                raise TypeError(s)

        # Update the actual dtypes.
        self._dtypes = [i.name for i in self._df.dtypes]

    def write(self, path: str) -> None:
        """Write the physicalData file.

        Args:
            path (str): path to the data file.
        """
        self._assertConformity()

        with open(path, "w") as file:
            file.write(self.SEP.join(self._header)+"\n")
            file.write(self.SEP.join(self._units)+"\n")
            file.write(self.SEP.join(self._dtypes)+"\n")

        self._df.to_csv(
            path, sep=self.SEP, mode="a", header=False, index=False
        )

    def _assertConformity(self):
        """Assert the conformity of the data.
        """
        # Assert that header, unit and dtype are of the same length as data.
        nCols: int = self._df.shape[1]
        assert nCols == len(self._header), \
            "Expected a name for each column"
        assert nCols == len(self._units), \
            "Expected a unit for each column."
        assert nCols == len(self._dtypes), \
            "Expected a data type for each column."

        # Assert headers are unique.
        assert len(self._header) == len(set(self._header)), \
            "Expected unique header names for every column."