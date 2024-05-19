import unittest
import os

import numpy as np

# This statement means from the namespace of module context import
# the myModules namespace.
from context import myModules, INPUTDIR, OUTPUTDIR

from myModules.physicalDataIO import physicalDataIO

class test_physicalDataIO(unittest.TestCase):
    testInputDir = os.path.join(INPUTDIR, "dataIO")
    testOutputDir = os.path.join(OUTPUTDIR, "dataIO")

    def test_physicalDataIORead(self):
        testInputDir = os.path.join(self.testInputDir, "physicalDataIO")

        sampleIO = physicalDataIO()
        sampleIO.read(os.path.join(testInputDir, "sample.txt"))

        self.assertEqual(sampleIO.getHeader(),
                         ["sample", "freq", "X_dft", "test"])

        self.assertEqual(sampleIO.getUnits(), ["", "Hz", "ms-1Hz-1", ""])

        data = sampleIO.getData()
        dtypes = sampleIO.getDtypes()
        for i,j in enumerate(data.dtypes):
            self.assertEqual(np.dtype(dtypes[i]).type, j.type)

        # Test different input files.
        for i in os.listdir(testInputDir):
            path = os.path.join(testInputDir, i)
            if "faulty" in i:
                if "Header" in i:
                    self.assertRaises(AssertionError, sampleIO.read, path)
                elif "Dtype" in i:
                    self.assertRaises(TypeError, sampleIO.read, path)
            else:
                sampleIO.read(path)

    def test_physicalDataIOWrite(self):
        testInputDir = os.path.join(self.testInputDir, "physicalDataIO")
        testOutputDir = os.path.join(self.testOutputDir, "physicalDataIO")

        if not os.path.exists(testOutputDir):
            os.makedirs(testOutputDir)

        sampleIO1 = physicalDataIO()
        sampleIO1.read(os.path.join(testInputDir, "sample.txt"))
        sampleIO1.setData(sampleIO1.getData(), sampleIO1.getUnits())
        sampleIO1.write(os.path.join(testOutputDir, "sample.txt"))

        sampleIO2 = physicalDataIO()
        sampleIO2.read(os.path.join(testOutputDir, "sample.txt"))

        self.assertListEqual(sampleIO1.getHeader(), sampleIO2.getHeader())
        self.assertListEqual(sampleIO1.getUnits(), sampleIO2.getUnits())
        self.assertListEqual(sampleIO1.getDtypes(), sampleIO2.getDtypes())

        df1 = sampleIO1.getData()
        df2 = sampleIO2.getData()
        self.assertEqual(df1.equals(df2), True)

if __name__=="__main__":
    unittest.main()