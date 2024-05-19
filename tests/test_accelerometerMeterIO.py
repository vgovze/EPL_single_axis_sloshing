import unittest
import os

import numpy as np

# This statement means from the namespace of module context import
# the myModules namespace.
from context import myModules, INPUTDIR, OUTPUTDIR

from myModules.accelerometerMeterIO import accelerometerMeterIO

class test_accelerometerMeterIO(unittest.TestCase):
    testInputDir = os.path.join(INPUTDIR, "dataIO")
    testOutputDir = os.path.join(OUTPUTDIR, "dataIO")

    def test_accelorometerMeterIO(self):
        testInputDir = os.path.join(
            self.testInputDir,
            "accelerometerMeterIO"
        )

        sampleIO = accelerometerMeterIO()

        for i in os.listdir(testInputDir):
            path = os.path.join(testInputDir, i)
            if "badInput" in i:
                with self.assertRaises(AssertionError):
                    sampleIO.read(path)
            else:
                sampleIO.read(path)
                df = sampleIO.getData()

                self.assertEqual(4, len(df.columns))
                self.assertEqual(4, df.shape[1])

if __name__=="__main__":
    unittest.main()