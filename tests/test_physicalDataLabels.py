import unittest

from context import myModules

from myModules.physicalDataLabels import label

class test_physicalDataIO(unittest.TestCase):
    def test_label(self):
        l = label()

        name = l.acc + l.SEP + l.ampl + l.SEP + l.freq
        testName = l.combine(l.acc,l.ampl,l.freq)
        expectedResult = l.acc+l.SEP+l.ampl

        self.assertEqual(l.extract(name, 1), expectedResult)
        self.assertEqual(testName, name)

if __name__ == "__main__":
    unittest.main()