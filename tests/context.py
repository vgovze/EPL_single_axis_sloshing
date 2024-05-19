import sys
import os

# To prevent installing the "myModules" package, explicitly add its path to
# sys.path (where python can find it).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

import myModules

# Specify I/O directories.
INPUTDIR: str  = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "input"))
OUTPUTDIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "output"))