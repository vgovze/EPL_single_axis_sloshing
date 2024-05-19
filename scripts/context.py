import sys
import os

_PARENTPATH = os.path.join(os.path.dirname(__file__), "..")
OUTPUTPATH=os.path.abspath(os.path.join(_PARENTPATH, "output"))
CONFIGPATH=os.path.abspath(os.path.join(_PARENTPATH, "config"))

# To prevent installing the "modules" package, explicitly add its containing
# folder path to sys.path (where python can find it).
sys.path.insert(0, os.path.abspath(_PARENTPATH))

import myModules