import os
import sys
from numpy.distutils.misc_util import get_shared_lib_extension as np_so_ext

__version__ = (1, 0, 0)
ROOT = os.path.dirname(os.path.realpath(__file__))

PYEXE = os.path.realpath(sys.executable)
CORE = os.path.join(ROOT, "core")
UTIL = os.path.join(ROOT, "utilities")
TOOL = os.path.join(ROOT, "toolset")
TEST = os.path.join(ROOT, "tests")
EXO = os.path.join(ROOT, "io/exodusii")

SO_EXT = np_so_ext()

# environment variables
PATH = os.getenv("PATH").split(os.pathsep)
if TOOL not in PATH:
    PATH.insert(0, TOOL)

# Add cwd to sys.path
sys.path.insert(0, os.getcwd())

# Environment to use when running subprocess.Popen or subprocess.call
ENV = dict(os.environ)
pypath = ENV.get("PYTHONPATH", "").split(os.pathsep)
pypath.extend([ROOT, EXO])
ENV["PYTHONPATH"] = os.pathsep.join(p for p in pypath if p.split())
ENV["PATH"] = os.pathsep.join(PATH)


SPLASH = """\
                  M           M    M           M    L
                 M M       M M    M M       M M    L
                M   M   M   M    M   M   M   M    L
               M     M     M    M     M     M    L
              M           M    M           M    L
             M           M    M           M    L
            M           M    M           M    L
           M           M    M           M    LLLLLLLLL
                     Material Model Laboratory v {0}

""".format(".".join("{0}".format(i) for i in __version__))


def check_prereqs():
    errors = []
    platform = sys.platform
    (major, minor, micro, relev, ser) = sys.version_info
    if (major != 3 and major != 2) or (major == 2 and minor < 7):
        errors.append("python >= 2.7 required")
        errors.append("  {0} provides {1}.{2}.{3}".format(
                sys.executable, major, minor, micro))

    # --- numpy
    try: import numpy
    except ImportError: errors.append("numpy not found")

    # --- scipy
    try: import scipy
    except ImportError: errors.append("scipy not found")
    return errors

