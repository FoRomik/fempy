import os
import sys
from distutils.spawn import find_executable as which

SQA = False
DEBUG = False
VERBOSITY = 1
ENABLE_WEAVE = None

def set_runopt(name, value):
    if name not in globals():
        raise SystemExit("Attempting to set a global attribute '{0}' "
                         "that did not previously exist".format(name))
    setattr(sys.modules[__name__], name, value)

def has_c_compiler(compiler="gcc"):
    if os.path.exists(compiler):
        path = compiler
    else:
        path = which(compiler)
    set_runopt("ENABLE_WEAVE", bool(path))
    return compiler
ENABLE_WEAVE = has_c_compiler()
