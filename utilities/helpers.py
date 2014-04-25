import os
import sys
import inspect
import numpy as np

def whoami():
    """ return name of calling function """
    return inspect.stack()[1][3]

def who_is_calling():
    """return the name of the calling function"""
    stack = inspect.stack()[2]
    return "{0}.{1}".format(
        os.path.splitext(os.path.basename(stack[1]))[0], stack[3])

def print_array(a, stream=sys.stdout):
    """Print an array line by line to stream"""
    numbers = (float, int, np.float, np.float64, np.int, np.int32)
    for row in a:
        if isinstance(row, numbers):
            stream.write("{0: .2f}\n".format(float(row)))
            continue

        if isinstance(row[0], numbers):
            stream.write(" ".join("{0: .2f}".format(float(x)) for x in row) + "\n")
        else:
            stream.write(" ".join("{0}".format(x) for x in row) + "\n")

def clean_wasatch(runid, cleanall=False):
    exts = [".ech", ".out"]
    if cleanall: exts.append(".exo")
    for ext in exts:
        try: os.remove(runid + ext)
        except OSError: pass
