import sys
import time
import numpy as np
import scipy.linalg.flapack as flapack

from runopts import DEBUG

def linsolve(a, b):
    """Interface to the lapack dposv solve function in scipy.linalg

    Parameters
    ----------
    a : ndarray
        Real, symmetric, positive-definite matrix (the stiffness matrix)
    b : ndarray
        RHS of system of equations

    Returns
    -------
    c : ndarray
    x : ndarray
        Solution to A x = b
    info : int
        info < 0 -> bad input
        info = 0 -> solution is in x
        ifno > 0 -> singular matrix

    Notes
    -----
    dposv solves the system of equations A x = b using lapack's dposv
    procedure. This interface function is used to avoid the overhead of
    calling down in to scipy, converting arrays to fortran order, etc.

    """
    if DEBUG:
        ti = time.time()
        sys.stdout.write('*** entering: fem.core.lento.linsolve\n')

    c, x, info = flapack.dposv(a, b, lower=0, overwrite_a=0, overwrite_b=0)

    if DEBUG:
        sys.stdout.write('***  exiting: fem.core.lento.linsolve (%.2f)s\n'
                         % (time.time() - ti) )

    return c, x, info
