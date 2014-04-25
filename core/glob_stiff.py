import sys
import numpy as np
from scipy.weave import inline, converters

from runopts import ENABLE_WEAVE


def global_stiffness(t, dt, X, elements, connect, du, nproc):
    """Assemble the global stiffness matrix

    Parameters
    ----------
    t : float
        Current simulation time

    dt : float
        Time step

    X : array like, (i, j,)
        Nodal coordinates
        X[i, j] -> jth coordinate of ith node for i=1...nnode, j=1...ncoord

    elements : array_like, (i,)
        Array of element classes

    connect : array_like, (i, j,)
        Nodal connections
        connect[i, j] -> jth node on the ith element

    du : array_like, (i, 1)
        Nodal displacement increment.

    Returns
    -------
    stif : array_like

    """
    maxnodes = np.amax([x.nnodes for x in elements])
    sdim = elements[0].ndof * X.shape[0]
    stif = np.zeros((sdim, sdim))

    # Loop over all the elements, compute element stiffness and add to global
    for (lmn, element) in enumerate(elements):
        kel = elstif(lmn, element, dt, maxnodes, connect[lmn], X, du)

        # Add the current element stiffness to the global stiffness
        nnodes = elements[lmn].nnodes
        ncoord = elements[lmn].ncoord
        ndof = elements[lmn].ndof

        if not ENABLE_WEAVE:
            for a in range(nnodes):
                for i in range(ndof):
                    for b in range(nnodes):
                        for k in range(ndof):
                            rw = ndof * connect[lmn, a] + i
                            cl = ndof * connect[lmn, b] + k
                            elrw = ndof * a + i
                            elcl = ndof * b + k
                            stif[rw, cl] += kel[elrw, elcl]

        else:
            code = """
                int rw, cl, elrw, elcl;
                for (int a=0; a < nnodes; ++a) {
                  for (int i=0; i < ndof; ++i) {
                    for (int b=0; b < nnodes; ++b) {
                      for (int k=0; k < ndof; ++k) {
                        rw = ndof * connect(lmn, a) + i;
                        cl = ndof * connect(lmn, b) + k;
                        elrw = ndof * a + i;
                        elcl = ndof * b + k;
                        stif(rw, cl) += kel(elrw, elcl);
                        }
                      }
                    }
                  }
                """
            inline(code, ["nnodes", "ndof", "lmn", "connect", "stif", "kel"],
                   type_converters=converters.blitz)

        continue

    return stif


def elstif(lmn, e, dt, maxnodes, conn, X, du):
    """Helper function for computing an elements stiffness

    Parameters
    ----------
    args : tuple
        Contains: lmn, e, dt, maxnodes, conn, X, du

    Returns
    -------
    lmn : int
        Element number

    k : ndarray
        Element stiffness

    Notes
    -----
    Helper function for compting element stiffnesses on multiple processors

    """
    # X_e will hold the original coordinates for nodes on element
    X_e = np.zeros((maxnodes, e.ncoord))
    du_e = np.zeros((maxnodes, e.ndof))

    # Extract coords of nodes, DOF for the current element
    for a in range(e.nnodes):
        X_e[a, :e.ncoord] = X[conn[a], :e.ncoord]
        du_e[a, :e.ndof] = [du[e.ndof * conn[a] + i] for i in range(e.ndof)]
        continue

    # Get the element stiffness
    k = e.stiffness(dt, X_e, du_e)
    return k
