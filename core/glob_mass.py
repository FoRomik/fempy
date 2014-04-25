import numpy as np
from scipy.weave import inline, converters

from configvars import HAS_CCOMPILER

def global_mass(X, elements, connect, nproc):
    """Assemble the global mass matrix

    Parameters
    ----------
    X : array like, (i, j,)
        Nodal coordinates
        X[i, j] -> jth coordinate of ith node for i=1...nnode, j=1...ncoord

    elements : array_like, (i,)
        Array of element classes

    connect : array_like, (i, j,)
        Nodal connections
        connect[i, j] -> jth node on the ith element

    Returns
    -------
    mass : array_like

    """
    maxnodes = np.amax([x.nnodes for x in elements])
    ndof = elements[0].ndof
    sdim = ndof * X.shape[0]
    mass = np.zeros((sdim, sdim))

    # Loop over all the elements
    for (lmn, element) in enumerate(elements):

        mel = _elmass(lmn, element, maxnodes, connect[lmn], X)

        # Add the current element mass to the global mass
        nnodes = elements[lmn].nnodes
        ncoord = elements[lmn].ncoord
        ndof = elements[lmn].ndof
        if not HAS_CCOMPILER:
            for a in range(nnodes):
                for i in range(ndof):
                    for b in range(nnodes):
                        for k in range(ndof):
                            rw = ndof * connect[lmn, a] + i
                            cl = ndof * connect[lmn, b] + k
                            elrw = ndof * a + i
                            elcl = ndof * b + k
                            mass[rw, cl] += mel[elrw, elcl]
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
                        mass(rw, cl) += mel(elrw, elcl);
                        }
                      }
                    }
                  }
                """
            inline(code, ["nnodes", "ndof", "lmn", "connect", "mass", "mel"],
                   type_converters=converters.blitz)

    return mass


def _elmass(args):
    """Helper function for computing an elements mass matrix

    Parameters
    ----------
    args : tuple
        Contains: lmn, element, maxnodes, conn, X)

    Returns
    -------
    lmn : int
        Element number

    m : ndarray
        Element mass matrix

    Notes
    -----
    Helper function for compting element mass on multiple processors

    """
    lmn, e, maxnodes, conn, X  = args
    # X_e will hold the original coordinates for nodes on element
    X_e = np.zeros((maxnodes, e.ncoord))

    # Extract coords of nodes, DOF for the current element
    for a in range(e.nnodes):
        X_e[a, :e.ncoord] = X[conn[a], :e.ncoord]

    # element mass
    m = e.mass(X_e)

    return lmn, m
