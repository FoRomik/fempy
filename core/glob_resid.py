import numpy as np
from scipy.weave import inline, converters

from runopts import ENABLE_WEAVE


def global_residual(t, dt, X, elements, connect, du):
    """Assemble the global residual vector

    Parameters
    ----------
    t : float
        Current simulation time

    dt : float
        Time step

    X : array like, (i, j,)
        Nodal coordinates
        X[i, j] -> jth coordinate of ith node

    elements : array_like, (i,)
        List of element objects

    connect : array_like, (i, j,)
        List of nodes on the jth element

    du: array_like, (i, 1)
        Nodal displacements increments


    Returns
    -------
    resid : array_like
        Global residual vector

    """

    # Assemble the global residual vector
    nnode = len(np.unique(connect))
    maxnodes = np.amax([x.nnodes for x in elements])
    ndof = elements[0].ndof
    ncoord = elements[0].ncoord

    resid = np.zeros((ndof * nnode))
    X_e = np.zeros((maxnodes, ncoord))
    du_e = np.zeros((maxnodes, ndof))

    for lmn, element in enumerate(elements):

        # Extract coords of nodes,  DOF for the current element
        for a in range(element.nnodes):
            X_e[a, :ncoord] = X[connect[lmn, a], :ncoord]
            du_e[a, :ndof] = [du[ndof * connect[lmn, a] + i] for i in range(ndof)]
            continue

        rel = element.residual(dt, X_e, du_e)
        nnodes = int(element.nnodes)

        # Add the current element residual to the global residual
        if not ENABLE_WEAVE:
            for a in range(nnodes):
                for i in range(ndof):
                    rw = ndof * connect[lmn, a] + i
                    resid[rw] = resid[rw] + rel[ndof * a + i]

        else:
            code = """
              int rw;
              for (int a=0; a<nnodes; ++a){
                for (int i=0; i<ndof; ++i) {
                    rw = ndof * connect(lmn, a) + i;
                    resid(rw) = resid(rw) + rel(ndof * a + i);
                  }
                }
                """
            inline(code,
                   ["nnodes", "ndof", "connect", "resid", "lmn", "rel"],
                   type_converters=converters.blitz)

        continue

    return resid
