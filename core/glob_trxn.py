import sys
import numpy as np


def global_traction(t, dt, X, elements, connect, tractions, nforces, du):
    """Assemble the global traction vector

    Parameters
    ----------
    t : float
        Current simulation time

    dt : float
        Time step

    elements : array_like, (i,)
        Element class list

    connect : array_like, (i, j,)
        List of nodes on the jth element

    du : array_like, (i, 1)
        Nodal displacement increments

    tractions : array_like, (i, j,)
        List of element tractions
            tractions[i, 0] -> Element number
            tractions[i, 1] -> face number
            tractions[i, 2:] -> Components of traction as a function of time

    nforces : array_like, (i, j,)
        List of nodal forces
            nforces[i, 0] -> Node number
            nforces[i, 1] -> DOF
            nforces[i, 2] -> Component of force as a function of time

    Returns
    -------
    r : array_like
        The global traction vector

    """

    nnode = len(np.unique(connect))
    ndof = elements[0].ndof
    ncoord = elements[0].ncoord
    r = np.zeros((ndof * nnode))
    traction = np.zeros((ndof))

    for trx in tractions:

        # Extract the coords of the nodes on the appropriate element face
        lmn, face = [int(x) for x in trx[:2]]

        element = elements[lmn]
        nfnodes = element.nface_nodes(face)
        nodelist = element.face_nodes(face)
        X_e = np.zeros((nfnodes, ncoord))

        for a in range(nfnodes):
            X_e[a, :ncoord] = X[connect[lmn, nodelist[a]], :ncoord]
            continue

        # Compute the element load vector
        rel = element.von_neumann(t, X_e, trx[2:])

        # Assemble the element load vector into global vector
        for a in range(nfnodes):
            for i in range(ndof):
                rw = connect[lmn, nodelist[a]] * ndof + i
                r[rw] += rel[a * ndof + i]
                continue
            continue

        continue

    # Add nodal contributions
    for (node, dof, fcn) in nforces:
        rw = node * ndof + dof
        r[rw] += fcn(t)

    return r
