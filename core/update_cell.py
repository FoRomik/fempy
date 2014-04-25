import sys
import numpy as np

from runopts import DEBUG

def update_element_states(t, dt, X, elements, connect, u, du):
    """Update the stress and state variables

    Parameters
    ----------
    dt : real
        Time step

    X : array like, (i, j,)
        ith coord of jth node, for i=1...ncoord; j=1...nnode

    connect : array_like, (i, j,)
        List of nodes on the jth element

    elements : array_like, (i,)
        Array of element class instances

    du : array_like, (i, 1)
        Nodal displacement increments

    Returns
    -------

    """
    maxnodes = np.amax([x.nnodes for x in elements])
    ndof = elements[0].ndof
    ncoord = elements[0].ncoord
    X_e = np.zeros((maxnodes, ncoord))
    du_e = np.zeros((maxnodes, ndof))
    u_e = np.zeros((maxnodes, ndof))

    if DEBUG:
        ti = time.time()
        sys.stdout.write("*** entering: "
                         "fe.core.utilities.update_element_states\n")

    # Loop over all the elements
    for element in elements:
        lmn = element.elid
        # Extract coords of nodes,  DOF for the current element
        for a in range(element.nnodes):
            X_e[a, :ncoord] = X[connect[lmn, a], :ncoord]
            du_e[a, :ndof] = [du[ndof * connect[lmn, a] + i] for i in range(ndof)]
            u_e[a, :ndof] = [u[ndof * connect[lmn, a] + i] for i in range(ndof)]
            continue
        element.update_state(t, dt, X_e, u_e, du_e)
        continue # lmn

    if DEBUG:
        sys.stdout.write("***  exiting: "
                         "core.update_element_states (%.2fs)\n"
                         % (time.time() - ti))
    return
