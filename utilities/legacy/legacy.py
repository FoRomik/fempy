import numpy as np
from src.base.errors import WasatchError
import src.base.consts as consts

def numberofintegrationpoints(ncoord, nelnodes, elident):
    """Return the number of integration points for each element type

    Parameters
    ----------
    ncoord : int
        No. spatial coords (2 for 2D, 3 for 3D)

    nelnodes : int
        No. nodes on the element

    elident : int
        Element identifier

    Returns
    -------
    n : int
        Number of integration points on element

    """
    n = None

    if ncoord == 1:

        n = nelnodes

    elif ncoord == 2:

        if nelnodes == 3:
            n = 1

        elif nelnodes == 6:
            n = 3

        elif nelnodes == 4:
            n = 4

        elif nelnodes == 8:
            n = 9

    elif ncoord == 3:

        if nelnodes == 4:
            n = 1

        elif nelnodes == 10:
            n = 4

        elif nelnodes == 8:
            n = 8

        elif nelnodes == 20:
            n = 27

    if n is None:
        raise WasatchError("error: {0}D element with {1} nodes not supported"
                         .format(ncoord, nelnodes))
    return n

def integrationpoints(ncoord, nelnodes, npoints, elident):
    """Define positions of integration points

    Parameters
    ----------
    ncoord : int
        No. spatial coords (2 for 2D, 3 for 3D)

    nelnodes : int
        No. nodes on the element

    npoints : int
        No. integration points on the element

    elident : int
        element identity

    Returns
    -------
    xi : array_like, (i, j)
        Integration points
            xi[i, j] ith coordinate of the jth position

   """
    xi = None

    # 1D elements
    if ncoord == 1:

        if npoints == 3:
            x = 0.7745966692
            xi = np.array([[-x, 0., x]])

    # 2D elements
    elif ncoord == 2:

        # Triangular element
        if nelnodes == 3 or nelnodes == 6:

            if npoints == 4:
                x = (1. / 3., .6, .2)
                xi = np.array([[x[0], x[1], x[2], x[2]],
                               [x[0], x[2], x[1], x[2]]])

        # Rectangular element
        elif nelnodes==4 or nelnodes==8:

            if npoints == 1:
                x = 0.
                xi = np.array([[x], [x]])

            elif npoints == 9:
                x = 0.7745966692
                xi = np.array([[-x, 0., x, -x, 0., x, -x, 0., x],
                               [-x, -x, -x, 0., 0., 0., x, x, x]])

    # 3D elements
    elif ncoord == 3:
        if nelnodes == 4 or nelnodes==10:
            if npoints == 1:
                x = .25
                xi = np.array([[x], [x], [x]])

            elif npoints == 4:
                x = (0.58541020, 0.13819660)
                xi = np.array([[x[0], x[1], x[1], x[1]],
                               [x[1], x[0], x[1], x[1]],
                               [x[1], x[1], x[0], x[1]]])

            elif nelnodes==8 or nelnodes==20:

                if npoints == 8:
                    xi = np.zeros((ncoord, npoints))
                    x = (-0.5773502692, 0.5773502692)
                    for k in range(2):
                        for j in range(2):
                            for i in range(2):
                                n = 4 * k + 2 * j + i
                                xi[0, n] = x[i]
                                xi[1, n] = x[j]
                                xi[2, n] = x[k]
                                continue
                            continue
                        continue

                elif npoints == 27:
                    xi = np.zeros((ncoord, npoints))
                    x = (-0.7745966692, 0., 0.7745966692)
                    for k in range(3):
                        for j in range(3):
                            for i in range(3):
                                n = 9 * k + 3 * j + i
                                xi[0, n] = x[i]
                                xi[1, n] = x[j]
                                xi[2, n] = x[k]
                                continue
                            continue
                        continue

    if xi is None:
        raise SystemExit("error: {0}D element with {1} nodes and {2} "
                         "integration points not supported"
                         .format(ncoord, nelnodes))

    if xi.shape != (ncoord, npoints):
        raise SystemExit("error: wrong integration points shape")

    return np.transpose(xi)


def integrationweights(ncoord, nelnodes, npoints, elident):
    """Defines integration weights w_i

    Parameters
    ----------
    ncoord : int
        No. spatial coords (2 for 2D, 3 for 3D)

    nelnodes : int
        No. nodes on the element

    npoints : int
        No. integration points on the element

    elident : int
        element identity

    Returns
    -------
    w : array_like, (i, 1)
        ith integration weight

    """

    w = None

    #  1D elements
    if ncoord == 1:

        if npoints == 3:
            x = (.5555555555, .8888888888)
            w = np.array([x[0], x[1], x[0]])

    # 2D elements
    elif ncoord == 2:

        # Triangular element
        if nelnodes == 3 or nelnodes == 6:

            if npoints == 4:
                x = (-27. / 96., 25. / 96)
                w = np.array([x[0], x[1], x[1], x[1]])

        # Rectangular element
        elif nelnodes==4 or nelnodes==8:

            if npoints == 4:
                x = 1.
                w = np.array([x for _ in range(4)])

            elif npoints == 9:
                x = (0.555555555, 0.888888888, 0.55555555555)
                for j in range(3):
                    for i in range(3):
                        n = 3 * j + i
                        w[n] = x[i] * x[j]
                        continue
                    continue

    # 3D elements
    elif ncoord == 3:

        if nelnodes == 4 or nelnodes == 10:

            if npoints == 1:
                x = 1. / 6.
                w = np.array([x])

            elif npoints == 4:
                x = 1. / 24.
                w = np.array([x for _ in range(4)])

        elif nelnodes==8 or nelnodes==20:

            if npoints == 8:
                x = 1.
                w = np.array([x for _ in range(8)])

            elif npoints == 27:
                x = (0.555555555, 0.888888888, 0.55555555555)
                for k in range(3):
                    for j in range(3):
                        for i in range(3):
                            n = 9 * k + 3 * j + i
                            w[n] = x[i] * x[j] * x[k]
                            continue
                        continue
                    continue

    if w is None:
        raise SystemExit("error: {0}D element with {1} nodes and {2} "
                         "integration points not supported"
                         .format(ncoord, nelnodes))
#    if w.shape[1] != 1:
#        raise SystemExit("error: wrong integration weight shape")

    return w


def shapefunctions(nelnodes, ncoord, elident, xi):
    """Calculates shape functions for various element types

    Parameters
    ----------
    nelnodes : int
        No. nodes on the element

    ncoord : int
        No. spatial coords (2 for 2D, 3 for 3D)

    elident : int
        element identity

    xi : array_like, (i,)
        Integration point
        xi[i] component of ith integration point on element

    Returns
    -------
    N : array_like, (i,)
        Shape functions
            N[i] ith shape function component

    """

    N = None

    # 1D elements
    if ncoord == 1:

        x = xi[0]

        if nelnodes == 2:
            N = np.array([.5 * (1. + x), .5 * (1. - x)])

        elif nelnodes == 3:
            N = np.array([-.5 * x * (1. - x),
                            .5 * x * (1. + x),
                            (1. - x) * (1. + x)])

    #  2D elements
    elif ncoord == 2:

        x, y = xi[:2]
        xy = x * y

        if  nelnodes == 6:
            xi3 = 1. - x - y
            N = np.array([(2. * x - 1.) * x,
                          (2. * y - 1.) * y,
                          (2. * xi3 - 1.) * xi3,
                          4. * x * y,
                          4. * y * xi3,
                          4. * xi3 * x])

        elif nelnodes == 8:
            N = np.array([-.25 * (1. - x) * (1. - y) * (1. + x + y),
                           .25 * (1. + x) * (1. - y) * (x - y - 1.),
                           .25 * (1. + x) * (1. + y) * (x + y - 1.),
                           .25 * (1. - x) * (1. + y) * (y - x - 1.),
                            .5 * (1. - x * x) * (1. - y),
                            .5 * (1. + x) * (1. - y * y),
                            .5 * (1. - x * x) * (1. + y),
                            .5 * (1. - x) * (1. - y * y)])

    # 3D elements
    elif ncoord==3:

        x, y, z = xi[:3]

        if nelnodes == 4:
            N = np.array([x, y, z, 1. - x - y - z])

        elif nelnodes == 10:
            xi4 = 1. - x - y - z
            N = np.array([(2. * x - 1.) * x,
                          (2. * y - 1.) * y,
                          (2. * z - 1.) * z,
                          (2. * xi4 - 1.) * xi4,
                          4. * x * y,
                          4. * y * z,
                          4. * z * x,
                          4. * x * xi4,
                          4. * y * xi4,
                          4. * z * xi4])


        elif nelnodes == 20:
            omxi0, opxi0 = 1. - x, 1. + x
            omxi1, opxi1 = 1. - y, 1. + y
            omxi2, opxi2 = 1. - z, 1. + z
            N = np.array([
                    omxi0 * omxi1 * omxi2 * (-x - y - z - 2.) / 8.,
                    opxi0 * omxi1 * omxi2 * (x - y - z - 2.) / 8.,
                    opxi0 * opxi1 * omxi2 * (x + y - z - 2.) / 8.,
                    omxi0 * opxi1 * omxi2 * (-x + y - z - 2.) / 8.,
                    omxi0 * omxi1 * opxi2 * (-x - y + z - 2.) / 8.,
                    opxi0 * omxi1 * opxi2 * (x - y + z - 2.) / 8.,
                    opxi0 * opxi1 * opxi2 * (x + y + z - 2.) / 8.,
                    omxi0 * opxi1 * opxi2 * (-x + y + z - 2.) / 8.,
                    (1. - x ** 2) * omxi1 * omxi2 / 4.,
                    opxi0 * (1. - y ** 2) * omxi2 / 4.,
                    (1. - x ** 2) * opxi1 * omxi2 / 4.,
                    omxi0 * (1. - y ** 2) * omxi2 / 4.,
                    (1. - x ** 2) * omxi1 * opxi2 / 4.,
                    opxi0 * (1. - y ** 2) * opxi2 / 4.,
                    (1. - x ** 2) * opxi1 * opxi2 / 4.,
                    omxi0 * (1. - y ** 2) * opxi2 / 4.,
                    omxi0 * omxi1 * (1. - z ** 2) / 4.,
                    opxi0 * omxi1 * (1. - z ** 2) / 4.,
                    opxi0 * opxi1 * (1. - z ** 2) / 4.,
                    omxi0 * opxi1 * (1. - z ** 2) / 4.])

    if N is None:
        raise SystemExit("error: {0}D element with {1} nodes and {2} "
                         "integration points not supported".format(ncoord, nelnodes))

    if N.shape != (nelnodes,):
        raise SystemExit("error: wrong shape function shape")

    return N


def shapefunctionderivs(nelnodes, ncoord, elident, xi):
    """Calculate the derivative of the shape function at xi
    Parameters
    ----------
    nelnodes : int
        No. nodes on the element

    ncoord : int
        No. spatial coords (2 for 2D, 3 for 3D)

    elident : int
        element identity

    xi : array_like, (i,)
        Integration point
        xi[i] component of ith integration point on element

    Returns
    -------
    dNdxi : array_like, (nelnodes, ncoord)
        Shape function derivative
            dNdxi[i, j] ith shape function derivative of the jth coordinate

    """
    dNdxi = None
    dNdxi = np.zeros((nelnodes, ncoord))

    #  1D elements
    if ncoord == 1:

        if nelnodes == 3:
            dNdxi[0, 0] =  - 0.5 + xi[0]
            dNdxi[1, 0] = 0.5 + xi[0]
            dNdxi[2, 0] =  - 2. * xi[0]

    #  2D elements
    elif ncoord == 2:

        x, y = xi[:2]

        if nelnodes == 6:
            xi3 = 1. - xi[0] - xi[1]
            dNdxi[0, 0] = 4. * xi[0] - 1.
            dNdxi[1, 1] = 4. * xi[1] - 1.
            dNdxi[2, 0] =  - (4. * xi3 - 1.)
            dNdxi[2, 1] =  - (4. * xi3 - 1.)
            dNdxi[3, 0] = 4. * xi[1]
            dNdxi[3, 1] = 4. * xi[0]
            dNdxi[4, 0] =  - 4. * xi[1]
            dNdxi[4, 1] =  - 4. * xi[0]
            dNdxi[5, 0] = 4. * xi3  -  4. * xi[0]
            dNdxi[5, 1] = 4. * xi3  -  4. * xi[1]

        elif nelnodes == 8:
            dNdxi[0, 0] = 0.25 * (1. - xi[1]) * (2. * xi[0] + xi[1])
            dNdxi[0, 1] = 0.25 * (1. - xi[0]) * (xi[0] + 2. * xi[1])
            dNdxi[1, 0] = 0.25 * (1. - xi[1]) * (2. * xi[0] - xi[1])
            dNdxi[1, 1] = 0.25 * (1. + xi[0]) * (2. * xi[1] - xi[0])
            dNdxi[2, 0] = 0.25 * (1. + xi[1]) * (2. * xi[0] + xi[1])
            dNdxi[2, 1] = 0.25 * (1. + xi[0]) * (2. * xi[1] + xi[0])
            dNdxi[3, 0] = 0.25 * (1. + xi[1]) * (2. * xi[0] - xi[1])
            dNdxi[3, 1] = 0.25 * (1. - xi[0]) * (2. * xi[1] - xi[0])
            dNdxi[4, 0] =  - xi[0] * (1. - xi[1])
            dNdxi[4, 1] =  - 0.5 * (1. - xi[0] * xi[0])
            dNdxi[5, 0] = 0.5 * (1. - xi[1] * xi[1])
            dNdxi[5, 1] =  - (1. + xi[0]) * xi[1]
            dNdxi[6, 0] =  - xi[0] * (1. + xi[1])
            dNdxi[6, 1] = 0.5 * (1. - xi[0] * xi[0])
            dNdxi[7, 0] =  - 0.5 * (1. - xi[1] * xi[1])
            dNdxi[7, 1] =  - (1. - xi[0]) * xi[1]

    #  3D elements
    elif ncoord==3:

        if nelnodes == 4:
            dNdxi[0, 0] = 1.
            dNdxi[1, 1] = 1.
            dNdxi[2, 2] = 1.
            dNdxi[3, 0] =  - 1.
            dNdxi[3, 1] =  - 1.
            dNdxi[3, 2] =  - 1.

        elif nelnodes == 10:
            xi4 = 1. - xi[0] - xi[1] - xi[2]
            dNdxi[0, 0] = (4. * xi[0] - 1.)
            dNdxi[1, 1] = (4. * xi[1] - 1.)
            dNdxi[2, 2] = (4. * xi[2] - 1.)
            dNdxi[3, 0] =  - (4. * xi4 - 1.)
            dNdxi[3, 1] =  - (4. * xi4 - 1.)
            dNdxi[3, 2] =  - (4. * xi4 - 1.)
            dNdxi[4, 0] = 4. * xi[1]
            dNdxi[4, 1] = 4. * xi[0]
            dNdxi[5, 1] = 4. * xi[2]
            dNdxi[5, 2] = 4. * xi[1]
            dNdxi[6, 0] = 4. * xi[2]
            dNdxi[6, 2] = 4. * xi[0]
            dNdxi[7, 0] = 4. * (xi4 - xi[0])
            dNdxi[7, 1] =  - 4. * xi[0]
            dNdxi[7, 2] =  - 4. * xi[0]
            dNdxi[8, 0] =  - 4. * xi[1]
            dNdxi[8, 1] = 4. * (xi4 - xi[1])
            dNdxi[8, 2] =  - 4. * xi[1]
            dNdxi[9, 0] =  - 4. * xi[2] * xi4
            dNdxi[9, 1] =  - 4. * xi[2]
            dNdxi[9, 2] = 4. * (xi4 - xi[2])

        elif nelnodes == 20:
            dNdxi[0, 0] = ( - (1. - xi[1]) * (1. - xi[2]) * ( - xi[0] - xi[1] - xi[2] - 2.) - (1. - xi[0]) * (1. - xi[1]) * (1. - xi[2])) / 8.
            dNdxi[0, 1] = ( - (1. - xi[0]) * (1. - xi[2]) * ( - xi[0] - xi[1] - xi[2] - 2.) - (1. - xi[0]) * (1. - xi[1]) * (1. - xi[2])) / 8.
            dNdxi[0, 2] = ( - (1. - xi[0]) * (1. - xi[1]) * ( - xi[0] - xi[1] - xi[2] - 2.) - (1. - xi[0]) * (1. - xi[1]) * (1. - xi[2])) / 8.
            dNdxi[1, 0] = ((1. - xi[1]) * (1. - xi[2]) * (xi[0] - xi[1] - xi[2] - 2.) + (1. + xi[0]) * (1. - xi[1]) * (1. - xi[2])) / 8.
            dNdxi[1, 1] = ( - (1. + xi[0]) * (1. - xi[2]) * (xi[0] - xi[1] - xi[2] - 2.) - (1. + xi[0]) * (1. - xi[1]) * (1. - xi[2])) / 8.
            dNdxi[1, 2] = ( - (1. + xi[0]) * (1. - xi[1]) * (xi[0] - xi[1] - xi[2] - 2.) - (1. + xi[0]) * (1. - xi[1]) * (1. - xi[2])) / 8.
            dNdxi[2, 0] = ((1. + xi[1]) * (1. - xi[2]) * (xi[0] + xi[1] - xi[2] - 2.) + (1. + xi[0]) * (1. + xi[1]) * (1. - xi[2])) / 8.
            dNdxi[2, 1] = ((1. + xi[0]) * (1. - xi[2]) * (xi[0] + xi[1] - xi[2] - 2.) + (1. + xi[0]) * (1. + xi[1]) * (1. - xi[2])) / 8.
            dNdxi[2, 2] = ( - (1. + xi[0]) * (1. + xi[1]) * (xi[0] + xi[1] - xi[2] - 2.) - (1. + xi[0]) * (1. + xi[1]) * (1. - xi[2])) / 8.
            dNdxi[3, 0] = ( - (1. + xi[1]) * (1. - xi[2]) * ( - xi[0] + xi[1] - xi[2] - 2.) - (1. - xi[0]) * (1. + xi[1]) * (1. - xi[2])) / 8.
            dNdxi[3, 1] = ((1. - xi[0]) * (1. - xi[2]) * ( - xi[0] + xi[1] - xi[2] - 2.) + (1. - xi[0]) * (1. + xi[1]) * (1. - xi[2])) / 8.
            dNdxi[3, 2] = ( - (1. - xi[0]) * (1. + xi[1]) * ( - xi[0] + xi[1] - xi[2] - 2.) - (1. - xi[0]) * (1. + xi[1]) * (1. - xi[2])) / 8.
            dNdxi[4, 0] = ( - (1. - xi[1]) * (1. + xi[2]) * ( - xi[0] - xi[1] + xi[2] - 2.) - (1. - xi[0]) * (1. - xi[1]) * (1. + xi[2])) / 8.
            dNdxi[4, 1] = ( - (1. - xi[0]) * (1. + xi[2]) * ( - xi[0] - xi[1] + xi[2] - 2.) - (1. - xi[0]) * (1. - xi[1]) * (1. + xi[2])) / 8.
            dNdxi[4, 2] = ((1. - xi[0]) * (1. - xi[1]) * ( - xi[0] - xi[1] + xi[2] - 2.) + (1. - xi[0]) * (1. - xi[1]) * (1. + xi[2])) / 8.
            dNdxi[5, 0] = ((1. - xi[1]) * (1. + xi[2]) * (xi[0] - xi[1] + xi[2] - 2.) + (1. + xi[0]) * (1. - xi[1]) * (1. + xi[2])) / 8.
            dNdxi[5, 1] = ( - (1. + xi[0]) * (1. + xi[2]) * (xi[0] - xi[1] + xi[2] - 2.) - (1. + xi[0]) * (1. - xi[1]) * (1. + xi[2])) / 8.
            dNdxi[5, 2] = ((1. + xi[0]) * (1. - xi[1]) * (xi[0] - xi[1] + xi[2] - 2.) + (1. + xi[0]) * (1. - xi[1]) * (1. + xi[2])) / 8.
            dNdxi[6, 0] = ((1. + xi[1]) * (1. + xi[2]) * (xi[0] + xi[1] + xi[2] - 2.) + (1. + xi[0]) * (1. + xi[1]) * (1. + xi[2])) / 8.
            dNdxi[6, 1] = ((1. + xi[0]) * (1. + xi[2]) * (xi[0] + xi[1] + xi[2] - 2.) + (1. + xi[0]) * (1. + xi[1]) * (1. + xi[2])) / 8.
            dNdxi[6, 2] = ((1. + xi[0]) * (1. + xi[1]) * (xi[0] + xi[1] + xi[2] - 2.) + (1. + xi[0]) * (1. + xi[1]) * (1. + xi[2])) / 8.
            dNdxi[7, 0] = ( - (1. + xi[1]) * (1. + xi[2]) * ( - xi[0] + xi[1] + xi[2] - 2.) - (1. - xi[0]) * (1. + xi[1]) * (1. + xi[2])) / 8.
            dNdxi[7, 1] = ((1. - xi[0]) * (1. + xi[2]) * ( - xi[0] + xi[1] + xi[2] - 2.) + (1. - xi[0]) * (1. + xi[1]) * (1. + xi[2])) / 8.
            dNdxi[7, 2] = ((1. - xi[0]) * (1. + xi[1]) * ( - xi[0] + xi[1] + xi[2] - 2.) + (1. - xi[0]) * (1. + xi[1]) * (1. + xi[2])) / 8.
            dNdxi[8, 0] =  - 2. * xi[0] * (1. - xi[1]) * (1. - xi[2]) / 4.
            dNdxi[8, 1] =  - (1. - xi[0] ^ 2) * (1. - xi[2]) / 4.
            dNdxi[8, 2] =  - (1. - xi[0] ^ 2) * (1. - xi[1]) / 4.
            dNdxi[9, 0] = (1. - xi[1] ^ 2) * (1. - xi[2]) / 4.
            dNdxi[9, 1] =  - 2. * xi[1] * (1. + xi[0]) * (1. - xi[2]) / 4.
            dNdxi[9, 2] =  - (1. - xi[1] ^ 2) * (1. + xi[0]) / 4.
            dNdxi[10, 0] =  - 2. * xi[0] * (1. + xi[1]) * (1. - xi[2]) / 4.
            dNdxi[10, 1] = (1. - xi[0] ^ 2) * (1. - xi[2]) / 4.
            dNdxi[10, 2] =  - (1. - xi[0] ^ 2) * (1. + xi[1]) / 4.
            dNdxi[11, 0] =  - (1. - xi[1] ^ 2) * (1. - xi[2]) / 4.
            dNdxi[11, 1] =  - 2. * xi[1] * (1. - xi[0]) * (1. - xi[2]) / 4.
            dNdxi[11, 2] =  - (1. - xi[1] ^ 2) * (1. - xi[0]) / 4.
            dNdxi[12, 0] =  - 2. * xi[0] * (1. - xi[1]) * (1. + xi[2]) / 4.
            dNdxi[12, 1] =  - (1. - xi[0] ^ 2) * (1. + xi[2]) / 4.
            dNdxi[12, 2] = (1. - xi[0] ^ 2) * (1. - xi[1]) / 4.
            dNdxi[13, 0] = (1. - xi[1] ^ 2) * (1. + xi[2]) / 4.
            dNdxi[13, 1] =  - 2. * xi[1] * (1. + xi[0]) * (1. + xi[2]) / 4.
            dNdxi[13, 2] = (1. - xi[1] ^ 2) * (1. + xi[0]) / 4.
            dNdxi[14, 0] =  - 2. * xi[0] * (1. + xi[1]) * (1. + xi[2]) / 4.
            dNdxi[14, 1] = (1. - xi[0] ^ 2) * (1. + xi[2]) / 4.
            dNdxi[14, 2] = (1. - xi[0] ^ 2) * (1. + xi[1]) / 4.
            dNdxi[15, 0] =  - (1. - xi[1] ^ 2) * (1. + xi[2]) / 4.
            dNdxi[15, 1] =  - 2. * xi[1] * (1. - xi[0]) * (1. + xi[2]) / 4.
            dNdxi[15, 2] = (1. - xi[1] ^ 2) * (1. - xi[0]) / 4.
            dNdxi[16, 0] =  - (1. - xi[1]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[16, 1] =  - (1. - xi[0]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[16, 2] =  - xi[2] * (1. - xi[0]) * (1. - xi[1]) / 2.
            dNdxi[17, 0] = (1. - xi[1]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[17, 1] =  - (1. + xi[0]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[17, 2] =  - xi[2] * (1. + xi[0]) * (1. - xi[1]) / 2.
            dNdxi[18, 0] = (1. + xi[1]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[18, 1] = (1. + xi[0]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[18, 2] =  - xi[2] * (1. + xi[0]) * (1. + xi[1]) / 2.
            dNdxi[19, 0] =  - (1. + xi[1]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[19, 1] = (1. - xi[0]) * (1. - xi[2] ^ 2) / 4.
            dNdxi[19, 2] =  - xi[2] * (1. - xi[0]) * (1. + xi[1]) / 2.

    if dNdxi is None:
        raise SystemExit("error: {0}D element with {1} nodes and {2} "
                         "integration points not supported"
                         .format(ncoord, nelnodes))

    if not np.any(dNdxi != 0.):
        raise SystemExit("error: {0}D element with {1} nodes and {2} "
                         "integration points not supported "
                         "[all dNdxi = 0.]".format(ncoord, nelnodes))

    if dNdxi.shape != (nelnodes, ncoord):
        raise SystemExit("error: wrong shape function derivative shape")

    # tjfulle : modify all to not have to do transpose
    return np.transpose(dNdxi)


def nfacenodes(ncoord, nelnodes, elident, face):
    """Return the number of nodes on each element face

    Parameters
    ----------
    ncoord : int
        No. spatial coords (2 for 2D, 3 for 3D)

    nelnodes : int
        No. nodes on the element

    elident : int
        Element identifier

    face : int
        Element face number

    Returns
    -------
    n : int
        Number of nodes on element face

    Notes
    -----
    Number of face nodes is needed for computing the surface integrals
    associated with the element traction vector

    """
    n = None
    if ncoord == 2:
        if nelnodes == 6 or nelnodes == 8:
            n=3

    elif ncoord == 3:
        if nelnodes == 4:
            n = 3
        elif nelnodes == 10:
            n = 6
        elif nelnodes == 20:
            n = 8

    if n is None:
        raise SystemExit("Number of face nodes not defined for element")

    return n


def facenodes(ncoord, nelnodes, elident, face):
    """Return the node numbers for the face

    Parameters
    ----------
    ncoord : int
        No. spatial coords (2 for 2D, 3 for 3D)

    nelnodes : int
        No. nodes on the element

    elident : int
        Element identifier

    face : int
        Element face number

    Returns
    -------
    nodes : array_like
        List of nodes on element face

    """
    i3 = [1, 2, 0]
    i4 = [1, 2, 3, 0]

    nodes = None

    # 2D
    if ncoord == 2:
        if nelnodes == 6:
            nodes[0] = [face, i3[face], face + 2]

        elif nelnodes == 8:
            nodes = [face, i4[face], face + 3]

    # 3D
    elif ncoord == 3:

        if nelnodes==4:

            if face == 0:
                nodes = [0, 1, 2]

            elif face == 1:
                nodes = [0, 3, 1]

            elif face == 2:
                nodes = [1, 3, 2]

            elif face == 3:
                nodes = [2, 3, 0]

        elif nelnodes == 10:

            if face == 0:
                nodes = [0, 1, 2, 4, 5, 6]

            elif face == 1:
                nodes = [0, 3, 1, 7, 8, 4]

            elif face == 2:
                nodes = [1, 3, 2, 8, 9, 5]

            elif face == 3:
                nodes = [2, 3, 0, 9, 7, 6]

        elif nelnodes == 20:

            if face == 0:
                nodes = [0, 1, 2, 3, 8, 9, 10, 11]

            elif face == 1:
                nodes = [4, 7, 6, 5, 15, 14, 13, 12]

            elif face == 2:
                nodes = [0, 4, 5, 1, 16, 12, 17, 8]

            elif face == 3:
                nodes = [1, 5, 6, 2, 17, 13, 18, 9]

            elif face == 4:
                nodes = [2, 6, 7, 3, 18, 14, 19, 10]

            elif face == 5:
                nodes = [3, 7, 4, 0, 19, 15, 16, 11]

    if nodes is None:
        raise SystemExit("Face nodes were not defined")

    nodes = np.array(nodes)

    return nodes

def fake_more_junk():

    E = 96
    nu = 1. / 3.
    # plane stress
    if cfg.sqa:

        if not np.allclose(kel, kel.T):
            print "kel not symmetric"
            print kel != kel.T
            errors += 1

        ka = np.array([[ 42,  18,  -6,   0, -21, -18, -15,   0],
                       [ 18,  78,   0,  30, -18, -39,   0, -69],
                       [ -6,   0,  42, -18, -15,   0, -21,  18],
                       [  0,  30, -18,  78,   0, -69,  18, -39],
                       [-21, -18, -15,   0,  42,  18,  -6,   0],
                       [-18, -39,   0, -69,  18,  78,   0,  30],
                       [-15,   0, -21,  18,  -6,   0,  42, -18],
                       [  0, -69,  18, -39,   0,  30, -18,  78]],
                      dtype=np.float)

        if not np.allclose(kel, ka, rtol=1e-4, atol=1e-4):
            print "Bad kel"
            utils.print_array(abs(ka - kel) < 1e-3)
            utils.print_array(ka)
            print
            utils.print_array(kel)


def fake_other():
    if False:
        if cfg.sqa:

            if not np.allclose(kel, kel.T):
                print "kel not symmetric"
                print kel != kel.T

            ka = np.array([[ 42,  18,  -6,   0, -21, -18, -15,   0],
                           [ 18,  78,   0,  30, -18, -39,   0, -69],
                           [ -6,   0,  42, -18, -15,   0, -21,  18],
                           [  0,  30, -18,  78,   0, -69,  18, -39],
                           [-21, -18, -15,   0,  42,  18,  -6,   0],
                           [-18, -39,   0, -69,  18,  78,   0,  30],
                           [-15,   0, -21,  18,  -6,   0,  42, -18],
                           [  0, -69,  18, -39,   0,  30, -18,  78]],
                          dtype=np.float)

            if not np.allclose(kel, ka, rtol=1e-4, atol=1e-4):
                print "Bad kel"
                utils.print_array(abs(ka - kel) < 1e-3)
                utils.print_array(ka)
                print
                utils.print_array(kel)
                sys.exit('bad stiff')
            sys.exit('good stiff')


def plane_stress_unit(nel):
    import src.fem.element.element as el

    if nel not in (1, 2):
        raise WasatchError("wrong NEL")

    # E, nu, Y, e0, n, edot0, m
    materialprops = np.array([96, .333333, 1.e99, 0., 0., 0., 1.])
    ncoord = 2
    ndof = 2

    def _mesh(n):
        """     3        [2]        2
                 o --------------- o
                 |                 |
                 |                 |
             [3] |                 | [1]
                 |                 |
                 |                 |
                 o --------------- o
                0        [0]        1
        """
        if n == 1:
            # 1 element
            coords = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
            connect = np.array([[0, 1, 2, 3]])

        elif n == 2:
            # 2 elements
            coords = np.array([[0., 0.], [.5, 0.], [.5, 1.], [0., 1.],
                               [.5, 0.], [1., 0.], [1., 1.], [.5, 1.]])
            connect = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

        return coords, connect

    coords, connect = _mesh(nel)
    nelnodes = np.array([len(x != np.nan) for x in connect])
    elident = np.array([el.initialize("Quad4") for i in range(len(connect))])

    # fixnodes : List of prescribed displacements at nodes
    #   fixnodes[i, 0] -> Node number
    #   fixnodes[i, 1] -> Displacement component (x: 0, y: 1, or z: 2)
    #   fixnodes[i, 2] -> Value of the displacement (function)
    comp = lambda x: {"x": 0, "y": 1, "z": 2}.get(x.lower())
    fixnodes = np.array([[0, comp("x"), lambda t: 0.],
                         [0, comp("y"), lambda t: 0.],
                         [3, comp("x"), lambda t: 0.],
                         [3, comp("y"), lambda t: 0.]])

    # dloads : List of element tractions
    #   dloads[i, 0] -> Element number
    #   dloads[i, 1] -> face number
    #   dloads[i, 2], dloads[i, 3], dloads[i, 4] -> Components of traction
    n = len(elident) - 1
    dloads = np.array([[n, 1, 10.e5, 0., 0.]])

    # control: nsteps, tol, maxit, relax, tstart, tterm, dtmult
    control = np.array([10, 1.e-4, 60, .5, 0., 5., 1.])
    control = np.array([10, 1.e-4, 30, 1., 0., 5., 1.])
    fe_solve(sys.stdout, control, ncoord, ndof, coords,
             connect, nelnodes, elident, fixnodes, dloads)

if __name__ == "fake__main__":

    nel = 1
    argv = sys.argv[1:]
    if argv:
        nel = int(argv[0])

    #plt.xlabel("Displacement")
    #plt.ylabel("Force")
    #plt.plot(forcevdisp[:, 0], forcevdisp[:, 1], c='r', lw=3)

    # Post-processing

    # Create a plot of the deformed mesh
    # nnode = len(np.unique(connect))
    #defcoords = np.zeros((nnode, ndof))
    #scalefactor = 1.0

    #for i in range(nnode):
    #    for j in range(ndof):
    #        defcoords[i, j] = coords[i, j] + scalefactor * dX[ndof * i + j]
    #        continue
    #    continue

    # plotmesh(coords, ncoord, connect, elements, nelnodes, 'g');
    # plotmesh(defcoords, ncoord, connect, elements, nelnodes, 'r');
