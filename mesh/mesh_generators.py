from utilities.errors import WasatchError, UserInputError
import numpy as np

def line_mesh(xmin, xblocks):
    """ create evenly spaced mesh given endpoints and number of elements

    """
    coords = np.array([xmin])
    for i, (xlen, xint) in enumerate(xblocks):
        xl = coords[-1]
        xr = xl + xlen
        n = xint + 1

        # check for bad inputs
        if xl >= xr:
            raise UserInputError("BLOCK {0}: xl > xr".format(i))

        if n <= 0:
            raise UserInputError("XBLOCK {0}: negative interval".format(i))

        # find element seperation
        block_coords = np.linspace(xl, xr, n)

        #force location of endpoints
        block_coords[0] = xl
        block_coords[-1] = xr
        coords = np.append(coords, block_coords[1:])
        continue
    nel = coords.size - 1
    conn = np.array([[i, i + 1] for i in range(nel)])
    return coords, conn


def brick_mesh(xpoints, ypoints, zpoints=None, test=False):
    """Generate a [2,3]D block mesh.

    Parameters
    ----------
    xpoints : array_like
        Points defining the x-axis from x0 to xf
    ypoints : array_like
        Points defining the y-axis from y0 to yf
    zpoints : array_like [optional]
        Points defining the z-axis from z0 to zf

    Returns
    -------
    coords : array_like, (i, j)
        Nodal coordinates
        coords[i, j] -> jth coordinate of ith node
    conn : array_like, (i, j)
        nodal connectivity
        conn[i, j] -> jth node of the ith element

    """
    dim = 2 if zpoints is None else 3

    if dim == 3:
        raise WasatchError("3D inline mesh not done")

    shape = [xpoints.size, ypoints.size,]
    if dim == 3:
        shape.append(zpoints.size)
    shape = np.array(shape, dtype=np.int)

    nnode = np.prod(shape)
    nel = np.prod(shape - 1)

    # Nodal coordinates
    if dim == 3:
        coords = [[x, y, z] for z in zpoints for y in ypoints for x in xpoints]
    else:
        coords = [(x, y, 0) for y in ypoints for x in xpoints]
    coords = np.array(coords, dtype=np.float64)

    # Connectivity
    if dim == 2:
        row = 0
        conn = np.zeros((nel, 4), dtype=np.int)
        nelx = xpoints.size - 1
        for lmn in range(nel):
            ii = lmn + row
            conn[lmn, :] = [ii, ii + 1, ii + nelx + 2, ii + nelx + 1]
            if (lmn + 1) % (nelx) == 0:
                row += 1
            continue

    else:
        grid = np.zeros(shape, dtype=np.int)
        for ii, ic in enumerate(_cycle(shape)):
            grid[tuple(ic)] = ii
        conn = np.zeros((nel, 8), dtype=np.int)
        for ii, (ix, iy, iz) in enumerate(_cycle(shape - 1)):
            conn[ii, :] = [grid[ix, iy, iz  ], grid[ix + 1, iy, iz],
                           grid[ix + 1, iy + 1, iz], grid[ix, iy + 1, iz],
                           grid[ix, iy, iz + 1], grid[ix + 1, iy, iz + 1],
                           grid[ix + 1, iy + 1, iz + 1], grid[ix, iy + 1, iz + 1]]

    return coords, conn

def _cycle(bounds):
    """
    Cycles through all combinations of bounds, returns a generator.

    More specifically, let bounds=[a, b, c, ...], so _cycle returns all
    combinations of lists [0<=i<a, 0<=j<b, 0<=k<c, ...] for all i,j,k,...

    Examples:
    In [9]: list(_cycle([3, 2]))
    Out[9]: [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

    In [14]: list(_cycle([3, 4]))
    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0],
    [2, 1], [2, 2], [2, 3]]

    Reference
    ---------
    1. http://sfepy.org/doc-devel/_modules/sfepy/linalg/utils.html#cycle

    """
    nb = len(bounds)
    if nb == 1:
        for ii in xrange(bounds[0]):
            yield [ii]
    else:
        for ii in xrange(bounds[0]):
            for perm in _cycle(bounds[1:]):
                yield [ii] + perm


if __name__ == "__main__":

    nelx = 4
    nely = 2
    lx = ly = 2
    xpoints = np.linspace(0, lx, nelx + 1)
    ypoints = np.linspace(0, ly, nely + 1)
    shape_2d = np.array([xpoints.size, ypoints.size])
    nnode = np.prod(shape_2d)
    nel = np.prod(shape_2d - 1)

    print "\n\n============================ 2D ==============================="
    print "COORDS"
    coords = np.array([(x, y, 0) for y in ypoints for x in xpoints])
    print coords
    print "\nCONNECTIVITY:"
    row = 0
    conn = np.zeros((nel, 4), dtype=np.int)
    nelx = xpoints.size - 1
    for lmn in range(nel):
        ii = lmn + row
        conn[lmn, :] = [ii, ii + 1, ii + nelx + 2, ii + nelx + 1]
        if (lmn + 1) % (nelx) == 0:
            row += 1
    print conn

    print "\n\n============================ 3D ==============================="
    print "COORDS"
    print "\nCONNECTIVITY:"


def gen_coords_conn_from_inline(etype, mins, blocks):
    """Generate nodal coordinates and element connectivity from inline
    specification

    Parameters
    ----------
    mins : ndarray
        List of minimums
    blocks : tuple of ndarray
        blocks[i] -> [[length, interval]_0, [length, interval]_1, ...]

    Returns
    -------
    dim : int
        Dimension
    coords : ndarray
        Nodal coordinates
    connect : ndarray
        Nodal connectivity

    """
    # Initialize local variables
    lohi = [(None, None)] * 3

    if "bar" in etype.lower(): etype = "BAR"
    elif "quad" in etype.lower(): etype = "QUAD"
    elif "hex" in etype.lower(): etype = "HEX"
    else: raise UserInputError("{0}: unrecognized inline mesh type".format(etype))

    dim = {"BAR": 1, "QUAD": 2, "HEX": 3}[etype.upper()]

    points = []
    for i in range(dim):

        if not blocks[i]:
            raise UserInputError("No {0}Block specified for inline "
                                 "mesh".format(("X", "Y", "Z")[i]))

        # Generate the 1d bar mesh in this coordinate
        line_points, line_conn = line_mesh(mins[i], blocks[i])
        lohi[i] = (line_points[0], line_points[-1])
        points.append(line_points)

        continue

    if dim == 1:
        # Mesh already created
        coords = np.array(points[0])
        connect = np.array(line_conn)

    else:
        coords, connect = brick_mesh(*points)

    return dim, coords, connect
