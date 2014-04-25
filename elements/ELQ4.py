import numpy as np

from elements.ELL2 import SHL2
from elements.element import Element
from elements.shapefunction import ShapeFunction
from utilities.constants import TOOR3

"""Element class for linear element defined in [1]

   Notes
   -----
   Node and element face numbering

               [2]
            3-------2
            |       |
       [3]  |       | [1]
            |       |
            0-------1
               [0]

   References
   ----------
   1. Taylor & Hughes (1981), p.49

"""


class SHQ4(ShapeFunction):
    """Shape function"""
    dim, nnodes = 2, 4
    ngauss = 4
    f = np.zeros((ngauss))
    df = np.zeros((dim, ngauss))
    cornernodes = np.array([0, 1, 2, 3])
    facenodes = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    gauss_coords = np.array([[-1, -1], [ 1, -1], [1,  1], [-1,  1]],
                            dtype=np.float64) * TOOR3
    gauss_weights = np.array([1, 1, 1, 1], dtype=np.float64)
    _topomap = {"ILO": 3, "IHI": 1, "JLO": 0, "JHI": 2}
    boundary = SHL2()

    def calc_shape(self, lcoord):
        f = np.zeros((self.ngauss))
        x, y = lcoord[:2]
        f[0] = (1. - x) * (1. - y)
        f[1] = (1. + x) * (1. - y)
        f[2] = (1. + x) * (1. + y)
        f[3] = (1. - x) * (1. + y)
        return f / 4.

    def calc_shape_deriv(self, lcoord):
        df = np.zeros((self.dim, self.ngauss))
        x, y = lcoord[:2]
        df[0, 0] = -(1. - y)
        df[0, 1] =  (1. - y)
        df[0, 2] =  (1. + y)
        df[0, 3] = -(1. + y)

        df[1, 0] = -(1. - x)
        df[1, 1] = -(1. + x)
        df[1, 2] =  (1. + x)
        df[1, 3] =  (1. - x)
        return df / 4.

    def next_pattern(self, lcoord):
        x, y = lcoord / np.amax(np.abs(lcoord)) * 1.01
        if x >  1:
            return [1, 2]
        elif x < -1:
            return [3, 0]
        elif y >  1:
            return [2, 3]
        elif y < -1:
            return [0, 1]
        else:
            return None


class ELQ4(Element, SHQ4):
    """Element class"""
    name = "ELQ4"
    elem_type = "QUAD"
    eid = 2
    ncoord = 2
    ndof = 2
    hasplanestress = True
    def __init__(self, n, nodes, coords, material, *args, **kwargs):
        super(ELQ4, self).__init__(n, nodes, coords, material, *args, **kwargs)

    @classmethod
    def volume(self, coords):
        # update the volume
        x, y = coords[:, [0, 1]].T
        depth = 1.
        area = .5 * ((x[0] - x[2]) * (y[1] - y[3]) - (x[1] * x[3]) * (y[0] - y[2]))
        return area * depth
