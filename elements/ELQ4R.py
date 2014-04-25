import numpy as np
from elements.shapefunction import ShapeFunction
from elements.ELB2 import ELB2


class ShapeFunction_Quad4R(ShapeFunction):
    """Element function for linear element defined in [1]

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

    name = "Q4R"
    dim, nnodes = 2, 4
    ngauss = 1
    f = np.zeros((ngauss))
    df = np.zeros((dim, ngauss))
    cornernodes = np.array([0, 1, 2, 3])
    facenodes = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    gauss_coords = np.array([[0, 0]], dtype=np.float64)
    gauss_weights = np.array([4], dtype=np.float64)
    _topomap = {"ILO": 3, "IHI": 1, "JLO": 0, "JHI": 2}
    bndry = ELB2()

    def calc_shape(self, lcoord):
        f = np.zeros((4))
        x, y = lcoord[:2]
        f[0] = (1. - x) * (1. - y)
        f[1] = (1. + x) * (1. - y)
        f[2] = (1. + x) * (1. + y)
        f[3] = (1. - x) * (1. + y)
        return f / 4.

    def calc_shape_deriv(self, lcoord):
        df = np.zeros((self.dim, 4))
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
