import numpy as np
import src.base.consts as consts
from src.fem.shapefunctions.prototype import ShapeFunctionPrototype


class ShapeFunction_Bar2R(ShapeFunctionPrototype):
    """Element function for linear element defined in [1]

    Notes
    -----
    Node and element face numbering

            0-------1
               [0]

    References
    ----------
    1. Taylor & Hughes (1981), p.49

    """

    name = "Bar2R"
    dim, nnodes = 1, 2
    ngauss = 1
    cornernodes = np.array([0, 1])
    facenodes = np.array([[]])
    gauss_coords = np.array([[0]], dtype=np.float64)
    gauss_weights = np.array([2], dtype=np.float64)
    _topomap = {"ILO": 0, "IHI": 1}

    def calc_shape(self, lcoord):
        x = lcoord[0]
        return np.array([.5 * (1. - x), .5 * (1. + x)])

    def calc_shape_deriv(self, lcoord):
        return np.array([[.5, -.5]])
