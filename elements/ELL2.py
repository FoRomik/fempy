import numpy as np

from utilities.constants import TOOR3
from elements.element import Element
from elements.shapefunction import ShapeFunction

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


class SHL2(ShapeFunction):
    dim, nnodes = 1, 2
    ngauss = 2
    nmass = 2
    cornernodes = np.array([0, 1])
    facenodes = np.array([[]])
    gauss_coords = np.array([[-1], [1]], dtype=np.float64) * TOOR3
    gauss_weights = np.array([1, 1], dtype=np.float64)
    _topomap = {"ILO": 0, "IHI": 1}
    boundary = None

    def calc_shape(self, lcoord):
        x = lcoord[0]
        f = np.array([(1. - x), (1. + x)])
        return f / 2.

    def calc_shape_deriv(self, lcoord):
        df = np.array([[1., -1.]])
        return df / 2.


class ELL2(Element, SHL2):
    """Linear bar element"""
    name = "ELL2"
    eid = 1
    ncoord = 1
    ndof = 1
    def __init__(self, n, nodes, coords, material, *args, **kwargs):
        super(ELL2, self).__init__(n, nodes, coords, material, *args, **kwargs)

    @classmethod
    def volume(self, coords=None):
        # update the volume
        area = 1.
        length = 1.
        return area * length
