import numpy as np
from elements.ELL2 import SHL2
from elements.element import Element
from elements.shapefunction import ShapeFunction

"""Element function for linear CST element defined in [1]

   Notes
   -----
   Node and element face numbering


            2
            |\
       [2]  |  \  [1]
            |    \
            0-----1
              [0]

   References
   ----------
   1. Taylor & Hughes (1981), p.49

"""


class SHT3(ShapeFunction):
    dim, nnodes = 2, 3
    ngauss = 1
    cornernodes = np.array([0, 1, 2])
    facenodes = np.array([[0, 1], [1, 2], [2, 0]])
    gauss_coords = np.array([[1., 1]], dtype=np.float64) / 3.
    gauss_weights = np.array([.5], dtype=np.float64)
    _topomap = {"ILO": 2, "IHI": 1, "JLO": 0, "JHI": 2}
    bndry = SHL2()

    """
    mass stuff
    if False:
        # mass stuff for explicit
        nmass = 4
        # mass gauss coords
        return np.array([[1. / 3., 1. / 3.],
                         [.6, .2],
                         [.2, .6],
                         [.2, .2]], dtype=np.float64)
        # mass weights
        return np.array([-27, 25, 25, 25], dtype=np.float64) / 96.
    """

    def calc_shape(self, lcoord):
        x, y = lcoord[:2]
        return np.array([x, y, 1. - x - y])

    def calc_shape_deriv(self, lcoord):
        return np.array([[1, 0, -1],
                         [0, 1, -1]], dtype=np.float64)


class ELT3(Element, SHT3):
    """Constant strain triangle"""
    name = "ELT3"
    elem_type = "TRI"
    eid = 4
    nnodes = 3
    ncoord = 2
    ndof = 2
    def __init__(self, n, nodes, coords, material, *args, **kwargs):
        super(ELT3, self).__init__(n, nodes, coords, material, *args, **kwargs)

    @classmethod
    def volume(self, coords=None):
        # update the volume
        depth = 1.
        area = 1.
        return area * depth


