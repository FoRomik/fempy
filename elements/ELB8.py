import numpy as np

from utilities.constants import TOOR3
from elements.ELQ4 import SHQ4
from elements.element import Element
from elements.shapefunction import ShapeFunction


"""Element function for linear element defined in [1]

   Notes
   -----
   Integration points in parentheses

          7---------------6
         /|(7)        (6)/|
        / |             / |
       /  |            /  |
      /   |           /   |
     / (4)|       (5)/    |
    4---------------5     |
    |     |         |     |
    |     3---------|-----2
    |    / (3)      | (2)/
    |   /           |   /
    |  /            |  /
    | /             | /
    |/ (0)       (1)|/
    0---------------1

       7-------6
      /|      /|
     / |     / |
    4-------5  |
    |  3----|--2
    | /     | /
    |/      |/
    0-------1

   References
   ----------
   1. Taylor & Hughes (1981), p.49

"""


class SHB8(ShapeFunction):
    dim, nnodes = 3, 8
    ngauss = 8
    cornernodes = [0, 1, 2, 3, 4, 5, 6, 7]
    facenodes = np.array([[0, 1, 5, 4],
                          [1, 2, 6, 5],
                          [3, 7, 6, 2],
                          [0, 4, 7, 3],
                          [0, 3, 2, 1],
                          [4, 5, 6, 7]])
    gauss_coords = np.array([[-1., -1., -1.],
                             [ 1., -1., -1.],
                             [-1.,  1., -1.],
                             [ 1.,  1., -1.],
                             [-1., -1.,  1.],
                             [ 1., -1.,  1.],
                             [-1.,  1.,  1.],
                             [ 1.,  1.,  1.]]) * TOOR3
    gauss_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    _topomap = {"ILO": 3, "IHI": 1, "JLO": 0, "JHI": 2, "KLO": 4, "KHI": 5}
    boundary = SHQ4()

    def calc_shape(self, lcoord):
        f = np.zeros((self.ngauss))
        x, y, z = lcoord[:3]
        f[0] = (1. - x) * (1. - y) * (1. - z)
        f[1] = (1. + x) * (1. - y) * (1. - z)
        f[2] = (1. + x) * (1. + y) * (1. - z)
        f[3] = (1. - x) * (1. + y) * (1. - z)
        f[4] = (1. - x) * (1. - y) * (1. + z)
        f[5] = (1. + x) * (1. - y) * (1. + z)
        f[6] = (1. + x) * (1. + y) * (1. + z)
        f[7] = (1. - x) * (1. + y) * (1. + z)
        return f / 8.

    def calc_shape_deriv(self, lcoord):
        df = np.zeros((self.dim, self.ngauss))
        x, y, z = lcoord[:3]

        df[0, 0] = -(1. - y) * (1. - z)
        df[0, 1] =  (1. - y) * (1. - z)
        df[0, 2] =  (1. + y) * (1. - z)
        df[0, 3] = -(1. + y) * (1. - z)
        df[0, 4] = -(1. - y) * (1. + z)
        df[0, 5] =  (1. - y) * (1. + z)
        df[0, 6] =  (1. + y) * (1. + z)
        df[0, 7] = -(1. + y) * (1. + z)

        df[1, 0] = -(1. - x) * (1. - z)
        df[1, 1] = -(1. + x) * (1. - z)
        df[1, 2] =  (1. + x) * (1. - z)
        df[1, 3] =  (1. - x) * (1. - z)
        df[1, 4] = -(1. - x) * (1. + z)
        df[1, 5] = -(1. + x) * (1. + z)
        df[1, 6] =  (1. + x) * (1. + z)
        df[1, 7] =  (1. - x) * (1. + z)

        df[2, 0] = -(1. - x) * (1. - y)
        df[2, 1] = -(1. + x) * (1. - y)
        df[2, 2] = -(1. + x) * (1. + y)
        df[2, 3] = -(1. - x) * (1. + y)
        df[2, 4] =  (1. - x) * (1. - y)
        df[2, 5] =  (1. + x) * (1. - y)
        df[2, 6] =  (1. + x) * (1. + y)
        df[2, 7] =  (1. - x) * (1. + y)
        return df / 8.

    def next_pattern(self, lcoord):
        x, y, z = lcoord[:3] / np.amax(np.abs(lcoord)) * 1.01
        if x >  1:
            return [1, 2, 6, 5]
        elif x < -1:
            return [0, 4, 7, 3]
        elif y >  1:
            return [2, 3, 7, 6]
        elif y < -1:
            return [0, 1, 5, 4]
        elif z >  1:
            return [4, 5, 6, 7]
        elif z < -1:
            return [0, 3, 2, 1]
        else:
            return None

"""
mass stuff for explicit
gauss_coords:
    nmass = 27
        if mode == "mass":
            x1D = np.array([-0.7745966692, 0., 0.7745966692], dtype=np.float64)
            xi = np.zeros((self.dim, self.ngauss))
            for k in range(3):
                for j in range(3):
                    for i in range(3):
                        n = 9 * k + 3 * j + i
                        xi[n, 0] = x1D[i]
                        xi[n, 1] = x1D[j]
                        xi[n, 2] = x1D[k]
            return xi

    def gauss_weights(self, mode=None):
        if mode == "mass":
            w = np.zeros(self.ngauss)
            w1D = np.array([0.555555555, 0.888888888, 0.55555555555])
            for k in range(3):
                for j in range(3):
                    for i in range(3):
                        n = 9 * k + 3 * j + i
                        w[n] = w1D[i] * w1D[j] * w1D[k]
            return w
"""


class ELB8(Element, SHB8):
    """Trilinear hex element"""
    name = "ELB8"
    eid = 3
    ncoord = 3
    ndof = 3
    def __init__(self, n, nodes, coords, material, *args, **kwargs):
        super(ELB8, self).__init__(n, nodes, coords, material, *args, **kwargs)

    @classmethod
    def volume(self, coords):
        # update the volume
        volume = 1.
        return volume


if __name__ == "__main__":
    shape = SHQ4()
    print shape.nface_nodes(2)
    print shape.face_nodes(2)
    print shape.next_pattern(np.array([1., 1.]))
