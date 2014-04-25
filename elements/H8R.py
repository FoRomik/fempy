import numpy as np

import src.base.consts as consts
from src.fem.shapefunctions.prototype import ShapeFunctionPrototype
from src.fem.shapefunctions.Q4 import ShapeFunction_Quad4


class ShapeFunction_Hex8R(ShapeFunctionPrototype):
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
    name = "Hex8R"
    dim, nnodes = 3, 8
    ngauss = 1
    cornernodes = [0, 1, 2, 3, 4, 5, 6, 7]
    facenodes = np.array([[0, 1, 5, 4],
                          [1, 2, 6, 5],
                          [3, 7, 6, 2],
                          [0, 4, 7, 3],
                          [0, 3, 2, 1],
                          [4, 5, 6, 7]])
    gauss_coords = np.array([[0, 0, 0]], dtype=np.float64)
    gauss_weights = np.array([8.], dtype=np.float64)
    _topomap = {"ILO": 3, "IHI": 1, "JLO": 0, "JHI": 2, "KLO": 4, "KHI": 5}
    bndry = ShapeFunction_Quad4()

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


if __name__ == "__main__":
    shape = ShapeFunction_Quad4()
    print shape.nface_nodes(2)
    print shape.face_nodes(2)
    print shape.next_pattern(np.array([1., 1.]))
