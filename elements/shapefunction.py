import numpy as np

class ShapeFunction(object):
    """Defines the prototype of a interpolation function cornernodes defines
    the nodes at the geometrical corner
    """
    dim, nnodes = 0, 0
    ngauss = 0
    nfacenodes = None
    facenodes = None
    cornernodes = []
    _topomap = {}
    gauss_coords = None
    gauss_weights = None
    gauss_shape = None
    gauss_shape_inv = None
    reduced = None

    def __init__(self):
        self.f  = np.zeros(self.nnodes)
        self.df = np.zeros((self.dim, self.nnodes))

    @classmethod
    def topomap(cls, dom):
        return cls._topomap.get(dom.upper())

    @classmethod
    def face_from_nodes(cls, fnodes):
        for face, nodes in enumerate(cls.facenodes):
            if np.all([x in nodes for x in fnodes]):
                return face
        else:
            return None

    def calc_shape(self, lcoord):
        return None

    def calc_shape_deriv(self, lcoord):
        return None

    def eval_shape(self, lcoord):
        self.calc_shape(lcoord)
        return self.f

    def eval_shape_deriv(self, lcoord):
        self.calc_shape_deriv(lcoord)
        return self.df

    def next_pattern(self, lcoord):
        return [0]

    def calc_gauss(self):
        """Calculate the inverse of the shape functions at the Gauss points"""
        a = []
        for lc in self.gauss_coords:
            a.append(np.array(self.calc_shape(lc)))
        self.gauss_shape = np.take(np.array(a), self.cornernodes, 1)
        self.gauss_shape_inv = np.linalg.inverse(self.gauss_shape)

    def nface_nodes(self, face=None):
        if face is None:
            return self.facenodes.shape[1]
        return self.face_nodes(face).size

    def face_nodes(self, face):
        return self.facenodes[face]
