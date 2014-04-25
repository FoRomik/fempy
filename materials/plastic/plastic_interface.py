import numpy as np

from src.fem.materials._material import Material
import src.fem.materials.plastic.plastic as plastic
import src.base.consts as consts
from src.base.tensor import iso, dev
from src.base.errors import WasatchError

class Plastic(Material):
    name = "plastic"
    mid = 3
    def __init__(self):
        """Instantiate the Plastic material

        """
        super(Plastic, self).__init__()
        self.register_parameters({"K": 0, "MU": 1, "A1": 2, "A4": 3, "DEVEL": 4})

    def setup(self, pdict):
        """Set up the Elastic material

        Parameters
        ----------
        pdict : dict
            Parameter dictionary

        """
        self.parse_input_parameters(pdict)
        plastic.plastic_check(self._params)

    def initialize_state(self, *args, **kwargs):
        nxtra, namea, keya, xtra = plastic.plastic_request_xtra(self._params)
        self.register_variables(nxtra, namea, keya, mig=True)
        self.xtra = xtra
        pass

    def update_state(self, dt, d, stress, xtra):
        """Compute updated stress given strain increment

        Parameters
        ----------
        dt : float
            Time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        xtra : array_like
            Extra variables

        Returns
        -------
        S : array_like
            Updated stress

        xtra : array_like
            Updated extra variables

        """
        plastic.plastic_update_state(dt, self._params, d, stress, xtra)
        return stress, xtra

    def stiffness(self, dt, d, stress, xtra):
        """Return the constant stiffness
        dt : float
            time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        """
        J = plastic.plastic_stiff(dt, self._params, d, stress, xtra)
        return J
