import numpy as np

from materials._material import Material
from utilities.tensor import I6, trace, dev, reduce_map


class ViscoPlastic(Material):
    name = "viscoplastic"
    mid = 2
    def __init__(self):
        """Instantiate the ViscoPlastic material

        """
        super(ViscoPlastic, self).__init__()
        self.register_parameters({"Ei": 0, "Nui": 1, "Yi": 2, "E0i": 3,
                                  "Ni": 4, "Edot0i": 5, "Mi": 6})

    def setup(self, pdict):
        """Set up the ViscoPlastic material
        Parameters
        ----------
        pdict : dict
            Parameter dictionary

        """
        self.parse_input_parameters(pdict)
        self.check_params()

    def initialize_state(self, *args, **kwargs):
        nxtra = 1
        names = ["Equivalent plastic strain"]
        keys = ["EQPS"]
        self.register_variables(nxtra, names, keys)
        self.set_initial_state(np.zeros(nxtra))

    def check_params(self):
        """Check parameters and set defaults

        """
        if self._params[self.Ei] < 0.:
            raise LentoError("Young's modulus E must be > 0")
        if not -1 < self._params[self.Nui] < .5:
            raise LentoError("Poisson's ratio NU out of bounds")
        if self._params[self.Yi] < 0:
            raise LentoError("Yield strength Y must be > 0")
        if self._params[self.Yi] == 0.:
            self._params[self.Yi] = 1.e99
        if self._params[self.Mi] == 0.:
            self._params[self.Mi] = 1.
        return

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
            Extra variables at beginning of step

        Returns
        -------
        S : array_like
            Updated stress

        dep : float
            Updated equivalent plastic strain

        """

        dstrain = d * dt
        dep = self.deplas(dt, stress, xtra, dstrain)
        xtra[0] += dep

        # Bulk modulus, Youngs modulus and Poissons ratio
        E = self._params[self.Ei]
        nu = self._params[self.Nui]
        K = E / (3. * (1. - 2. * nu))


        devol = np.sum(dstrain[:3])
        p = trace(stress)

        # S is the deviatoric stress predictor
        S = stress - p / 3. * I6 + E / (1 + nu) * dev(dstrain)
        se = np.sqrt(1.5 * np.sum(S * S))

        if se > 0:
            beta = 1. - 1.5 * E * dep / ((1 + nu) * se)

        else:
            beta = 1.;

        # S now stores the full stress tensor
        S = beta * S + (p / 3. + K * devol) * I6

        return S, xtra

    def deplas(self, dt, stress, xtra, dstrain):
        """Compute plastic strain increment given strain increment
        Parameters
        ----------
        dt : float
            Time step
        stress : array_like
            Stress at beginning of step
        xtra : array_like
            Extra variables
        dstrain : array_like
            Strain increment

        Returns
        -------
        e : float
            Plastic strain

        """
        #  Material properties
        E = self._params[self.Ei]
        nu = self._params[self.Nui]
        Y = self._params[self.Yi]
        e0 = self._params[self.E0i]
        n = self._params[self.Ni]
        edot0 = self._params[self.Edot0i]
        m = self._params[self.Mi]

        devol = np.sum(dstrain[:3])
        p = np.sum(stress[:3])

        # deviatoric stress predictor
        S = -I6 * p / 3 + E / (1 + nu) * dev(dstrain)
        sequiv = np.sqrt(1.5 * np.sum(S * S))

        e = 10e-15
        err = Y
        tol = 10e-06 * Y

        if sequiv * edot0 == 0:
            e = 0.

        else:
            while err > tol:
                c = (1 + (xtra + e) / e0) ** (1 / n)
                c *= (e / (dt * edot0)) ** (1 / m)
                f = sequiv / Y - 1.5 * e * E / (Y * (1 + nu)) - c
                dfde = (-1.5 * E / (Y * (1 + nu))
                         - c * (1 / (n * (xtra +e + e0)) + 1 / (m * e)))
                enew = e - f / dfde
                if enew < 0.:
                    # e must be > 0, so if new approx to e < 0 the solution
                    # must lie between current approx to e and zero.
                    e = e / 10.
                else:
                    e = enew

                err = np.abs(f)
                continue

        return e

    def stiffness(self, dt, d, stress, xtra):
        """Compute the material stiffness tensor

        Parameters
        ----------
        dt : float
            time step

        d : array_like
            Deformation rate

        stress : array_like
            Stress at beginning of step

        xtra : float
            Extra variables

        Returns
        -------
        C : array_like
            The material stiffness

        Notes
        -----
        Currently coded either for plane strain or general 3D. Note that in
        this procedure `stress' is the current estimate for stress at the end
        of the increment S_n+1

        """
        dstrain = d * dt
        eplas = xtra[0]
        E = self._params[self.Ei]
        nu = self._params[self.Nui]
        K = E / (3. * (1. - 2. * nu))
        G = 3. * K * E / (9. * K - E)
        e0 = self._params[self.E0i]
        n = self._params[self.Ni]
        m = self._params[self.Mi]

        devol = trace(dstrain)
        p = trace(stress)

        S = dev(stress)
        se = np.sqrt(1.5 * np.sum(S * S))

        # tjfulle: fix model
        dep = 0.

        if se * dep > 0:
            beta = 1. / (1. + 1.5 * E * dep / ((1. + nu) * se))
            gamma = beta * (1.5 * E / ((1 + nu) * se)
                            + (1 / (n * (e0 + eplas + dep)) + 1. / (m * dep)))
            factor = 1.5 * 1.5 * E * (dep - 1. / gamma) / ((1. + nu) * se ** 3)
        else:
            beta = 1.
            factor = 0.

        C = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        ik = reduce_map((i, k), 3)
                        jl = reduce_map((j, l), 3)
                        jk = reduce_map((j, k), 3)
                        il = reduce_map((i, l), 3)
                        ij = reduce_map((i, j), 3)
                        kl = reduce_map((k, l), 3)
                        w = ((I6[ik] * I6[jl] + I6[jk] * I6[il]) / 2.,
                             -I6[ij] * I6[kl] / 3.,
                             factor * S[ij] * S[kl])
                        c = (beta  *  E / (1 + nu) * (w[0] + w[1] + w[2]),
                             K * I6[ij] * I6[kl])
                        C[ij, kl] = c[0] + c[1]
                        continue
                    continue
                continue
            continue

        return C
