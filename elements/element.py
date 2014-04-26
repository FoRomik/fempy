import sys
import numpy as np
from scipy.weave import inline, converters

import runopts as ro
from variables.varinc import *
from model.model_data import eb_data
from utilities.tensor import asarray, asmatrix, reduce_map, reduce_map_C
from utilities.tensor import axialv, axialt
from utilities.tensor import NSYMM, NTENS, I3x3, I6, I9
from utilities.errors import UserInputError, WasatchError

# Element data is stored in an array with the following shape:
#     ELDAT = (2, NGAUSS + 1, NDAT)
# The element data array contains the following data:
#     ELDAT[i] -> data at beginning (i=0) of step and current (i=1)
#     ELDAT[:, j:ngauss] -> data for jth gauss point
#     ELDAT[:, j+ngauss] -> element average value
#     ELDAT[:, :, k]     -> kth data point for element, according to:
#       k =  0:5   -> Stress
#       k =  6:11  -> Left stretch
#       k = 12:20  -> Rotation
#       k = 21:26* -> Symmetric part of velocity gradient
#       k = 27:29* -> Skew symmetric part of velocity gradient (vorticity)
#       k = 30:35* -> Strain
#       k = 36:    -> Material extra variables
# * Variables are stored only for plotting.
STRESS = 0
LEFTV = 6
ROTATE = 12
LSYMM = 21
LSKEW = 27
STRAIN = 30
XTRA = 36
DATMAP = {"STRESS": (STRESS, STRESS + NSYMM),
          "LEFTV": (LEFTV, LEFTV + NSYMM),
          "ROTATE": (ROTATE, ROTATE + NTENS),
          "LSYMM": (LSYMM, LSYMM + NSYMM),
          "LSKEW": (LSKEW, LSKEW + 3),
          "STRAIN": (STRAIN, STRAIN + NSYMM),
          "XTRA": (XTRA, -1)}


class Element(object):

    name = None
    planestress = False
    reducedint = False
    canreduce = False
    hasplanestress = False
    rho = 1.
    lumped_mass = 1.

    def __init__(self, elid, elem_blk_ids, nodes, coords, 
                 material, mat_params, perturbed, *args, **kwargs):
        if len(nodes) != self.nnodes:
            raise UserInputError(
                "{0}: {1} nodes required, got {2}".format(
                    self.name, self.nnodes, len(nodes)))

        # set up the model, perturbing parameters if necessary
        self.material = material
        for key, (i, val) in perturbed.items():
            mat_params[key] = val
        self.material.setup(mat_params)
        self.material.initialize_state()

        self._variables = []
        self.reducedint = "REDUCED_INTEGRATION" in kwargs
        self.planestress = "PLANE_STRESS" in kwargs
        if self.reducedint and not self.canreduce:
            raise UserInputError("Element {0} cannot use reduced integration"
                                 .format(self.name))
        if self.planestress and not self.hasplanestress:
            raise UserInputError("Element {0} not plane stress compatible"
                                 .format(self.name))
        ro.reducedint = self.reducedint
        self.elid = elid
        self.elid_this_blk, self.elem_blk_id = elem_blk_ids
        self.ievar = 0

        self.register_variable("CAUCHY-STRESS", vtype="SYMTENS")
        self.register_variable("LEFT-STRETCH", vtype="SYMTENS")
        self.register_variable("ROTATION", vtype="TENS")
        self.register_variable("SYMM-L", vtype="SYMTENS")
        self.register_variable("SKEW-L", vtype="SKEWTENS")
        self.register_variable("GREEN-STRAIN", vtype="SYMTENS")

        for var in self.material.variables():
            self.register_variable(var, vtype="SCALAR")

        # Element data array.  See comments above.
        self.nxtra = self.material.nxtra
        self.ndat = XTRA + self.nxtra + len(perturbed)
        self._p = XTRA + self.nxtra
        self.data = np.zeros((2, self.ngauss + 1, self.ndat))

        # register perturbed parameters as variables
        self._pidx = []
        for i, (key, (idx, val)) in enumerate(perturbed.items()):
            self._pidx.append(idx)
            self.register_variable("{0}_STAT".format(key))
            self.data[:, :, self._p + i] = val

        # Element volume
        self._volume = self.volume(coords)

        # initialize nonzero data
        self.data[:, :, LEFTV:LEFTV+NSYMM] = I6
        self.data[:, :, ROTATE:ROTATE+NTENS] = I9
        self.data[:, :, XTRA:XTRA+self.nxtra] = self.material.initial_state()

        # Initialize the stiffness
        self.kel = np.zeros((self.ndof * self.nnodes, self.ndof * self.nnodes))

        pass

    @classmethod
    def volume(self, coords):
        raise WasatchError("Element {0} must define volume"
                           .format(self.name))

    def register_variable(self, var, vtype="SCALAR"):
        """Register element variable

        """
        vtype = vtype.upper()
        var = var.upper()
        if vtype == "SCALAR":
            self._variables.append(var)

        elif vtype == "TENS":
            self._variables.extend(["{0}-{1}".format(var, x)
                                    for x in ("XX", "XY", "XZ",
                                              "YX", "YY", "YZ",
                                              "ZX", "ZY", "ZZ")])
        elif vtype == "SYMTENS":
            self._variables.extend(["{0}-{1}".format(var, x)
                                    for x in ("XX", "YY", "ZZ", "XY", "YZ", "XZ")])

        elif vtype == "SKEWTENS":
            self._variables.extend(["{0}-{1}".format(var, x)
                                    for x in ("XY", "YZ", "XZ")])

        else:
            raise WasatchError("{0}: unrecognized vtype".format(vtype))


    def register_elem_data(self):
        ebid = self.elem_blk_id
        eb_data[ebid].register_variable("CAUCHY-STRESS", VAR_SYMTENSOR)
        eb_data[ebid].register_variable("LEFT-STRETCH", VAR_SYMTENSOR,
                                        initial_value=I6)
        eb_data[ebid].register_variable("ROTATION", VAR_TENSOR, initial_value=I9)
        eb_data[ebid].register_variable("SYMM-L", VAR_SYMTENSOR)
        eb_data[ebid].register_variable("SKEW-L", VAR_SKEWTENSOR)
        eb_data[ebid].register_variable("GREEN-STRAIN", VAR_SYMTENSOR)
        xinit = self.material.initial_state()
        for (i, var) in enumerate(self.material.variables()):
            eb_data[ebid].register_variable(var, VAR_SCALAR, initial_val=xinit[i])


    def variables(self):
        """Returns a list of the names of the plotable variables.

        Returns
        -------
        variables : array_like
            Plotable data names

        """
        return self._variables

    def element_data(self, ave=False, exo=False, item=None):
        """Return the current element data

        Returns
        -------
        data : array_like
            Element data

        """
        a, b = 0, self.ndat
        if item is not None:
            # Get the index of item
            try:
                a, b = DATMAP.get(item.upper())
            except TypeError:
                raise WasatchError("{0}: not in element data".format(item))
            if b == -1:
                b = self.ndat
        if ave or exo:
            return self.data[0, self.ngauss][a:b]
        return self.data[0, :, a:b]

    def plotable_variables(self):
        """Return the list of plotable variables for the element

        Returns
        -------
        plotable_variables : array_like
            List of names of plotable variables

        """
        return self.variables


    def stiffness(self, dt, coords, du):
        """Assemble the element stiffness

        Parameters
        ----------
        dt : float
            Time step

        coords : array like, (i, j,)
            Nodal coordinates
            coords[i, j] -> jth coordinate of ith node

        du : array_like
            Displacement increment vector
            du[i, a] -> ath component of displacement increment at ith node

        Returns
        -------
        kel : array_like
            The element stiffness

        """
        # Set up integration points and weights
        xilist = self.gauss_coords
        w = self.gauss_weights

        self.kel[:] = 0.

        #  Loop over the integration points
        for intpt in range(self.ngauss):

            # Compute shape functions and derivatives wrt local coords
            xi = xilist[intpt]
            N = self.calc_shape(xi)
            dNdxi = self.calc_shape_deriv(xi)

            # Compute the Jacobian matrix J = dNdxi.x
            dxdxi = np.dot(dNdxi, coords)
            dtm = np.linalg.det(dxdxi)

            # Convert shape function derivatives to derivatives wrt global coords
            dxidx = np.linalg.inv(dxdxi)
            dNdx = np.dot(dxidx, dNdxi)

            D = self.material.stiffness(dt,
                self.data[1, intpt, LSYMM:LSYMM+NSYMM],
                self.data[1, intpt, STRESS:STRESS+NSYMM],
                self.data[1, intpt, XTRA:XTRA+self.nxtra])

            if self.ncoord == 2:
                # Modify the stiffness for 2D according to:
                # 1) Plane strain: Remove rows and columns of the stiffness
                #    corresponding to the plane of zero strain
                # 2) Plane stress: Invert the stiffness and remove the rows
                #    and columns of the compliance corresponding the plane of
                #    zero stress.
                idx = [[[0], [1], [3]], [0, 1, 3]]

                if self.planestress:
                    # Invert the stiffness to get the compliance
                    D = np.linalg.inv(np.linalg.inv(D)[idx])

                else:
                    D = D[idx]

            self.add_to_kel(self.kel, dNdx, D, float(w[intpt]), float(dtm))

            continue # intpt

        if self.reducedint:
            self.correct_stiff(dt, coords, du, self.kel, mode=1)

        return self.kel

    def add_to_kel(self, kel, dNdx, D, w, dtm, mode=0):
        """Put the stiffness at the quadrature point in to the element
        stiffness.

        Parameters
        ----------
        kel : array_like
            Current element stiffness

        dNdx : array_like
            Shape functions

        D : array_like
            Material stiffness at Gauss point.  Stored as 3x3 or 2x2 matrix

        w : float
            Gauss integration weight

        dtm : float
            Jacobian

        mode : int
            Mode flag
            mode == 0 -> Assemble stiffness regularly, subtracting off some
                         deviatoric contributions if reduced integration is
                         used
            mode == 1 -> Add bulk contribution back to the element
        Returns
        -------
        None

        Notes
        -----
        kel is changed in place

        """
        # use only two space since there are so many loops
        nnodes, ndof, ncoord = self.nnodes, self.ndof, self.ncoord
        reducedint = 0 if not self.reducedint else 1

        if not ro.ENABLE_WEAVE:
            for A in range(nnodes):
                for i in range(ndof):
                    for j in range(ncoord):
                        for k in range(ncoord):
                            for l in range(ndof):
                                for B in range(nnodes):
                                    rw = A * ndof + j
                                    cl = B * ndof + k
                                    I = reduce_map(i, j, ncoord)
                                    L = reduce_map(k, l, ncoord)
                                    JJ = reduce_map(j, j, ncoord)
                                    if mode == 0:
                                        kAB = dNdx[i, A] * D[I, L] * dNdx[l, B]
                                        kR = 0.
                                        if reducedint == 1:
                                            kR = -dNdx[i, A] * D[JJ, L] * dNdx[l, B]

                                    else:
                                       # Adding back in some deviatoric contribution
                                       kAB = 0.
                                       kR = dNdx[i, A] * D[JJ, L] * dNdx[l, B]

                                    # add the contribution to the stiffness
                                    kC = (kAB + kR / ncoord) * w * dtm
                                    kel[rw, cl] = kel[rw, cl] + kC

        else:
            code = """
              int rw, cl, I, L, JJ;
              double kAB, kR;
              for (int A=0; A < nnodes; ++A) {
                for (int i=0; i < ndof; ++i) {
                  for (int j=0; j < ncoord; ++j) {
                    for (int k=0; k < ncoord; ++k) {
                      for (int l=0; l < ndof; ++l) {
                        for (int B=0; B < nnodes; ++B) {
                          rw = A * ndof + j;
                          cl = B * ndof + k;
                          I = reduce_map_C(i, j, ncoord);
                          L = reduce_map_C(k, l, ncoord);
                          JJ = reduce_map_C(j, j, ncoord);
                          if (mode == 0) {
                            kAB = dNdx(i, A) * D(I, L) * dNdx(l, B);
                            kR = 0.;
                            if (reducedint == 1) {
                              kR = -dNdx(i, A) * D(JJ, L) * dNdx(l, B);
                              }
                            }
                          else {
                            // Adding back in some deviatoric contribution
                            kAB = 0.;
                            kR = dNdx(i, A) * D(JJ, L) * dNdx(l, B);
                            }
                          // add the contribution to the stiffness
                          kel(rw, cl) = kel(rw, cl) + (kAB + kR / ncoord) * w * dtm;
                          }
                        }
                      }
                    }
                  }
                }
              """
            inline(code, ["nnodes", "ndof", "ncoord", "mode", "kel",
                          "D", "dNdx", "dtm", "w", "reducedint"],
                   support_code=reduce_map_C, type_converters=converters.blitz)

        return

    def von_neumann(self, time, coords, tractions):
        """Apply the Von Neumann (natural) BC to the element

        Von Neummann BC is applied as a surface load on element faces.

        Parameters
        ----------
        time : float
            Simulation time

        coords : array_like
            Current nodal coords

        tractions : array_like
            Tractions
            tractions[i] -> traction on coordinate i as a function of time

        Returns
        -------
        r : array_like
            Element distributed load vector

        """
        r = np.zeros((self.ndof * self.nface_nodes()))

        xilist = self.boundary.gauss_coords
        w = self.boundary.gauss_weights

        ndof, nfnodes = self.ndof, self.nface_nodes()
        for intpt in range(self.boundary.ngauss):

            # Compute shape functions and derivatives wrt local coords
            xi = xilist[intpt]
            N = self.boundary.calc_shape(xi)
            dNdxi = self.boundary.calc_shape_deriv(xi)

            # Compute the Jacobian matrix J = dNdxi.x
            dxdxi = np.dot(dNdxi, coords)

            if self.ncoord == 2:
                dtm = np.sqrt(dxdxi[0, 0] ** 2 + dxdxi[0, 1] ** 2)

            elif self.ncoord == 3:
                a = (dxdxi[0, 1] * dxdxi[1, 2]) - (dxdxi[1, 1] * dxdxi[0, 2])
                b = (dxdxi[0, 0] * dxdxi[1, 2]) - (dxdxi[1, 0] * dxdxi[0, 2])
                c = (dxdxi[0, 0] * dxdxi[1, 1]) - (dxdxi[1, 0] * dxdxi[0, 1])
                dtm = np.sqrt(a ** 2 + b ** 2 + c ** 2)

            for a in range(nfnodes):
                for i in range(ndof):
                    row = self.ndof * a + i
                    traction = N[a] * tractions[i](time)
                    r[row] += traction * w[intpt] * dtm
                    continue
                continue

            continue

        return r

    def residual(self, dt, coords, du):
        """Assemble the element residual force

        Parameters
        ----------
        dt : float
            Time step

        coords : array_like
            Nodal coordinates
            coords[i, a] -> ith coord of ath node

        materialprops : array_like
            Material properties passed on to constitutive procedures

        du : array_like
            Displacement increment vector
            du[i, a] -> ath component of displacement increment at ith node

        Returns
        -------
        rel : array_like
            Element residual

        """
        # output array
        rel = np.zeros((self.ndof * self.nnodes))

        # Set up integration points and weights
        xilist = self.gauss_coords
        w = self.gauss_weights

        # Loop over the integration points
        nnodes, ndof, ncoord = self.nnodes, self.ndof, self.ncoord

        for intpt in range(self.ngauss):

            # Compute shape functions and derivatives wrt local coords
            xi = xilist[intpt]
            N = self.calc_shape(xi)
            dNdxi = self.calc_shape_deriv(xi)

            # Compute the Jacobian matrix J = dNdxi.x
            dxdxi = np.dot(dNdxi, coords)
            dtm = float(np.linalg.det(dxdxi))

            # Convert shape function derivatives to derivatives wrt global coords
            dxidx = np.linalg.inv(dxdxi)
            dNdx = np.dot(dxidx, dNdxi)

            # Compute the element residual
            sig = self.data[1, intpt, STRESS:STRESS+NSYMM]
            if not ro.ENABLE_WEAVE:
                for a in range(nnodes):
                    for i in range(ndof):
                        row = ndof * a + i
                        for j in range(ncoord):
                            I = reduce_map(i, j, 3)
                            rel[row] += sig[I] * dNdx[j, a] * w[intpt] * dtm;

            else:
                code = """
                    int row, I;
                    for (int a=0; a < nnodes; ++a) {
                      for (int i=0; i < ndof; ++i) {
                        row = ndof * a + i;
                        for (int j=0; j < ncoord; ++j) {
                          I = reduce_map_C(i, j, 3);
                          rel(row) += sig(I) * dNdx(j, a) * w(intpt) * dtm;
                          }
                        }
                      }
                    """
                inline(code, ["nnodes", "ndof", "ncoord", "rel",
                              "dNdx", "dtm", "w", "sig", "intpt"],
                       support_code=reduce_map_C,
                       type_converters=converters.blitz)

            continue # intpt

        return rel

    def update_state(self, t, dt, coords, u, du):
        """Compute the stress and accumulated plastic strain at the end of a
        load increment at all integration points in an element

        Parameters
        ----------
        coords : array_like
            Nodal coordinates
            coords[i, a] -> ith coord of ath node

        du : array_like
            Displacement increment vector
            du[i, a] -> ath component of displacement increment at ith node

        """
        # Set up integration points and weights
        self._volume = self.volume(coords)
        xilist = self.gauss_coords
        w = self.gauss_weights

        # Loop over the integration points
        for intpt in range(self.ngauss):

            # Stretch and rotation from beginning of step
            V = asmatrix(self.data[0, intpt, LEFTV:LEFTV+NSYMM])
            R = asmatrix(self.data[0, intpt, ROTATE:ROTATE+NTENS])

            # Compute shape functions and derivatives wrt local coords
            xi = xilist[intpt]
            N = self.calc_shape(xi)
            dNdxi = self.calc_shape_deriv(xi)

            # Compute the Jacobian matrix J = dNdxi.x
            dxdxi = np.dot(dNdxi, coords)
            dtm = np.linalg.det(dxdxi)

            # Convert shape function derivatives to derivatives wrt global coords
            dxidx = np.linalg.inv(dxdxi)
            dNdx = np.dot(dxidx, dNdxi)

            # Compute the velocity gradient
            # L_ij = dv_i / dx_j = d(du_i/dt) / dx_j
            #      = du_iI dN_I / dx_j * 1 / dt
            L = np.zeros((3, 3))
            L[:self.ncoord, :self.ncoord] = np.dot(dNdx, du) / dt

            # symmetric and deviatoric parts -> needed for finite rotations
            D = .5 * (L + L.T)
            W = L - D

            z = -2 * axialv(np.dot(V, D))
            w = -2. * axialv(W)
            _w_ = w - 2. * np.dot(np.linalg.inv(V - np.trace(V) * I3x3), z)
            _W_ = -.5 * axialt(_w_)

            # Update the rotation
            A = I3x3 - _W_ * dt / 2
            RHS = np.dot(I3x3 + _W_ * dt / 2, R)

            # updated rotation
            R = np.dot(np.linalg.inv(A), RHS)

            # Rate of stretch
            Vdot = np.dot((D + W), V) - np.dot(V, _W_)
            V += Vdot * dt

            # Unrotate deformation rate
            d = np.dot(R.T, np.dot(D, R))

            # Unrotate Cauchy Stress
            T = asmatrix(self.data[0, intpt, STRESS:STRESS+NSYMM])
            sig = np.dot(R.T, np.dot(T, R))

            # Convert quantities to arrays that will be passed to material model
            d = asarray(d)
            sig = asarray(sig)

            # --- Compute the stress
            # before calling, replace the material parameters with the
            # perturbed values
            self.material._params[self._pidx] = self.data[0, intpt, self._p:]
            sig, xtra = self.material.update_state(
                dt, d, sig, self.data[0, intpt, XTRA:XTRA+self.nxtra])

            # Rotate stress to material frame
            T = np.dot(R, np.dot(asmatrix(sig), R.T))

            # Calculate strain
            F = np.dot(V, R)
            E = .5 * (np.dot(F.T, F) - I3x3)

            # save element data
            self.data[1, intpt, STRESS:STRESS+NSYMM] = asarray(T)
            self.data[1, intpt, LEFTV:LEFTV+NSYMM] = asarray(V)
            self.data[1, intpt, ROTATE:ROTATE+NTENS] = asarray(R, symm=False)
            self.data[1, intpt, LSYMM:LSYMM+NSYMM] = asarray(D)
            self.data[1, intpt, LSKEW:LSKEW+NSYMM] = asarray(W, skew=True)
            self.data[1, intpt, STRAIN:STRAIN+NSYMM] = asarray(E)
            self.data[1, intpt, XTRA:XTRA+self.nxtra] = xtra
            continue # intpt

        # now average over all Gauss points. This is not valid for all
        # elements, just a place holder
        self.data[1, self.ngauss] = np.average(self.data[1, :self.ngauss], axis=0)
        j, i = self.elid_this_blk, self.elem_blk_id
        eb_data[i].set_var("ELDAT", self.data[1, self.ngauss][:], j)
        return

    def advance_state(self):
        """Advance the element state

        """
        self.data[0] = self.data[1]

    def correct_stiff(self, dt, coords, du, kel, mode=1):
        """Add correction to element stiff for reduced integration

        """

        # Set up integration points and weights for the reduced integration
        xilist = self.reduced.gauss_coords
        w = self.reduced.gauss_weights

        #  Loop over the integration points
        for intpt in range(self.reduced.ngauss):

            # Compute shape functions and derivatives wrt local coords
            xi = xilist[intpt]
            N = self.reduced.calc_shape(xi)
            dNdxi = self.reduced.calc_shape_deriv(xi)

            # Compute the Jacobian matrix J = dNdxi.x
            dxdxi = np.dot(dNdxi, coords)
            dtm = np.linalg.det(dxdxi)

            # Convert shape function derivatives to derivatives wrt global coords
            dxidx = np.linalg.inv(dxdxi)
            dNdx = np.dot(dxidx, dNdxi)

            # Compute the material tangent stiffness (d stress/d dstrain)
            D = self.material.stiffness(dt,
                self.data[1, self.ngauss, LSYMM:LSYMM+NSYMM],
                self.data[1, self.ngauss, STRESS:STRESS+NSYMM],
                self.data[1, self.ngauss, XTRA:XTRA+self.nxtra])

            if self.ncoord == 2:
                # Modify the stiffness for 2D according to:
                # 1) Plane strain: Remove rows and columns of the stiffness
                #    corresponding to the plane of zero strain
                # 2) Plane stress: Invert the stiffness and remove the rows
                #    and columns of the compliance corresponding the plane of
                #    zero stress.
                idx = [[[0], [1], [3]], [0, 1, 3]]

                if self.planestress:
                    # Invert the stiffness to get the compliance
                    D = np.linalg.inv(np.linalg.inv(D)[idx])

                else:
                    D = D[idx]

            self.add_to_kel(kel, dNdx, D, w[intpt], dtm, mode=mode)

            continue # intpt

        return

    def mass(self, coords):
        """Assemble the element mass matrix

        """
        mel = np.zeros((self.ndof * self.nnodes, self.ndof * self.nnodes))

        # Set up integration points and weights
        xilist = self.mass_gauss_coords
        w = self.mass_gauss_weights

        #  Loop over the integration points
        for intpt in range(self.nmass):

            # Compute shape functions and derivatives wrt local coords
            xi = xilist[intpt]
            N = self.calc_mass_shape(xi)
            dNdxi = self.calc_mass_shape_deriv(xi)

            # Compute the Jacobian matrix J = dNdxi.x
            dxdxi = np.dot(dNdxi, coords)
            dtm = np.linalg.det(dxdxi)

            for a in range(self.nnodes):
                for b in range(self.nnodes):
                    for i in range(self.ndof):
                        row = self.ndof * a + i
                        col = self.ndof * b + i
                        mel[row, col] += self.rho * N[a] * N[b] * w[intpt] * dtm
                        continue
                    continue
                continue

            continue

        # Evaluate a lumped mass matrix using the row sum method
        lm = self.lumped_mass
        if lm:
            mel = (1. - lm) * mel + lm * np.diag(np.sum(mel, axis=1))

        return mel
