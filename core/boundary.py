import sys
import numpy as np
from scipy.weave import inline, converters

from runopts import ENABLE_WEAVE


def apply_dirichlet_bcs(ndof, nnode, t, bcu, u, du, umag, K, rhs, M=None):
    """Apply Dirichlet (essential) boundary conditions

    Parameters
    ----------
    ndof : int
        Number of degrees of freedom

    nnode : int
        Number of nodes

    t : float
        Current time

    bcu : array_like
        List of prescribed displacements at nodes
            bcu[i, 0] -> Node number
            bcu[i, 1] -> Displacement dof (x: 0, y: 1, or z: 2)
            bcu[i, 2] -> Value of the displacement

    u : array_like
        Nodal displacements

    du : array_like
        Increment of nodal displacements

    umag : float
        Magnitude of prescribed displacement, bcu[i, 2] * umag

    K : array_like
        The stiffness matrix

    rhs : array_like
        RHS of system of equations

    Returns
    -------
    K, (M), and rhs are modified in place

    """

    mass = np.empty(0) if M is None else M
    mass_matrix = 0 if M is None else 1

    # --- Prescribed displacements
    for (n, dof, disp) in bcu:
        # n:    node number
        # dof:  displacement component
        # disp: displacement function, as function of time
        rw = int(ndof * n + dof)

        # Current value of displacement
        u_cur = du[rw] + u[rw]
        ufac = float(umag * disp(t) - u_cur)

        # Modify the RHS and set all Kij = 0. for this DOF.
        # Modify rows and cols of K, M
        if not ENABLE_WEAVE:

            for i in range(ndof * nnode):
                rhs[i] -= K[i, rw] * ufac
                K[rw, i] = K[i, rw] = 0.
                if mass_matrix == 1:
                    mass[rw, i] = mass[i, rw] = 0.

            K[rw, rw] = 1.;
            if mass_matrix == 1:
                mass[rw, rw] = 1.

            rhs[rw] = ufac

        else:
            code = """
                for (int i=0; i < ndof * nnode; ++i) {
                  rhs(i) -= K(i, rw) * ufac;
                  K(rw, i) = 0.; K(i, rw) = 0.;
                  if (mass_matrix == 1) {
                    mass(rw, i) = 0.; mass(i, rw) = 0.;
                    }
                  }
                K(rw, rw) = 1.;
                if (mass_matrix == 1) {
                  mass(rw, rw) = 1.;
                  }
                rhs(rw) = ufac;
            """
            inline(code, ["nnode", "ndof", "rw", "K", "mass", "mass_matrix",
                          "ufac", "rhs"],
                   type_converters=converters.blitz)

        continue
