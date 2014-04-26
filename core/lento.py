import os
import sys
import time
import numpy as np

import runopts as ro
import model.fe_model as fem
from utilities.lapackjac import linsolve
from utilities.errors import WasatchError
from utilities.logger import logger
from core.glob_trxn import global_traction
from core.glob_resid import global_residual
from core.glob_stiff import global_stiffness
from core.boundary import apply_dirichlet_bcs
from core.update_cell import update_element_states
from variables.varinc import VAR_SCALAR, LOC_GLOB
from model.model_data import global_data, nodal_data

HEAD = """\
 Step  Iteration   Load   Time   Time  Correction  Residual  Tolerance
         Number   Factor         Step""" 
ITER_FMT = ("{0:=4}    {1:=4}       {2:.2f}    {3:.2f}  {4:.2f}   "
            "{5:.2E}   {6:.2E}   {7:.2E}")

class Lento(fem.FEModel):

    def __init__(self, runid):
        super(Lento, self).__init__(runid)

        

    def solve(self, nproc=1, disp=0):
        """ 2D and 3D Finite Element Code

        Currently configured to run either plane strain in 2D or general 3D but
        could easily be modified for plane stress or axisymmetry.

        """
        # Local Variables
        # ---------------
        # du : array_like, (i,)
        #     Nodal displacements.
        #     Let wij be jth displacement component at ith node. Then du
        #     contains [w00, w01, w10, w11, ...] for 2D
        #     and [w00, w01, w02, w10, w11, w12, ...) for 3D

        # dw : array_like, (i,)
        #     Correction to nodal displacements.

        # K : array_like, (i, j,)
        #     Global stiffness matrix. Stored as
        #              [K_1111 K_1112 K_1121 K_1122...
        #               K_1211 K_1212 K_1221 K_1222...
        #               K_2111 K_2112 K_2121 K_2122...]
        #     for 2D problems and similarly for 3D problems

        # F : array_like, (i, )
        #     Force vector.
        #     Currently only includes contribution from tractions acting on
        #     element faces (body forces are neglected)
        # R : array_like, (i, )
        #     Volume contribution to residual
        # b : array_like (i, )
        #     RHS of equation system
        runid = self.runid
        control = self.control_params()
        X = self.mesh.nodes()
        connect = self.mesh.connect()
        elements = self.mesh.elements()
        fixnodes = self.mesh.displacement_bcs()
        nforces = self.mesh.nodal_forces()
        tractions = self.mesh.traction_bcs()

        t0 = time.time()

        dim = elements[0].ndof
        nelems = elements.shape[0]
        nnode = X.shape[0]
        ndof = elements[0].ndof
        ncoord = elements[0].ncoord
        u = np.zeros((nnode * ndof))
        du = np.zeros((nnode * ndof))
        nodal_stresses = np.zeros((nnode, 6))
        # tjf: nodal_state will have to be adjusted for multi-material where
        # each node may be connected to elements of different material.
        # nodal_state = np.zeros((nnode, max(el.material.nxtra for el in elements)))

        nproc = 1.

        #  Simulation setup
        (tint, nsteps, tol, maxit, relax, tstart,
         tterm, dtmult, verbosity) = control

        nsteps, maxit = int(nsteps), int(maxit)
        t = tstart
        dt = (tterm - tstart) / float(nsteps) * dtmult
        global_data.set_var("TIME", t)
        global_data.set_var("TIME_STEP", dt)
        nodal_data.set_var("DISPL", u)

        findstiff = True

        logger.write_intro("Implicit", runid, nsteps, tol,
                            maxit, relax, tstart, tterm,
                            ndof, nelems, nnode, elements)

        logger.write(HEAD)

        for step in range(nsteps):

            loadfactor = float(step + 1) / float(nsteps)
            err1 = 1.
            t += dt

            # Newton-Raphson loop
            mult = 10 if step == 0 and ro.reducedint else 1
            for nit in range(mult * maxit):

                # --- Update the state of each element to end of Newton step
                update_element_states(t, dt, X, elements, connect, u, du)

                # --- Update nodal stresses
                for (inode, els) in enumerate(self.mesh._node_el_map):
                    sig = np.zeros(6)
                    # nx = elements[els[0]].material.nxtra
                    # xtra = np.zeros(nx)
                    acc_volume = 0.
                    for iel in els:
                        if iel == -1:
                            break
                        el = elements[iel]
                        vol = el._volume
                        sig += el.element_data(ave=True, item="STRESS") * vol
                        # xtra += el.element_data(ave=True, item="XTRA") * vol
                        acc_volume += vol
                    nodal_stresses[inode][:] = sig / acc_volume
                    # nodal_state[inode][0:nx] = xtra / acc_volume
                    continue

                # --- Get global quantities
                if findstiff:
                    gK = global_stiffness(t, dt, X, elements, connect, du, nproc)
                    findstiff = False
                K = np.array(gK)
                F = global_traction(t, dt, X, elements, connect,
                                    tractions, nforces, du)
                R = global_residual(t, dt, X, elements, connect, du)
                b = loadfactor * F - R

                apply_dirichlet_bcs(ndof, nnode, t, fixnodes, u, du,
                                    loadfactor, K, b)

                # --- Solve for the correction
                c, dw, info = linsolve(K, b)
                if info > 0:
                    logger.write("using least squares to solve system",
                                  beg="*** ")
                    dw = np.linalg.lstsq(K, b)[0]
                elif info < 0:
                    raise WasatchError(
                        "illegal value in %d-th argument of internal dposv" % -info)

                # --- update displacement increment
                du += relax * dw

                # --- Check convergence
                wnorm = np.dot(du, du)
                err1 = np.dot(dw, dw)
                if err1 > 1.E-10:
                    findstiff = True
                if wnorm != 0.:
                    err1 = np.sqrt(err1 / wnorm)
                err2 = np.sqrt(np.dot(b, b)) / float(ndof * nnode)

                logger.write_formatted(step+1, nit+1, loadfactor, t, dt,
                                       err1, err2, tol, fmt=ITER_FMT) 
                if err1 < tol:
                    break

                continue

            else:
                raise WasatchError("Problem did not converge")

            # Update the total displacecment
            u += du

            # Update the nodal coordinates
            x = np.zeros((nnode, ndof))
            for i in range(nnode):
                for j in range(ndof):
                    x[i, j] = X[i, j] + u[ndof * i + j]
                    continue
                continue

            # Advance the state of each element
            for element in elements:
                element.advance_state()

            global_data.set_var("TIME", t)
            global_data.set_var("TIME_STEP", dt)
            nodal_data.set_var("DISPL", u)

            self.io.write_data_to_db()

            continue

        self.io.finish()
        tf = time.time()

        logger.write("\n{0}: execution finished".format(self.runid))
        logger.write("{0}: total execution time: {1:8.6f}s".format(
                runid, tf - t0))

        retval = 0
        if disp:
            retval = {
                "DISPLACEMENT": u, "TIME": t, "DT": dt,
                "ELEMENT DATA": np.array([el.element_data() for el in elements]),
                "NODAL STRESSES": nodal_stresses,
                # "NODAL STATES": nodal_state,
                "NODAL COORDINATES": x}

        return retval
