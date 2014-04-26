import os
import sys
import time
import numpy as np
import multiprocessing as mp

from runopts import VERBOSITY
import model.exomgr as exomgr
from utilities.errors import WasatchError
from utilities.constants import HUGE
from utilities.logger import Logger
from core.boundary import apply_dirichlet_bcs
from core.utilities import update_element_states
from core.glob_stiff import global_stiffness
from core.glob_trxn import global_traction
from core.glob_mass import global_mass


def run_job(user_input, nproc=1):
    """Run the problem given a UserInput instance

    Parameters
    ----------
    user_input : object
        UserInput class instance

    """
    exo_io = user_input.io
    runid = user_input.runid
    control = user_input.control_params()
    X = user_input.mesh.nodes()
    connect = user_input.mesh.connect()
    elements = user_input.mesh.elements()
    fixnodes = user_input.mesh.displacement_bcs()
    nforces = user_input.mesh.nodal_forces()
    tractions = user_input.mesh.traction_bcs()
    return fe_solve(exo_io, runid, control, X,
                    connect, elements, fixnodes, tractions, nforces, nproc)


def fe_solve(exo_io, runid, control, X, connect, elements, fixnodes,
             tractions, nforces, nproc=1):
    """ 2D and 3D Finite Element Code

    Currently configured to run either plane strain in 2D or general 3D but
    could easily be modified for plane stress or axisymmetry.

    Parameters
    ----------
    control : array_like, (i,)
        control[0] -> time integration scheme
        control[1] -> number of steps
        control[2] -> Newton tolerance
        control[3] -> maximum Newton iterations
        control[4] -> relax
        control[5] -> starting time
        control[6] -> termination time
        control[7] -> time step multiplier

    X : array like, (i, j,)
        Nodal coordinates
        X[i, j] -> jth coordinate of ith node for i=1...nnode, j=1...ncoord

    connect : array_like, (i, j,)
        Nodal connections
        connect[i, j] -> jth node on  on the ith element

    elements : array_like, (i,)
        Element class for each element

    fixnodes : array_like, (i, j,)
        List of prescribed displacements at nodes
            fixnodes[i, 0] -> Node number
            fixnodes[i, 1] -> Displacement component (x: 0, y: 1, or z: 2)
            fixnodes[i, 2] -> Value of the displacement

    tractions : array_like, (i, j,)
        List of element tractions
            tractions[i, 0] -> Element number
            tractions[i, 1] -> face number
            tractions[i, 2:] -> Components of traction as a function of time

    Returns
    -------
    retval : init
        0 on completion
        failure otherwise

    Notes
    -----
    Original code was adapted from [1].

    References
    ----------
    1. solidmechanics.org

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

    # number of processors
    nproc = np.amin([mp.cpu_count(), nproc, elements.size])

    # Set up timing and logging
    t0 = time.time()
    logger = Logger(runid)

    # Problem dimensions
    dim = elements[0].ndof
    nelems = elements.shape[0]
    nnode = X.shape[0]
    ndof = elements[0].ndof
    ncoord = elements[0].ncoord

    # Setup kinematic variables
    u = np.zeros((2, nnode * ndof))
    v = np.zeros((2, nnode * ndof))
    a = np.zeros((2, nnode * ndof))

    #  Simulation setup
    tint, nsteps, tol, maxit, relax, tstart, tterm, dtmult = control
    nsteps, maxit = int(nsteps), int(maxit)
    t = tstart
    dt = (tterm - tstart) / float(nsteps) * dtmult

    # Newmark parameters
    b = [.5, 0.]

    # Global mass, find only once
    M = global_mass(X, elements, connect, nproc)

    # Determine initial accelerations
    du = np.diff(u, axis=0)[0]
    K = global_stiffness(0., 1., X, elements, connect, du, nproc)
    findstiff = False
    F = global_traction(
        0., 1., X, elements, connect, tractions, nforces, du)
    apply_dirichlet_bcs(ndof, nnode, 0., fixnodes, u[0], du, 1., K, F, M)
    a[0] = np.linalg.solve(M, -np.dot(K, u[0]) + F)

    logger.write_intro("Explicit", runid, nsteps, tol, maxit, relax, tstart,
                       tterm, ndof, nelems, nnode))

    for step in range(nsteps):

        err1 = 1.
        t += dt

        logger.write(
            "Step {0:.5f}, Time: {1}, Time step: {2}".format(step + 1, t, dt))

        # --- Update the state of each element to end of step
        du = np.diff(u, axis=0)[0]
        update_element_states(dt, X, elements, connect, du)


        # --- Get global quantities
        K = global_stiffness(
            t, dt, X, elements, connect, du, nproc)
        F = global_traction(
            t, dt, X, elements, connect, tractions, du)
        MK = M + .5 * b[1] * dt * dt * K

        apply_dirichlet_bcs(ndof, nnode, t, fixnodes, u[0], du, 1., K, F, M)

        # --- Update kinematic variables
        b = F - np.dot(K, u[0] + (v[0] + .5 * (1. - b[1]) * a[0] * dt) * dt)
        a[1] = np.linalg.solve(MK, b)
        v[1] = v[0] + (1. - b[0]) * a[0] * dt + b[0] * a[1] * dt
        u[1] = u[0] + v[0] * dt + ((1. - b[1]) * a[0] + b[1] * a[1]) * dt * dt / 2.

        # Advance kinematic variables
        u[0] = u[1]
        v[0] = v[1]
        a[0] = a[1]

        # Advance the state of each element
        for element in elements:
            element.advance_state()

        # Update the nodal coordinates
        x = np.zeros((nnode, ndof))
        for i in range(nnode):
            for j in range(ndof):
                x[i, j] = X[i, j] + u[0, ndof * i + j]
                continue
            continue

        # Print the current solution
        if VERBOSITY > 2:
            exomgr.write_results(logger, t, X, connect, elements,
                                 fixnodes, tractions, u)
        if exo_io:
            exo_io.dump_time_step_data(t, dt, elements, u[0])

        continue

    if exo_io:
        exo_io.finish()
    tf = time.time()

    logger.write("\nTotal execution time: {0:.2f}s".format(tf - t0))

    return 0
