import os
import sys
import numpy as np

import runopts as ro
import mesh.mesh as femesh
from utilities.logger import logger
from mesh.mesh_generators import gen_coords_conn_from_inline
from utilities.exomgr import ExoManager
from utilities.errors import UserInputError
from utilities.function_builder import build_lambda, build_interpolating_function
from utilities.inpparse import UserInputParser


class FEModel(object):

    def __init__(self, runid, control, mesh):
        """ FE Model object

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
            control[8] -> verbosity

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
        self.runid = runid
        self.mesh = mesh
        self.dim = self.mesh.dim
        self.io = None

        self.control = control
        ro.verbosity = control[8]
        self.logger = logger
        self.logger.add_file_handler(runid + ".log")
        self.logger.set_verbosity(control[8])

        self.setup_exodus_file()

    def setup_exodus_file(self):
        """Set up the output file manager

        """
        # convert blocks and sets for Exodus II
        self.elem_blks = []
        for (blk_id, blk_els) in self.mesh.element_blocks().items():
            elem_type = self.mesh.elements(blk_els[0]).elem_type
            num_nodes_per_elem = self.mesh.elements(blk_els[0]).nnodes
            ele_var_names = self.mesh.elements(blk_els[0]).variables()
            self.elem_blks.append([blk_id, blk_els, elem_type,
                                   num_nodes_per_elem, ele_var_names])


        side_sets = []
        for (set_id, side_set) in self.mesh.sideset("all").items():
            side_sets.append([set_id, side_set[:, 0], side_set[:, 1]])

        node_sets = []
        for (set_id, node_set) in self.mesh.nodeset("all").items():
            node_sets.append([set_id, node_set])

        all_element_data = []
        for item in self.elem_blks:  #j in range(self.num_elem_blk):
            elem_blk_id = item[0]
            el_ids_this_block = item[1]
            num_elem_this_blk = len(el_ids_this_block)
            elem_this_blk = self.mesh.elements()[el_ids_this_block]
            elem_blk_data = np.array([el.element_data(ave=True)
                                      for el in elem_this_blk])
            all_element_data.append(
                (elem_blk_id, num_elem_this_blk, elem_blk_data))

        self.io = ExoManager(self.runid, self.dim, self.mesh.coords(),
                             self.mesh.connect(), self.elem_blks,
                             node_sets, side_sets, all_element_data,
                             self.mesh.exo_emap)
        pass

    def control_params(self):
        return self.control

    def mesh_instance(self):
        return self.mesh

    def distributed_loads(self):
        if self._distloads is None:
            return [lambda x: 0.]
        if len(self._distloads) > 1:
            raise UserInputError("Only one distributed load for now")
        distloads = []
        for n, distload in self._distloads.items():
            func_id = distload["FUNCTION"]
            scale = distload["SCALE"]
            blocks = distload["BLOCKS"]

            # get the function
            func = self.functions(func_id)
            if func is None:
                raise UserInputError("Function {0} not defined".format(func_id))
            distloads.append(lambda x, scale=scale: scale * func(x))
            continue
        return distloads

    def dump_time_step_data(self, t, dt, u):
        """Write the element data to the exodus file

        """
        all_element_data = []
        for item in self.elem_blks:  #j in range(self.num_elem_blk):
            elem_blk_id = item[0]
            el_ids_this_block = item[1]
            num_elem_this_blk = len(el_ids_this_block)
            elem_this_blk = self.mesh.elements()[el_ids_this_block]
            elem_blk_data = np.array([el.element_data(ave=True)
                                      for el in elem_this_blk])
            all_element_data.append(
                (elem_blk_id, num_elem_this_blk, elem_blk_data))
        self.io.write_element_data(t, dt, all_element_data, u)
        del all_element_data
        return

    @classmethod
    def from_input_file(cls, fpath, verbosity=None):
        if not os.path.isfile(fpath):
            raise UserInputError("{0}: no such file".format(fpath))
        fdir, fbase = os.path.split(fpath)
        fnam, fext = os.path.splitext(fpath)
        runid = fnam
        return cls.from_input_string(runid, open(fpath, "r").read(), fdir,
                                     verbosity=verbosity)

    @staticmethod
    def from_input_string(runid, inp, rundir=None, verbosity=None):
        """Parse input file and instantiate the FEModel object

        No other methods should interact with the dictionary parsed and returned
        from the input file parser.

        Parameters
        ----------
        fpath : str
            Path to input file

        Returns
        -------
        UserInput : object
            UserInput instance

        """
        if rundir is None:
            rundir = os.getcwd()
        ui = UserInputParser(inp) #runid, inp, rundir)

        # simulation control parameters
        if verbosity is not None:
            ui.control[8] = verbosity

        mesh = femesh.Mesh(runid, ui.dim, ui.coords, ui.connect, ui.el_blocks,
                           ui.ssets, ui.nsets, ui.prdisps, ui.prforces,
                           ui.tractions, ui.blk_options, ui.materials,
                           ui.periodic_masters_slaves)

        implicit = ui.control[0] == 0.
        if implicit:
            import core.lento as lento
            return lento.Lento(runid, ui.control, mesh)
        else:
            import core.veloz as veloz
            return veloz.Veloz(runid, ui.control, mesh)
