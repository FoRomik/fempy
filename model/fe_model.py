import os
import sys
import numpy as np

import runopts as ro
from utilities.logger import logger
from utilities.errors import UserInputError
from model.exomgr import ExoManager


class FEModel(object):

    def __init__(self, runid):
        """ FE Model object

        """
        self.runid = runid
        self.mesh = None

        self.logger = logger
        self.logger.add_file_handler(runid + ".log")
        self.io = None

    def attach_mesh(self, mesh):
        self.mesh = mesh
        self.dim = self.mesh.dim

    def attach_control(self, control):
        self.control = control

    def setup(self):
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

        return
