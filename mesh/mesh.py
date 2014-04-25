import sys
import numpy as np
from numpy.random import random as rand, seed

import utilities.defaults as defaults
from utilities.errors import UserInputError as UserInputError
from elements.elemdb import element_class_from_name
from materials.material import create_material


class Mesh(object):

    def __init__(self, runid, dim, coords, connect, el_blocks, ssets, nsets,
                 prdisps, prforces, tractions, block_options, materials,
                 periodic_masters_slaves):

        self.dim = dim
        self._sidesets = {}
        self._elem_blocks = {}
        self._nodesets = {}
        self._displacement_bcs = []
        self._nodal_forces = []
        self._traction_bcs = []
        self._elements = None
        self._coords = []
        self._conn = []

        coords, connect = self.check_coords_and_conn(dim, coords, connect)

        # assign element blocks
        # Do this first because element ordering changes
        formatted_el_blocks = self.format_el_blocks(coords, connect, el_blocks)

        # elements are put in to the exodus output by element set. As a
        # result, the element numbering in the exodus file will differ from
        # the element numbering in the fea code. exo_emap maps the number
        # in the exodus file to the number in the fea code as
        #        exodus_element_number = exo_map[i]
        # where i is the internal element number
        i = 0
        self.exo_emap = np.empty(len(connect), dtype=int)
        for (ielset, eltype, els, elopts) in formatted_el_blocks:
            for el in els:
                self.exo_emap[el] = i
                i += 1

        # instantiate element classes
        elements = self.form_elements(runid, formatted_el_blocks, connect, coords,
                                      block_options, materials)

        # assign sets
        # Sidesets are a dict made up of ID: DOMAIN pairs, where ID is the
        # integer for the sideset and DOMAIN is the domain - one of {ILO, IHI,
        # JLO, JHI}. We first get the nodes that lie on the DOMAIN and the
        # connected elements
        sidesets = self.format_sidesets(coords, connect, elements, ssets)
        nodesets = self.format_nodesets(coords, connect, nsets)

        # Now that sets have been assigned, assign the displacement and
        # traction boundary conditions to appropriate sets.
        displacement_bcs = self.parse_displacement_bc(prdisps, nodesets)

        # Nodal forces
        nodal_forces = self.parse_nodal_forces(prforces, nodesets)

        # tractions
        tractions = self.parse_traction_bc(
            tractions, sidesets, coords, connect, elements)

        # assign class members
        self._coords = coords
        self._conn = connect

        self.assign_sidesets(sidesets)
        self.assign_nodesets(nodesets)
        self.assign_elem_blocks(formatted_el_blocks)

        self.assign_displacement_bcs(displacement_bcs)
        self.assign_nodal_forces(nodal_forces)
        self.assign_traction_bcs(tractions)

        self._elements = np.array(elements)

        self._node_el_map = self.generate_node_el_map(
            self._coords.shape[0], self._conn)

        if periodic_masters_slaves and len(periodic_masters_slaves) > 1:
            raise UserInputError("Only one periodic bc currently supported")

        pass

    def nodes(self):
        return self._coords

    def coords(self):
        return self._coords

    def nnodes(self):
        """Return the number of nodes"""
        return self._coords.shape[0]

    def ncoord(self):
        return self.dim

    def connect(self):
        return self._conn

    def displacement_bcs(self):
        return self._displacement_bcs

    def traction_bcs(self):
        return self._traction_bcs

    def nodal_forces(self):
        return self._nodal_forces

    def element_blocks(self, key=None):
        if key is None:
            return self._elem_blocks
        return self._elem_blocks.get(key)

    def sideset(self, i):
        """Return a sideset

        Parameters
        ----------
        i : int
            Sideset integer ID

        Returns
        -------
        sideset : array_like, (i, j)
            sideset[i, j] -> jth face of element i in the sideset

        """
        if i == "all":
            return self._sidesets
        return self._sidesets.get(i)

    @staticmethod
    def form_elements(runid, el_blocks, connect, coords, block_options, materials):
        elements = [None] * len(connect)

        for key, val in block_options.items():
            mtlid = val[0]
            el_block = [x for x in el_blocks if x[0] == key][0]
            i, etype, els, elopts = el_block
            elopts["RUNID"] = runid
            mtl_name, mtl_params = materials.get(mtlid, (None, None))

            # Check if element is an RVE
            if mtl_name.lower() == "rve":
                etype = "RVE"
                elopts.update(mtl_params)
                # give default elastic material
                mtl_name = "elastic"
                mtl_params = {"E": 10.0E+09, "NU": .333333333}

            if mtl_name is None:
                raise UserInputError(
                    "Material {0} not defined in input file".format(mtlid))

            # instantiate the material model for element
            material = create_material(mtl_name)

            # get the element class for elements in this block
            ecls = element_class_from_name(etype)

            # determine volumes of each element
            v_els = None

            # Check for any perturbed parameters
            matparams, perturbed = {}, {}
            seed(12)
            for key, val in mtl_params.items():
                idx = material.parameter_index(key)
                if idx is None:
                    recognized = ", ".join(material.param_map)
                    raise UserInputError("{0}: {1}: unrecognized parameter. "
                                         "Recognized parameters are: {2}"
                                         .format(material.name, key, recognized))

                weibull = False
                if not weibull:
                    matparams[key] = val
                    continue

                # weibull perturbation requested
                if v_els is None:
                    v_els = np.array(
                        [ecls.volume(coords[connect[eid]]) for eid in els])
                    v_ave = np.average(v_els)

                p0 = val.get("MEDIAN")
                if p0 is None:
                    raise UserInputError("{0}: no median value given".format(key))
                k = float(val.get("VOLUME EXPONENT", 1.))
                m = float(val.get("WEIBULL MODULUS", 10.))
                msf = float(val.get("MODULUS SCALE FACTOR", 1.))
                p = [p0 * (v_ave / v_els[i]) ** (k / m) *
                     (np.log(rand()) / np.log(.5)) ** (1. / (m * msf))
                     for i, el in enumerate(els)]
                perturbed[key] = (idx, np.array(p))

            # Fill parameter array
            for iel, eid in enumerate(els):
                p = {}
                for key, (i, val) in perturbed.items():
                    p[key] = (i, val[iel])
                nodes = connect[eid]
                nodal_coords = coords[nodes]
                elements[eid] = ecls(eid, nodes, nodal_coords, material,
                                     matparams, p, **elopts)

        del matparams
        del perturbed
        return elements

    @staticmethod
    def format_el_blocks(coords, conn, el_blocks):
        """Format element sets to be passed to exodusii

        Parameters
        ----------
        coords : ndarray
            Nodal coordinates

        conn : ndarray
            Element connectivity

        el_blocks : array_like
            User given element sets ***

        Returns
        -------
        formatted_el_blocks :array_like
            Sorted list of element sets
            formatted_el_blocks[i][j] -> jth element of the ith set

        Notes
        -----
        el_blocks can be given in one of two ways:

          1) Explicitly give each element of a set.
             [[1, [23, 24, 25]]] would assign element 23, 24, 25 to element set 1

          2) Define ranges to find elements
             [[1, ((xi, xl), (yi, yl), (zi, zl))]] would assign to element set 1
             all elements in the ranges.

        """
        formatted_el_blocks, explicit, used = [], [], []
        unassigned = None

        for i, (j, eltype, members, elopts) in enumerate(el_blocks):

            if members == "all":
                members = range(len(conn))

            if members == "unassigned":
                unassigned = j
                continue

            fromrange = len(members) == 3 and all([len(x) == 2 for x in members])

            # find all element sets that are given explicitly
            if not fromrange:
                members = np.array(members, dtype=np.int)
                formatted_el_blocks.append([j, eltype, members, elopts])
                used.extend(members)
                explicit.append(j)
                continue

        # now find elements within given ranges
        hold = {}
        for (iel, el) in enumerate(conn):

            ecoords = coords[el]

            for i, eltype, members, elopts in el_blocks:
                if i in explicit or i == unassigned:
                    continue

                # look for all elements in the given range
                if all([np.amin(ecoords[:, 0]) >= members[0][0],
                        np.amax(ecoords[:, 0]) <= members[0][1],
                        np.amin(ecoords[:, 1]) >= members[1][0],
                        np.amax(ecoords[:, 1]) <= members[1][1],
                        np.amin(ecoords[:, 2]) >= members[2][0],
                        np.amax(ecoords[:, 2]) <= members[2][1]]):

                    hold.setdefault(i, []).append(iel)

        for iset, members in hold.items():
            # get eltype for this set
            el_block = [x for x in el_blocks if x[0] == iset][0]
            eltype = el_block[1]
            elopts = el_block[3]
            formatted_el_blocks.append([iset, eltype, members, elopts])
            used.extend(iset)

        # put all elements not already assigned in set 'unassigned'
        used = np.unique(used)
        orphans = [x for x in range(len(conn)) if x not in used]
        if orphans:
            if unassigned is None:
                raise UserInputError("nothing to do for unassigned elements")
            el_block = [x for x in el_blocks if x[0] == unassigned][0]
            eltype = el_block[1]
            elopts = el_block[3]
            formatted_el_blocks.append([unassigned, eltype, orphans, elopts])
        return sorted(formatted_el_blocks, key=lambda x: x[0])

    @staticmethod
    def format_sidesets(coords, conn, elements, ssets):
        """Format sidesets

        Parameters
        ----------

        Returns
        -------

        """
        sidesets = []

        lims = []
        for col in coords.T:
            lims.append([np.amin(col), np.amax(col)])
        lims = np.array(lims)

        for i, members in ssets:
            try:
                dom = members.upper()
            except AttributeError:
                dom = None

            if dom is not None:
                # user specified as ID, DOM
                if dom not in ("IHI", "ILO", "JHI", "JLO", "KHI", "KLO"):
                    raise UserInputError("Domain {0} not recognized".format(dom))
                axis = ({"I": 0, "J": 1, "K": 2}[dom[0]],
                        {"LO": 0, "HI": 1}[dom[1:]])
                nodes, els = Mesh.nodes_elements_at_pos(coords, conn,
                                                        axis[0], lims[axis])

                # Now get the face numbers for the elements in the sideset
                faces = [elements[el].topomap(dom) for el in els]
                ef = [[el, faces[n]] for n, el in enumerate(els)]
                sidesets.append(Mesh.format_sideset(i, ef))

            else:
                # given as ID, [EL, FACE]
                # Collect all from a given set and hold for next lines down
                for member in members:
                    if len(member) != 2:
                        raise UserInputError(
                            "Sidesets must be defined as [el, face], got {0}"
                            .format(repr(member)))

                sidesets.append(Mesh.format_sideset(i, members))

            continue

        return sidesets

    @staticmethod
    def format_sideset(i, ef):
        """Format a sideset list for sideset i made up of els and faces

        Parameters
        ----------
        i : int
            Sideset ID

        ef : array_like, (i, j)
            Elements and faces of sideset
            ef[i, j] jth face of ith element

        Returns
        -------
        sideset : array_like
            Formatted sideset array

        """
        return (i, ef)

    def assign_sidesets(self, sidesets):
        """Assign a sideset

        """
        for (i, sideset) in sidesets:
            self._sidesets[i] = np.array(sideset)
        return

    def assign_elem_blocks(self, elem_blocks):
        """Assign a element sets

        """
        for (i, eltype, elements, elopts) in elem_blocks:
            self._elem_blocks[i] = np.array(elements)
        return

    def exo_element_map(self):
        return self.exo_emap

    def nels(self):
        """Return the number of elements"""
        return self._conn.shape[0]

    def nelnodes(self):
        """Return the number of nodes per element"""
        return np.array([len(x != np.nan) for x in self._conn])

    def elements(self, i=None):
        """Return the ID of each element"""
        if i is None:
            return self._elements
        return self._elements[i]

    def nodeset(self, i):
        """Return a sideset

        Parameters
        ----------
        i : int
            Nodeset integer ID

        Returns
        -------
        nodeset : array_like (i,)
            nodeset[i] -> ith node of the nodeset

        """
        if i == "all":
            return self._nodesets
        return self._nodesets.get(i)

    @staticmethod
    def format_nodesets(coords, conn, nsets):
        """Format assigned nodesets and nodesets

        Parameters
        ----------
        nsets : array_like
            User given nodesets

        Returns
        -------
        nodesets : array_like
            Formatted nodeset arrays

        """
        nodesets = []

        lims = []
        for col in coords.T:
            lims.append([np.amin(col), np.amax(col)])
        lims = np.array(lims)
        for i, members in nsets:
            dom = None
            if isinstance(members, dict):
                # Look for node at point
                x, y, z = members["X"], members["Y"], members["Z"]
                nodes = Mesh.node_at_point(coords, x, y, z)

            elif members in ("ILO", "IHI", "JLO", "JHI", "KLO", "KHI"):
                dom = members.upper()
                axis = ({"I": 0, "J": 1, "K": 2}[dom[0]],
                        {"LO": 0, "HI": 1}[dom[1:]])
                nodes, els = Mesh.nodes_elements_at_pos(
                    coords, conn, axis[0], lims[axis])

            else:
                nodes = np.array(members)

            nodesets.append(Mesh.format_nodeset(i, nodes))

            continue

        return nodesets

    @staticmethod
    def format_nodeset(i, nodes):
        """Format a nodeset array for nodeset i made up of nodes

        Parameters
        ----------
        i : int
            Sideset ID

        nodes : array_like
            List of nodes in the nodeset

        Returns
        -------
        nodeset : array_like
            Formatted nodeset array

        """
        return (i, nodes)

    def assign_nodesets(self, nodesets):
        """Assign a sideset

        """
        for (i, nodes) in nodesets:
            self._nodesets[i] = np.array(nodes)
        return

    @staticmethod
    def parse_traction_bc(tractions, ssets, coords, connect, elements):
        """Parse the traction bc.

        """
        trax = []
        for (func, scale, sideset_id) in tractions:

            # now get the actual sideset
            try:
                sideset = [x[1] for x in ssets if sideset_id == x[0]][0]
            except IndexError:
                raise UserInputError("Sideset {0} not defined".format(sideset_id))

            trax.extend(Mesh.format_traction_bc(sideset, func, scale, connect,
                                                coords, elements))
            continue
        return trax

    @staticmethod
    def format_traction_bc(sideset, fcn, scale, conn, coords, elements):
        """Format a traction boundary condition

        Parameters
        ----------
        sideset : array_like
            Formatted sideset list

        fcn : function
            The function to apply

        scale : float
            Scale factor for function

        Returns
        -------
        trax : array_like
            Formatted traction BC
        """
        # sideset[i, j] -> jth face of element i in the sideset
        tractions = []
        for (element, face) in sideset:

            traction = [element, face]

            if hasattr(fcn, "__call__"):
                # tjfulle: hard coded for 2D quad
                ecls = elements[element]
                en = ecls.facenodes[face]
                nodes = conn[element, en]
                xyz = coords[nodes]
                # x = (x1, x2), dx = x2 - x1
                # y = (y1, y2), dy = y2 - y1
                # normal = (-dy, dx), (dy, -dx)
                dx = np.diff(xyz, axis=0)[0]
                normal = np.array([dx[1], -dx[0], 0.])
                normal = normal / np.sqrt(np.dot(normal, normal))
                # Assign components of the traction
                traction.extend([lambda x, scale=scale, N=N:
                                     fcn(x) * scale * N for N in normal])

            elif isinstance(fcn, list):
                # All three components of traction force given
                traction.extend([lambda x, scale=scale[i]: fcn[i](x) * scale
                                 for i in range(3)])

            else:
                raise UserInputError("Unrecognized traction function type")


            tractions.append(traction)
            continue

        return tractions

    def assign_traction_bcs(self, tractions):
        """Assign a traction boundary condition

        """
        self._traction_bcs = np.array(tractions)

    def parse_displacement_bc(self, prdisps, nsets):
        dbcs = []
        for prdisp in prdisps:
            fcn = prdisp[0]
            scale = prdisp[1]
            nodeset = prdisp[2]
            key = prdisp[3].upper()
            dofs = default_dofs(self.dim, key)

            # get the nodes in the nodeset
            try:
                nodes = [x[1] for x in nsets if nodeset == x[0]][0]
            except IndexError:
                raise UserInputError("Nodeset {0} not defined".format(nodeset))

            for dof in dofs:
                dbcs.extend(Mesh.format_displacement_bc(nodes, dof, fcn, scale))
            continue
        return dbcs

    @staticmethod
    def format_displacement_bc(nodes, dof, fcn, scale):
        """Assign a displacement boundary condition

        """
        prdisp = [[node, dof, lambda x, scale=scale: fcn(x) * scale]
                  for node in nodes]
        return prdisp

    def assign_displacement_bcs(self, displacement_bcs):
        """Assign a displacement boundary condition

        """
        self._displacement_bcs = np.array(displacement_bcs)

    def parse_nodal_forces(self, prforces, nsets):
        nfrcs = []
        for prforce in prforces:
            fcn = prforce[0]
            scale = prforce[1]
            nodeset = prforce[2]
            key = prforce[3].upper()
            dofs = default_dofs(self.dim, key)

            # get the nodes in the nodeset
            try:
                nodes = [x[1] for x in nsets if nodeset == x[0]][0]
            except IndexError:
                raise UserInputError("Nodeset {0} not defined".format(nodeset))

            for dof in dofs:
                nfrcs.extend(Mesh.format_nodal_force(nodes, dof, fcn, scale))
            continue
        return nfrcs

    @staticmethod
    def format_nodal_force(nodes, dof, fcn, scale):
        """Assign a nodal force boundary condition

        """
        nforce = [[node, dof, lambda x, scale=scale: fcn(x) * scale]
                  for node in nodes]
        return nforce

    def assign_nodal_forces(self, nodal_forces):
        """Assign a nodal force boundary condition

        """
        self._nodal_forces = np.array(nodal_forces)

    @staticmethod
    def nodes_elements_at_pos(coords, conn, axis, pos):
        """Returns nodes (and connected elements) whose position on the axis
        fall on pos.

        Parameters
        ----------
        axis : int
            Coordinate axis {x: 0, y: 1, z: 2}

        pos : float
            position

        """

        # find the nodes.
        # nonzero returns a tuple of arrays, one for each dimension of
        # coords, containing the indices of the non-zero elements in
        # that dimension. Since we are only interested in the node number -
        # which corresponds to its row - we only take the first component
        nodes = np.nonzero(coords[:, axis] == pos)[0]
        elements = np.unique(
            [i for node in nodes for (i, el) in enumerate(conn) if node in el])
        return nodes, elements

    @staticmethod
    def node_at_point(coords, x, y, z):
        """Returns node at the point

        Parameters
        ----------
        coords : ndarray
            Nodal coordinates

        x, y, z : float
            Cartesian coordinates of point

        node : int
            The nodal ID

        """
        # find the node
        node = np.where((np.abs(coords[:, 0] - x) < 1.e-10) &
                        (np.abs(coords[:, 1] - y) < 1.e-10) &
                        (np.abs(coords[:, 2] - z) < 1.e-10))
        return node[0]

    def write_ascii(self, runid):
        with open(runid + ".mesh", "w") as fobj:
            fobj.write("# Mesh Version 1\n")
            fobj.write("VERTICES\n")
            for coords in self._coords:
                ascii = " ".join("{0: 12.6E}".format(float(x)) for x in coords)
                fobj.write("{0}\n".format(ascii))
                continue
            fobj.write("END\n")
            etype =self._elements[0].name[:3].upper()
            fobj.write("CONNECTIVITY\n{0}\n".format(etype))
            ne = len(str(self.nels()))
            for e, conn in enumerate(self._conn):
                el = self._elements[e]
                mid = el.material.mid
                ascii = " ".join("{0:d}".format(int(x)) for x in conn)
                fobj.write("{0} {1:d}\n".format(ascii, mid))
                continue
            fobj.write("END\n")

            fobj.write("NODESETS\n")
            for i, nodes in self._nodesets.items():
                ascii = " ".join("{0:d}".format(int(x)) for x in nodes)
                fobj.write("{0:d} {1}\n".format(i, ascii))
                continue
            fobj.write("END\n")

            fobj.write("SIDESETS\n")
            for i, sideset in self._sidesets.items():
                for item in sideset:
                    el, face = [int(x) for x in item]
                    ascii = "{0:0{1}d} {2:d}".format(el, ne, face)
                    fobj.write("{0:d} {1}\n".format(i, ascii))
                continue
            fobj.write("END\n")

    @staticmethod
    def generate_node_el_map(nnodes, connect):
        node_el_map = [[] for i in range(nnodes)]
        for (iel, nodes) in enumerate(connect):
            for inode in nodes:
                node_el_map[inode].append(iel)
        max_node_el = max(len(x) for x in node_el_map)
        node_el_map = [x + [-1] * (max_node_el - len(x)) for x in node_el_map]
        return np.array(node_el_map)


    @staticmethod
    def check_coords_and_conn(dim, coords, connect):
        """Check the coordinates and connectivity arrays

        """
        # coords
        if not np.any(coords):
            raise UserInputError("No coords given")
        coords = np.array(coords)
        if coords.shape[1] == 2:
            coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        nnodes = coords.shape[0]

        # connectivity
        if not np.any(connect):
            raise UserInputError("No nodal connectivity given")
        conn = []
        for lmn, nodes in enumerate(connect):
            if max(nodes) > nnodes:
                raise UserInputError("Node number {0} in connect exceeds number "
                                     "of nodes in coords".format(max(nodes)))
            conn.append(nodes)
            continue
        connect = np.array(conn)
        return coords, connect


def default_dofs(dim, dof):
    _default_dofs = {"X": [defaults.X], "Y": [defaults.Y], "Z": [defaults.Z]}
    if dim == 1:
        _default_dofs.update({"ALL": [defaults.X]})
    elif dim == 2:
        _default_dofs.update({"ALL": [defaults.X, defaults.Y]})
    else:
        _default_dofs.update({"ALL": [defaults.X, defaults.Y, defaults.Z]})
    return _default_dofs[dof]
