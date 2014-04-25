import os
import re
import sys
import numpy as np
import xml.dom as xmldom
import xml.dom.minidom as xdom

from utilities.errors import UserInputError
from utilities.constants import W_CONST_FCN
from utilities.function_builder import build_lambda, build_interpolating_function
from mesh.mesh_generators import gen_coords_conn_from_inline

AE = "ANALYTIC EXPRESSION"
PWL = "PIECEWISE LINEAR"
COMPMAP = {"X": 0, "Y": 1, "Z": 2}
NUMREGEX = r"([+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?|\d+(?!\w+))"

class UserInputParser:
    """Parse Wasatch input files

    """
    def __init__(self, user_input):
        """Parse the xml input file

        Parameters
        ----------
        file_name : str
          path to input file

        """
        user_input = fill_in_includes(user_input)
        dom = xdom.parseString(user_input)

        # Get the root element (Should always be "MaterialModel")
        model_input = dom.getElementsByTagName("WasatchModel")
        if not model_input:
            raise UserInputError("Expected Root Element 'WasatchModel'")

        # ------------------------------------------ get and parse blocks --- #
        input_blocks = {}
        recognized_blocks = ("SolutionControl", "Mesh", "Boundary",
                             "Materials", "Functions", "Blocks")
        for block in recognized_blocks:
            elements = model_input[0].getElementsByTagName(block)
            try:
                parse_function = getattr(self, "p{0}".format(block))

            except AttributeError:
                sys.exit("{0}: not finished parsing".format(block))

            input_blocks[block] = parse_function(elements)

            for element in elements:
                p = element.parentNode
                p.removeChild(element)

        # --------------------------------------------------- check input --- #
        #   0) pop needed data from input_blocks dict and save
        #   1) check that functions are defined for entities requiring function
        #   2) replace function ids with actual function
        self.control = input_blocks.pop("SolutionControl")
        self.materials = input_blocks.pop("Materials")
        self.blk_options = input_blocks.pop("Blocks")
        self.periodic_masters_slaves = None

        boundary = input_blocks.pop("Boundary")
        functions = input_blocks.pop("Functions")
        (self.dim, self.coords, self.connect, self.el_blocks,
         self.ssets, self.nsets) = input_blocks.pop("Mesh")

        # element blocks
        for blk, items in self.blk_options.items():
            # make sure block is defined in mesh
            if blk not in [x[0] for x in self.el_blocks]:
                raise UserInputError("Blocks.Block: Block {0} not defined in "
                                     "mesh".format(blk))
            material = items[0]
            try:
                self.materials[material]
            except KeyError:
                raise UserInputError("Blocks.Block({0}): material {1} "
                                     "not defined".format(blk, material))

        # --- boundary
        self.prdisps = []
        for prdisp in boundary.get("PrescribedDisplacement", []):
            (fid, scale, nodeset, dof) = prdisp
            func = functions.get(fid)
            if func is None:
                raise UserInputError("Boundary.PrescribedDisplacement function "
                                     "{0} not defined".format(fid))
            if nodeset not in [x[0] for x in self.nsets]:
                raise UserInputError("Boundary.PrescribedDisplacement nodeset "
                                     "{0} not defined".format(nodeset))

            self.prdisps.append([func, scale, nodeset, dof])

        self.prforces = []
        for (fid, scale, nodeset, dof) in boundary.get("PrescribedForce", []):
            func = functions.get(fid)
            if func is None:
                raise UserInputError("Boundary.PrescribedForce function {0} "
                                     "not defined".format(fid))
            self.prforces.append([func, scale, nodeset, dof])

        self.tractions = []
        for (fid, scale, sideset) in boundary.get("Traction", []):
            # test if fid is a function ID or IDs
            func = None
            try:
                func = [functions[x] for x in fid]
            except TypeError:
                # must be an integer
                func = functions.get(fid)
            if func is None:
                raise UserInputError("Boundary.Traction function {0} not "
                                     "defined".format(fid))

            self.tractions.append([func, scale, sideset])

        self.distloads = []
        for (fid, scale, blx) in boundary.get("DistributedLoad", []):
            func = functions.get(fid)
            if func is None:
                raise UserInputError("Boundary.DistributedLoad function {0} "
                                     "not defined".format(fid))
            self.distloads.append([func, scale, blx])

    # ------------------------------------------------- Parsing functions --- #
    # Each XML block is parsed with a corresponding function 'pBlockName'
    # where BlockName is the name of the block as entered by the user in the
    # input file
    def pSolutionControl(self, element_list):
        """Parse the SolutionControl block and set defaults

        """
        if not element_list:
            raise UserInputError("SolutionControl: input not found")
        if len(element_list) > 1:
            raise UserInputError("SolutionControl: expected 1 block, found: "
                                 " {0}".format(len(element_list)))
        control = []
        lowstr = lambda x: str(x).lower()
        gt = lambda x: x > 0
        gte = lambda x: x >= 0
        # keyword, default, type, test
        keywords = (
            ("TimeIntegrator", 0,
             lambda x: {"implicit": 0, "explicit": 1}.get(x.lower()),
             lambda x: x in (0, 1)),
            ("NumberOfSteps", 10, int, gt),
            ("Tolerance", 1.e-4, float, gt),
            ("MaximumIterations", 30, int, gt),
            ("Relax", 1., float, gt),
            ("StartTime", 0., float, gte),
            ("TerminationTime", 10., float, gte),
            ("TimeStepMultiplier", 1., float, gt),
            ("Verbosity", 1., int, lambda x: True))

        for i, (key, default, dtype, test) in enumerate(keywords):
            tag = element_list[0].getElementsByTagName(key)
            if not tag:
                control.append(default)
                continue
            tag = tag[-1].firstChild.data.strip()
            try:
                value = dtype(tag)
            except:
                raise UserInputError(
                    "SolutionControl: {0}: expected {1}, got {2}".format(
                        key, repr(dtype), tag))
            if not test(value):
                raise UserInputError(
                    "SolutionControl: {0}: invalid value {1}".format(key, value))
            control.append(value)
        return np.array(control)

    def pMesh(self, element_list):
        """Parse the Mesh block and set defaults

        """
        if not element_list:
            raise UserInputError("Mesh: input not found")
        if len(element_list) > 1:
            raise UserInputError("Mesh: expected 1 block, found: {0}".format(
                    len(element_list)))

        mesh = element_list[0]
        mesh_type = mesh.attributes.get("type")
        if mesh_type is None:
            raise UserInputError("Mesh: mesh type not found")
        mesh_type = mesh_type.value.strip().lower()

        node_sets = []
        side_sets = []
        el_blocks = []
        for s in mesh.getElementsByTagName("AssignGroups"):
            node_sets.extend(self.get_node_sets(s.getElementsByTagName("Nodeset")))
            side_sets.extend(self.get_side_sets(s.getElementsByTagName("Sideset")))
            el_blocks.extend(self.get_el_blocks(s.getElementsByTagName("Block")))

        if not el_blocks:
            raise UserInputError("Mesh.AssignGroups: no element block assignments "
                                 "found")

        if mesh_type == "inline":
            eltype, dim, coords, connect = self.parse_inline_mesh(mesh)
            for el_block in el_blocks:
                if eltype.lower()[:3] != el_block[1][:3]:
                    raise UserInputError("Mesh.inline: {0}: inconsistent element "
                                         "type for assigned block {1}".format(
                            el_block[1], el_block[0]))

        elif mesh_type == "ascii":
            dim, coords, connect = self.parse_ascii_mesh(mesh)

        else:
            raise UserInputError("{0}: invalid mesh type".format(mesh_type))

        return dim, coords, connect, el_blocks, side_sets, node_sets

    def pBoundary(self, element_list):
        """Parse a boundary element

        """
        boundary = {}
        if not element_list:
            raise UserInputError("Boundary: input not found")
        if len(element_list) > 1:
            raise UserInputError("Boundary: expected 1 block, found: {0}".format(
                    len(element_list)))
        element = element_list[0]

        prdisps = self.parse_prescribed_displacement_block(
            element.getElementsByTagName("PrescribedDisplacement"))
        boundary["PrescribedDisplacement"] = prdisps

        prforces = self.parse_prescribed_force_block(
            element.getElementsByTagName("PrescribedForce"))
        boundary["PrescribedForce"] = prforces

        tractions = self.parse_traction_block(
            element.getElementsByTagName("Traction"))
        boundary["Traction"] = tractions

        return boundary

    def pBlocks(self, element_list):
        """Parse the Blocks block

        """
        if not element_list:
            raise UserInputError("Blocks: input not found")
        if len(element_list) > 1:
            raise UserInputError("Blocks: expected 1 block, found: {0}".format(
                    len(element_list)))
        element = element_list[0]

        blocks = {}
        for block in element.getElementsByTagName("Block"):
            blkid = block.attributes.get("id")
            if blkid is None:
                raise UserInputError("Blocks.Block: id not found")
            blkid = int(blkid.value)

            material = block.attributes.get("material")
            if material is None:
                raise UserInputError("Blocks.Block({0}): material not "
                                     "found".format(blkid))
            material = int(material.value)

            blocks[blkid] = (material,)

        return blocks

    def pMaterials(self, element_list):
        """Parse the materials block

        """
        if not element_list:
            raise UserInputError("Materials: input not found")
        if len(element_list) > 1:
            raise UserInputError("Materials: expected 1 block, found: {0}".format(
                    len(element_list)))
        element = element_list[0]

        materials = {}
        for material in element.getElementsByTagName("Material"):
            mtlid = material.attributes.get("id")
            if mtlid is None:
                raise UserInputError("Materials.Material: id not found")
            mtlid = int(mtlid.value)

            model = material.attributes.get("model")
            if model is None:
                raise UserInputError("Materials.Material: model not found")
            model = model.value.strip()

            params = {}
            for node in material.childNodes:
                if node.nodeType != material.ELEMENT_NODE:
                    continue
                val = " ".join(node.firstChild.data.split())
                try:
                    params[node.nodeName] = float(val)
                except ValueError:
                    params[node.nodeName] = str(val)
            materials[mtlid] = (model, params)

        return materials

    def pFunctions(self, element_list):
        """Parse the functions block

        """
        if len(element_list) > 1:
            raise UserInputError("Functions: expected 1 block, found: {0}".format(
                    len(element_list)))

        functions = {W_CONST_FCN: lambda x: 1.}
        if not element_list:
            return functions

        for function in element_list[0].getElementsByTagName("Function"):
            fid = function.attributes.get("id")
            if fid is None:
                raise UserInputError("Functions.Function: id not found")
            fid = int(fid.value)

            if fid == W_CONST_FCN:
                raise UserInputError("Function id {0} is reserved".format(fid))
            if fid in functions:
                raise UserInputError("{0}: function already defined".format(fid))

            ftype = function.attributes.get("type")
            if ftype is None:
                raise UserInputError("Functions.Function: type not found")
            ftype = " ".join(ftype.value.split()).upper()

            if ftype not in (AE, PWL):
                raise UserInputError("{0}: invalid function type".format(ftype))
            expr = function.firstChild.data.strip()

            if ftype == AE:
                func, err = build_lambda(expr, disp=1)
                if err:
                    raise UserInputError("{0}: in analytic expression in "
                                         "function {1}".format(err, fid))

            elif ftype == PWL:
                # parse the table in expr
                try:
                    columns = str2list(function.attributes.get("columns").value,
                                       dtype=str)
                except AttributeError:
                    columns = ["x", "y"]
                except TypeError:
                    columns = ["x", "y"]

                table = []
                ncol = len(columns)
                for line in expr.split("\n"):
                    line = [float(x) for x in line.split()]
                    if not line:
                        continue
                    if len(line) != ncol:
                        nl = len(line)
                        raise UserInputError("Expected {0} columns in function "
                                             "{1}, got {2}".format(ncol, fid, nl))
                    table.append(line)

                func, err = build_interpolating_function(np.array(table), disp=1)
                if err:
                    raise UserInputError("{0}: in piecwise linear table in "
                                         "function {1}".format(err, fid))
            functions[fid] = func
            continue

        return functions

    @staticmethod
    def parse_inline_mesh(inline_mesh_element):
        lmn_types = ("Quad",)
        lmn = {}
        for lmn_type in lmn_types:
            tags = inline_mesh_element.getElementsByTagName(lmn_type)
            if not tags:
                continue
            if len(tags) > 1:
                raise UserInputError(
                    "Mesh.inline: expected 1 {0}, got {1}".format(
                        lmn_type, len(tags)))
            lmn.setdefault(lmn_type, []).append(tags[0])

        if not lmn:
            raise UserInputError(
                "Mesh.inline: expected one of {0} block types".format(
                    ", ".join(lmn_types)))

        if len(lmn) > 1:
            raise UserInputError(
                "Mesh.inline: expected only one of {0} block types".format(
                    ", ".join(lmn_types)))

        lmn_type = lmn.keys()[0]
        lmn = lmn[lmn_type][0]

        # get [xyz]mins
        mins = []
        for item in ("xmin", "ymin", "zmin"):
            val = lmn.attributes.get("xmin", 0.)
            if val:
                val = float(val.value.strip())
            mins.append(val)

        xyzblocks = [None] * 3
        attributes = ("order", "length", "interval")
        blk_types = ("XBlock", "YBlock", "ZBlock")
        for i, tag in enumerate(blk_types):
            o = 0
            blks = []
            for xyzblock in lmn.getElementsByTagName(tag):
                blk = []
                for item in attributes:
                    attr = xyzblock.attributes.get(item)
                    if not attr:
                        raise UserInputError(
                            "Mesh.inline.{0}: expected {1} attribute".format(
                                tag, item))
                    attr = float(attr.value.strip())
                    if item == "order":
                        o += 1
                        if attr != o:
                            raise UserInputError("Mesh.inline.{0}s must be ordered "
                                                 "contiguously".format(tag))
                        continue
                    blk.append(attr)
                blks.append(blk)
            xyzblocks[i] = blks

        mesh = gen_coords_conn_from_inline(lmn_type, mins, xyzblocks)
        dim, coords, connect = mesh
        return lmn_type, dim, coords, connect

    @staticmethod
    def parse_ascii_mesh(ascii_mesh_element):

        # check for dim, Vertices, and Connectivity definitions
        vertices = ascii_mesh_element.getElementsByTagName("Vertices")
        if not vertices:
            raise UserInputError("Mesh.ascii: no Vertices found")
        if len(vertices) > 1:
            lv = len(vertices)
            raise UserInputError("Mesh.ascii: expected 1 Vertices block, "
                                 "got {0}".format(lv))
        connect = ascii_mesh_element.getElementsByTagName("Connectivity")
        if not connect:
            raise UserInputError("Mesh.ascii: no Connectivity found")
        if len(connect) > 1:
            lc = len(connect)
            raise UserInputError("Mesh.ascii: expected 1 Connectivity block, "
                                 "got {0}".format(lc))
        dim = connect[0].attributes.get("dim")
        if dim is None:
            raise UserInputError("Mesh.ascii.Connectivity: dim attribute not "
                                 "found")
        dim = int(dim.value)
        if dim not in (2, 3):
            raise UserInputError("Mesh.ascii.Connectivity: {0}: invalid "
                                 "dim".format(dim))

        # Vertices
        data = []
        for line in vertices[0].firstChild.data.split("\n"):
            line = [float(x) for x in line.split()]
            if not line:
                continue
            if len(line) > 3:
                ll = len(line)
                raise UserInputError("Mesh.ascii: expected 3 vertex coordinates, "
                                     "got {0}".format(ll))
            data.append(line + [0.] * (3 - len(line)))
        vertices = np.array(data)

        # Connectivity
        data = []
        for line in connect[0].firstChild.data.split("\n"):
            line = [int(x) for x in line.split()]
            if not line:
                continue
            data.append(line)
        connect = np.array(data)

        return dim, vertices, connect

    @staticmethod
    def parse_prescribed_displacement_block(elements):
        prescribed_displacements = []
        attributes = ("constant", "function", "dof", "scale", "nodeset")
        for (i, element) in enumerate(elements):
            const = element.attributes.get("constant")
            func = element.attributes.get("function")
            scale = element.attributes.get("scale")
            dof = element.attributes.get("dof")
            nodeset = element.attributes.get("nodeset")

            # check for required arguments and conflicts
            if dof is None:
                raise UserInputError("Boundary.PrescribedDisplacement: dof "
                                     "not specified")
            dof = dof.value.strip()

            if nodeset is None:
                raise UserInputError("Boundary.PrescribedDisplacement: nodeset "
                                     "not specified")
            nodeset = int(nodeset.value)

            # either const or func must be specified
            if const is None:
                # func must be given
                if func is None:
                    raise UserInputError("Boundary.PrescribedDisplacement: "
                                         "expected either constant or function "
                                         "attribute")
                func = int(func.value)
                if scale is None:
                    scale = 1.
                else:
                    scale = float(scale.value)

            else:
                const = float(const.value)
                if func is not None:
                    raise UserInputError("Boundary.PrescribedDisplacement: "
                                         "incompatible attributes: 'function, "
                                         "constant'")
                if scale is not None:
                    raise UserInputError("Boundary.PrescribedDisplacement: "
                                         "incompatible attributes: 'scale, "
                                         "constant'")
                func = W_CONST_FCN
                scale = const

            assert func is not None, "func is None"
            assert scale is not None, "scale is None"
            assert nodeset is not None, "nodeset is None"
            assert dof is not None, "dof is None"
            prescribed_displacements.append([func, scale, nodeset, dof])

            continue

        return prescribed_displacements

    @staticmethod
    def parse_prescribed_force_block(elements):
        prescribed_forces = []
        attributes = ("constant", "function", "dof", "scale", "nodeset")
        for (i, element) in enumerate(elements):
            const = element.attributes.get("constant")
            func = element.attributes.get("function")
            scale = element.attributes.get("scale")
            dof = element.attributes.get("dof")
            nodeset = element.attributes.get("nodeset")

            # check for required arguments and conflicts
            if dof is None:
                raise UserInputError("Boundary.PrescribedForce: dof "
                                     "not specified")
            dof = dof.value.strip()

            if nodeset is None:
                raise UserInputError("Boundary.PrescribedForce: nodeset "
                                     "not specified")
            nodeset = int(nodeset.value)

            # either const or func must be specified
            if const is None:
                # func must be given
                if func is None:
                    raise UserInputError("Boundary.PrescribedForce: expected "
                                         "either constant or function attribute")
                func = int(func.value)
                if scale is None:
                    scale = 1.
                else:
                    scale = float(scale.value)

            else:
                const = float(const.value)
                if func is not None:
                    raise UserInputError("Boundary.PrescribedForce: incompatible "
                                         "attributes: 'function, constant'")
                if scale is not None:
                    raise UserInputError("Boundary.PrescribedForce: incompatible "
                                         "attributes: 'scale, constant'")
                func = W_CONST_FCN
                scale = const

            assert func is not None, "func is None"
            assert scale is not None, "scale is None"
            assert nodeset is not None, "nodeset is None"
            assert dof is not None, "dof is None"
            prescribed_forces.append([func, scale, nodeset, dof])

            continue

        return prescribed_forces

    @staticmethod
    def parse_traction_block(elements):
        """Parse the Boundary.Traction element

        Tractions are given as either constant or function. Alternatively each
        force component can be specified as components="x=xval y=yval z=zval"

        """
        tractions = []
        attributes = ("components", "constant", "sideset", "function", "scale")
        for (i, element) in enumerate(elements):
            components = element.attributes.get("components")
            sideset = element.attributes.get("sideset")
            const = element.attributes.get("constant")
            function = element.attributes.get("function")
            scale = element.attributes.get("scale")

            # check for required arguments and conflicts
            if sideset is None:
                raise UserInputError("Boundary.Traction: sideset not specified")
            sideset = int(sideset.value)

            one_rqd = [x for x in (const, function, components) if x is not None]
            if not one_rqd:
                raise UserInputError("Boundary.Traction: expected one of "
                                     "(constant, function, components) attribute")
            if len(one_rqd) > 1:
                raise UserInputError("Boundary.Traction: expected only one of "
                                     "(constant, function, components) attribute")

            if components is not None:
                # components specified, find out which
                scale = np.zeros(3)
                components = " ".join(components.value.upper().split())
                for i, label in enumerate("XYZ"):
                    S = re.search("{0}\s*=\s*(?P<val>{1})".format(
                            label, NUMREGEX), components)
                    if S is not None:
                        scale[i] = float(S.group("val"))
                # assign the constant function
                function = [W_CONST_FCN] * 3

            elif const is not None:
                const = float(const.value)
                if scale is not None:
                    raise UserInputError("Boundary.PrescribedForce: incompatible "
                                         "attributes: 'scale, constant'")
                function = W_CONST_FCN
                scale = const

            else:
                function = int(function.value)
                if scale is None:
                    scale = 1.
                else:
                    scale = float(scale.value)


            assert sideset is not None, "sideset is None"
            tractions.append([function, scale, sideset])

            continue

        return tractions

    @staticmethod
    def get_node_sets(node_sets_element):
        """Parse Mesh.Set.Nodeset element trees

        Three options for specifying nodeset: sub_domain, nodes, point
        """
        node_sets = []
        for node_set in node_sets_element:
            nsid = node_set.attributes.get("id")
            if nsid is None:
                raise UserInputError("Set: Nodeset: 'id' not found")
            nsid = int(nsid.value.strip())

            dom = node_set.attributes.get("sub_domain")
            if dom is not None:
                dom = dom.value.strip().upper()

            nodes = node_set.attributes.get("nodes")
            if nodes is not None:
                nodes = str2list(nodes.value.strip())

            point = node_set.attributes.get("atpoint")
            if point is not None:
                point = str2list(point.value.strip(), dtype=float)
                point += [0.] * (3 - len(point))
                point = dict(zip(("X", "Y", "Z"), point))

            spec = [x for x in (dom, nodes, point) if x is not None]
            if not spec:
                raise UserInputError("Set: Nodeset: expected one of "
                                     "{sub_domain, nodes, atpoint}")
            elif len(spec) > 1:
                raise UserInputError("Set: Nodeset: specify only one of "
                                     "{sub_domain, nodes, atpoint}")
            node_sets.append([nsid, spec[0]])
        return node_sets

    @staticmethod
    def get_side_sets(side_sets_element):
        """Parse Mesh.Set.Sideset element trees

        """
        side_sets = []
        for side_set in side_sets_element:
            ssid = side_set.attributes.get("id")
            if ssid is None:
                raise UserInputError("Set: Sideset: 'id' not found")
            ssid = int(ssid.value.strip())

            dom = side_set.attributes.get("sub_domain")
            if dom is not None:
                dom = dom.value.strip().upper()

            members = side_set.attributes.get("members")
            if members is not None:
                # given as element:face pairs
                matches = re.findall(r"[0-9]+\s*?:\s*?[0-9]+", members.value)
                members = [[int(i) for i in x.split(":")] for x in matches]
                for member in members:
                    if len(member) != 2:
                        raise UserInputError("Sidesets must be defined as "
                                             "EL:FACE")
            spec = [x for x in (dom, members) if x is not None]
            if not spec:
                raise UserInputError("Mesh.AssignGroups.Sideset({0}): expected "
                                     "one of {sub_domain, members}".foramt(ssid))
            if len(spec) > 1:
                raise UserInputError("Mesh.AssignGroups.Sideset({0}): expected "
                                     "only one of {sub_domain, "
                                     "members}".foramt(ssid))
            side_sets.append([ssid, spec[0]])
        return side_sets

    @staticmethod
    def get_el_blocks(el_blocks_element):
        """Parse Mesh.Blocks.Block element trees

        """
        el_blocks = []
        unassigned, allassigned = 0, 0
        for el_block in el_blocks_element:
            el_block_id = el_block.attributes.get("id")
            if el_block_id is None:
                raise UserInputError("Blocks: Block: 'id' not found")
            el_block_id = int(el_block_id.value.strip())

            # element block elements
            elements = el_block.attributes.get("elements")
            if elements is None:
                raise UserInputError("AssignGroups.Block({0}): no elements "
                                     "assigned".format(el_block_id))
            elements = elements.value.strip().lower()

            # check special cases first
            if elements == "all":
                allassigned += 1
            elif elements == "unassigned":
                unassigned += 1

            if elements not in ("all", "unassigned"):
                elements = str2list(elements, dtype=int)

            # element type for block
            eltype = el_block.attributes.get("eltype")
            if eltype is None:
                raise UserInputError("AssignGroups.Block({0}): no eltype "
                                     "assigned".format(el_block_id))
            eltype = eltype.value.strip().lower()

            # element options for block
            elopts = el_block.attributes.get("elopts", {})
            if elopts:
                S = re.findall(r".*?=\s*[\w\.]+.*?", elopts.value)
                for (i, item) in enumerate(S):
                    item = item.split("=")
                    if len(item) != 2:
                        raise UserInputError("elopts must be of form key=val")
                    try:
                        val = int(item[1])
                    except ValueError:
                        val = str(item[1])
                    key = str("_".join(item[0].split()).upper())
                    S[i] = (key, val)
                elopts = dict(S)

            el_blocks.append([el_block_id, eltype, elements, elopts])

        if unassigned and allassigned:
            raise UserInputError("AssignGroups: elements='all' inconsistent with "
                                 "elements='unassigned'")

        if allassigned and len(el_blocks) > 1:
            raise UserInputError("AssignGroups: elements='all' inconsistent with "
                                 "mutliple assigned blocks")

        if unassigned and len(el_blocks) == 1:
            el_blocks[0][2] = "all"

        return el_blocks

def fill_in_includes(lines):
    """Look for 'include' commands in lines and insert then contents in place

    Parameters
    ----------
    lines : str
        User input

    Returns
    -------
    lines : str
        User input, modified in place, with inserts inserted

    """
    regex = r"<include\s(?P<include>.*)/>"
    _lines = []
    for line in lines.split("\n"):
        if not line.split():
            _lines.append(line.strip())
            continue
        include = re.search(regex, line)
        if include is None:
            _lines.append(line)
            continue

        href = re.search(r"""href=["'](?P<href>.*?)["']""",
                         include.group("include"))
        if not href:
            raise UserInputError("expected href='...'")
        name = href.group("href").strip()
        fpath = os.path.realpath(os.path.expanduser(name))
        try:
            fill = open(fpath, "r").read()
        except IOError:
            raise UserInputError(
                "{0}: include not found".format(repr(name)))
        _lines.extend(fill_in_includes(fill).split("\n"))
        continue
    return "\n".join(_lines)

def str2list(string, dtype=int):
    string = re.sub(r"[, ]", " ", string)
    return [dtype(x) for x in string.split()]


if __name__ == "__main__":
    f = sys.argv[1]
    parser = UserInputParser(f)
    print parser.materials
