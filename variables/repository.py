import numpy as np


from varinc import *


class VariableRepository(object):

    def __init__(self):
        self.shutdown = False
        self.registered = {LOC_GLOB: {}, LOC_NODE: {}, LOC_ELEM: {}}
        self.num_glo_vars = 0
        self.num_nod_vars = 0
        self.num_ele_vars = 0
        pass

    def register_variable(self, variable, location):

        if location not in self.registered:
            raise Exception("{0}: invalid location")

        if self.shutdown:
            raise Exception("attempting to register after shutdown")

        if variable.name in self.registered[location]:
            raise Exception("{0}: attempting to register "
                            "duplicate variable".format(variable.name))

        self.registered[location][variable.name] = {}
        if location == LOC_GLOB:
            a, b = self.num_glo_vars, self.num_glo_vars + variable.length
            self.num_glo_vars += variable.length
        elif location == LOC_NODE:
            a, b = self.num_nod_vars, self.num_nod_vars + variable.length
            self.num_nod_vars += variable.length
        else:
            # must be an element variable
            a, b = self.num_ele_vars, self.num_ele_vars + variable.length
            self.num_ele_vars += variable.length

        order = len(self.registered[location])
        self.registered[location][variable.name]["variable"] = variable
        self.registered[location][variable.name]["order"] = order
        self.registered[location][variable.name]["slice"] = slice(a, b)
        pass
        
    def shutdown_registry(self):

        self.shutdown = True    
       
        # set up the initial data array
        self.data_container = {LOC_GLOB: np.zeros(self.num_glo_vars),
                               LOC_NODE: np.zeros(self.num_nod_vars),
                               LOC_ELEM: np.zeros(self.num_ele_vars)}

        for (loc, variables) in self.registered.items():
            order = lambda k: variables[k]["order"]
            for name in sorted(variables, key=order):
                r = variables[name]
                i = r["variable"].initial_value
                s = r["slice"]
                self.data_container[loc][s] = i
        
    @property
    def element_data(self):
        return self.data_container[LOC_ELEM]

    @property
    def global_data(self):
        return self.data_container[LOC_GLOB]

    @property
    def nodal_data(self):
        return self.data_container[LOC_NODE]

    def advance_variables(self):
        for loc in self.data_container:
            self.data_container[loc][0] = self.data_container[loc][1]
