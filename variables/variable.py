import numpy as np


from varinc import *

def catstr(a, b): return "{0}_{1}".format(a, b)

class VariableRepository(object):

    def __init__(self):
        self.shutdown = False
        self.varz = {}
        self.num_varz = 0
        self.repeat = None
        pass

    def register_variables(self, var_names, var_type, 
                           initial_value=None, length=None):
        for var_name in var_names:
            self.register_variable(var_name, var_type, 
                                   initial_value=initial_value, length=length)

    def register_variable(self, var_name, var_type,
                          initial_value=None, length=None, repeat=None):

        if self.shutdown:
            raise Exception("attempting to register after shutdown")

        if var_name in self.varz:
            return
            raise Exception("{0}: attempting to register "
                            "duplicate variable".format(var_name))

        self.repeat = repeat
        variable = _Variable(var_name, var_type, initial_value, length)

        self.varz[var_name] = {}
        a, b = self.num_varz, self.num_varz + variable.length
        order = len(self.varz)
        self.varz[variable.name]["variable"] = variable
        self.varz[variable.name]["order"] = order
        self.varz[variable.name]["slice"] = slice(a, b)
        self.num_varz += variable.length
        pass

        
    def alloc_data(self):

        self.shutdown = True    
       
        # set up the initial data array
        if self.repeat:
            self.data_container = np.zeros((self.repeat, self.num_varz))
        else:
            self.data_container = np.zeros(self.num_varz)
        order = lambda k: self.varz[k]["order"]
        for name in sorted(self.varz, key=order):
            s = self.varz[name]["slice"]
            var = self.varz[name]["variable"]
            if self.repeat:
                self.data_container[:, s] = var.initial_value
            else:
                self.data_container[s] = var.initial_value
        
    @property
    def data(self):
        return self.data_container

    @property
    def keys(self):
        keys = []
        order = lambda k: self.varz[k]["order"]
        for name in sorted(self.varz, key=order):
            keys.extend(self.varz[name]["variable"].keys)
        return keys

    def set_var(self, name, val, ndx=None): 
        var = self.varz.get(name)
        if var is None:
            raise Exception("{0}: invalid variable name".format(name))
        if not ndx:
            self.data_container[var["slice"]] = val 
        else:
            self.data_container[ndx, var["slice"]] = val 

    def get_var(self, name):
        var = self.varz.get(name)
        if var is None:
            raise Exception("{0}: invalid variable name".format(name))
        return self.data_container[var["slice"]]


class _Variable(object):
    """Variable class

    """

    def __init__(self, var_name, var_type, initial_value=None, length=None):

        if var_type not in VAR_TYPES:
            raise Exception("{0}: unknown variable type".format(var_type))

        if var_type == VAR_ARRAY:
            if length is None:
                raise Exception("array variables must define a length")
            keys = [catstr(var_name, CMP_ARRAY(i)) for i in range(length)]

        elif length is not None: 
            raise Exception("{0}: attempting to assign length".format(var_name))

        elif var_type == VAR_SCALAR: 
            length = DIM_SCALAR
            keys = [var_name]
            
        elif var_type == VAR_VECTOR: 
            length = DIM_VECTOR
            keys = [catstr(var_name, CMP_VECTOR(i)) for i in range(length)]

        elif var_type == VAR_TENSOR: 
            length = DIM_TENSOR
            keys = [catstr(var_name, CMP_TENSOR(i)) for i in range(length)]

        elif var_type == VAR_SYMTENSOR: 
            length = DIM_SYMTENSOR
            keys = [catstr(var_name, CMP_SYMTENSOR(i)) for i in range(length)]

        elif var_type == VAR_SKEWTENSOR: 
            length = DIM_SKEWTENSOR
            keys = [catstr(var_name, CMP_SKEWTENSOR(i)) for i in range(length)]

        else:
            raise Exception("{0}: unexpected variable type".format(var_type))

        # set initial value
        if initial_value is None:
            initial_value = np.zeros(length)
 
        elif isscalar(initial_value):
            initial_value = np.ones(length) * initial_value  
  
        elif len(initial_value) != length:
            raise Exception("{0}: initial_value must have "
                            "length {1}".format(var_name, length))


        self.name = var_name
        self.vtype = var_type
        self.length = length
        self.initial_value = initial_value
        self.keys = keys

        return


def isscalar(a):
    try: [i for i in a]
    except TypeError: return True
    else: return False


if __name__ == "__main__":
    elem = VariableRepository(LOC_ELEM)
    elem.register_variable("DISP", VAR_VECTOR, LOC_ELEM, 
                           initial_value=[1., 2., 3.])
    elem.register_variable("ARR", VAR_ARRAY, LOC_ELEM, length=5)

    nod = VariableRepository(LOC_NODE)
    nod.register_variable("SYMTENSOR", VAR_SYMTENSOR, LOC_NODE, 
                          initial_value=np.linspace(4, 24, 6))

    glob = VariableRepository(LOC_GLOB)
    glob.register_variable("TENSOR", VAR_TENSOR, LOC_GLOB, 
                           initial_value=np.linspace(43, 83, 9))
    glob.register_variable("SCALAR", VAR_SCALAR, LOC_GLOB, 
                           initial_value=88.)
