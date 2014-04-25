import numpy as np

from utilities.errors import WasatchError

class Material(object):

    def __init__(self):
        self.nxtra = 0
        self.xtra_var_names = []
        self.xtra_var_keys = []
        self.xtra = np.empty(self.nxtra)
        self.param_map = {}

    def register_parameters(self, parameters):
        for name, idx in parameters.items():
            self._register_parameter(name, idx)
        self._params = np.zeros(len(parameters))
        return

    def _register_parameter(self, name, idx):
        name = name.upper()
        self.param_map[name] = idx
        setattr(self, name, idx)
        return

    def parameters(self):
        return self._params

    def parameter_index(self, key):
        return self.param_map.get(key.upper())

    def parse_input_parameters(self, pdict):
        self.pdict = pdict
        for name, val in pdict.items():
            i = self.param_map.get(name.upper())
            if i is None:
                if cfg.debug:
                    raise WasatchError("{0}: {1}: unrecognized parameter"
                                       .format(self.name, name))
                continue
            self._params[i] = val

    def setup(self, pdict):
        raise WasatchError("setup must be provided by model")

    def update_state(self, *args, **kwargs):
        raise WasatchError("update_state must be provided by model")

    def initialize_state(self, *args, **kwargs):
        return

    def register_variables(self, nxtra, names, keys, mig=False):
        self.nxtra = nxtra
        self.xtra = np.zeros(self.nxtra)
        if mig:
            names = [" ".join(x.split())
                     for x in "".join(names).split("|") if x.split()]
            keys = [" ".join(x.split())
                    for x in "".join(keys).split("|") if x.split()]
        self.xtra_var_names = names
        self.xtra_var_keys = keys

    def set_initial_state(self, init):
        self.xtra[:] = init

    def initial_state(self):
        return self.xtra

    def variables(self, names=False):
        if names:
            return self.xtra_var_names
        return self.xtra_var_keys
