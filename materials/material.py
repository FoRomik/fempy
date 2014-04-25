import sys

from utilities.errors import WasatchError, UserInputError
from materials.elastic import Elastic
from materials.viscoplastic import ViscoPlastic

models = {"VISCOPLASTIC": ViscoPlastic,
          "ELASTIC": Elastic}


def create_material(matname):
    model = models.get(matname.upper())
    if model is None:
        raise UserInputError("model {0} not recognized".format(matname))
    return model()
