# all elements
import ELL2, ELQ4, ELT3, ELB8
from utilities.errors import WasatchError
elemdb = {}
elemdb["ELL2"] = ELL2.ELL2
elemdb["ELQ4"] = ELQ4.ELQ4
elemdb["ELT3"] = ELT3.ELT3
elemdb["ELB8"] = ELB8.ELB8

def element_class_from_name(name):
    for clsnam, el in elemdb.items():
        if el.name[:3].upper() == name[:3].upper():
            return el
    raise WasatchError("{0}: unkown element type".format(name))

def element_class_from_id(eid):
    for name, el in elemdb.items():
        if el.eid == eid:
            return el
    raise WasatchError("{0}: unkown element type".format(eid))

def initialize(eltyp, material):
    el = elemdb.get(eltyp.upper())
    if el:
        return el(material)
    raise WasatchError("{0}: element type not recognized".format(eltyp))
