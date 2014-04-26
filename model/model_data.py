from variables.variable import VariableRepository

global_data = VariableRepository()
nodal_data = VariableRepository()
elem_data = VariableRepository()

eb_data = {}
def initialize_eb_data(elem_blk_id):
    global eb_data
    eb_data[elem_blk_id] = VariableRepository()


def alloc_data():
    global eb_data, global_data, nodal_data, elem_data
    for (key, repo) in eb_data.items():
        repo.alloc_data()
    global_data.alloc_data()
    nodal_data.alloc_data()
