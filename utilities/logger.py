import os
import sys


class Logger(object):
    def __init__(self):
        self.f = None
        self.fh = None
        self.ch = sys.stdout
        self.verbosity = 1
        pass

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    def add_file_handler(self, filename):
        self.f = filename
        self.fh = open(filename, "w")
        pass

    def write(self, message, beg="", end="\n"):
        message = "{0}{1}{2}".format(beg, message, end)
        if self.verbosity:
            self.ch.write(message)
        if self.fh:
            self.fh.write(message)

    def flush(self):
        self.ch.flush()
        if self.fh:
            self.fh.flush()

    def close(self):
        self.flush()
        if self.fh:
            self.fh.close()

    def write_intro(self, integration, runid, nsteps, tol, maxit, relax,
                    tstart, tterm, ndof, nelems, nnode, elements):
        return """\
Starting Wasatch simulation for {1}

Summary of simulation input
======= == ========== =====

  Integration type
  ----------- ----
    {0}

  Control information
  ------- -----------
    number of load steps = {2:d}
               tolerance = {3:8.6f}
      maximum iterations = {4:d}
              relaxation = {5:8.6f}
              start time = {6:8.6f}
        termination time = {7:8.6f}

  Mesh information
  ---- -----------
    number of dimensions = {8:d}
      number of elements = {9:d}
      number of vertices = {10:d}
""".format(integration, runid, nsteps, tol, maxit, relax, tstart, tterm,
           ndof, nelems, nnode)


logger = Logger()



