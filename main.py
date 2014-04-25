import os
import sys
import argparse
import time
import numpy as np
import shutil

np.set_printoptions(precision=4)

D, F = os.path.split(os.path.realpath(__file__))

from runopts import set_runopt, has_c_compiler
from utilities.errors import WasatchError
import core.fe_model as fem
sys.tracebacklimit = 20


def run_from_cl(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--chk-mesh", default=False, action="store_true",
        help="Stop to check mesh before running simulation [default: %(default)s]")
    parser.add_argument("--piecewise", default=False, action="store_true",
        help="""Print the piecewise solution as a function of x
                [default: %(default)s""")
    parser.add_argument("--dbg", default=False, action="store_true",
        help="Debug mode [default: %(default)s]")
    parser.add_argument("--sqa", default=False, action="store_true",
        help="SQA mode [default: %(default)s]")
    parser.add_argument("-v", default=None, type=int,
        help="Verbosity [default: %(default)s]")
    parser.add_argument("--wm", default=False, action="store_true",
        help="Write mesh to ascii file [default: %(default)s]")
    parser.add_argument("-j", default=1, type=int,
        help="Number of proccesses to run simultaneously [default: %(default)s]")
    parser.add_argument("-E", default=False, action="store_true",
        help="Write exodus file [default: %(default)s]")
    parser.add_argument("--clean", default=False, action="store_true",
        help="Clean simulation output [default: %(default)s]")
    parser.add_argument("--cleanall", default=False, action="store_true",
        help="Clean all simulation output [default: %(default)s]")
    parser.add_argument("--profile", default=False, action="store_true",
        help="Run the simulation in a profiler [default: %(default)s]")
    parser.add_argument("-d", nargs="?", default=None, const="_RUNID_",
        help="Directory to run analysis [default: %(default)s]")
    parser.add_argument("--ccompiler", default="gcc",
        help=("(Optional) C compiler for compiling weave.inline code "
              "[default: %(default)s]"))
    args = parser.parse_args(argv)

    if args.profile:
        raise WasatchError("Profiling must be run from __main__")

    # set some simulation wide configurations
    set_runopt("SQA", args.sqa)
    set_runopt("DEBUG", args.dbg)
    set_runopt("VERBOSITY", args.v)
    cc = has_c_compiler(args.ccompiler)

    if args.clean or args.cleanall:
        from src.base.utilities import clean_wasatch
        clean_wasatch(os.path.splitext(args.file)[0], args.cleanall)
        return 0

    ti = time.time()

    infile = args.file
    if args.d is not None:
        fdir, fname = os.path.split(os.path.realpath(args.file))
        if args.d == "_RUNID_":
            d = os.path.join(fdir, os.path.splitext(fname)[0])
        else:
            d = os.path.realpath(args.d)
        try: os.makedirs(d)
        except OSError: pass
        infile = os.path.join(d, fname)
        try: os.remove(infile)
        except: pass
        shutil.copyfile(args.file, infile)
        os.chdir(d)

    fe_model = fem.FEModel.from_input_file(infile, verbosity=args.v)

    if args.wm:
        fe_model.mesh.write_ascii(fe_model.runid)

    if args.chk_mesh:
        fe_model.logger.write("See {0} for initial mesh".format(
                fe_model.runid + ".exo"))
        resp = raw_input("Continue with simulation? (Y/N) [N]: ")
        stop = not {"Y": True}.get(resp.upper().strip(), False)
        if stop:
            return 0

    try:
        retval = fe_model.solve(nproc=args.j)
    except KeyboardInterrupt:
        sys.stderr.write("\nKeyboard interrupt\n")
        retval = -1

    tf = time.time()
    fe_model.logger.write("wasatch: total simulation time: {0:.2f}s".format(
            tf - ti))
    fe_model.logger.close()
    return retval

if __name__ == "__main__":
    try:
        sys.argv.remove("--profile")
    except ValueError:
        sys.exit(run_from_cl())
    import cProfile as profile
    sys.exit(profile.run("run_from_cl()", "wasatch.prof"))
