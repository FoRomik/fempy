import os
import subprocess

D = os.path.dirname(os.path.realpath(__file__))

f90_models = {"plastic":
                  {"signature": "plastic.pyf", "module name": "plastic",
                   "directory": "./plastic",
                   "files": ("plastic.f90",
                             "plastic_interface.f90")}}

def makemf(*args, **kwargs):

    f2py = kwargs.get("F2PY", "f2py")
    template = """{0}:
\t$(F2PY) -c -m {1} {2} {3}
\tmv {1}.so {4}"""

    makefile = []
    makefile.append("F2PY = {0}".format(f2py))
    models = []

    for key, model in f90_models.items():
        d = model.get("directory", "./")
        files = [os.path.realpath(os.path.join(D, d, x)) for x in model["files"]]
        signature = os.path.realpath(os.path.join(D, d, model["signature"]))
        name = model["module name"]
        models.append(key)
        makefile.append(template.format(key, name, signature, " ".join(files), d))

    _phony = ".PHONY: {0}".format(" ".join(models))
    _all = "all: {0}".format(" ".join(models))
    makefile.insert(1, _all)
    makefile.insert(1, _phony)
    mfile = os.path.join(D, "Makefile")
    with open(mfile, "w") as fobj:
        fobj.write("\n".join(makefile))

    # now make it
    cwd = os.getcwd()
    os.chdir(D)
    log = open(os.devnull, "a")
    make = subprocess.Popen(["make"], stdout=log, stderr=log)
    make.wait()
    os.chdir(cwd)
    return make.returncode


if __name__ == "__main__":
    makemf()
