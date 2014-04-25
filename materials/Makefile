F2PY = /opt/anaconda/bin/f2py
.PHONY: plastic
all: plastic
plastic:
	$(F2PY) -c -m plastic /home/e63302/Developer/Applications/wasatch/src/fem/materials/plastic/plastic.pyf /home/e63302/Developer/Applications/wasatch/src/fem/materials/plastic/plastic.f90 /home/e63302/Developer/Applications/wasatch/src/fem/materials/plastic/plastic_interface.f90
	mv plastic.so ./plastic