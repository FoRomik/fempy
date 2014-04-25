import os
import sys
import math
import numpy as np


def run_problem(user_input, verbosity=1):

    # ----------------------------------------------------------------------- #
    # -- To change the model problem, modify the following user inputs:
    #    nl, mp, bc, pf, pfl, q

    #
    # nl - nodal locations
    #
    nl = user_input.mesh.nodes()

    #
    # mp - material properties
    #
    # Currently, only constant material properties are supported. The use
    # must specify a in au'' + q = 0.  For example, for a solids problem,
    # a=EA and the user must specify the product of E and A
    mp = user_input.material_properties()

    #
    # bc - boundary conditions
    #
    # Robin boundary conditions are entered as python lists in the following
    # order: [a_0, b_0, g_0, a_L, b_L, g_L]
    bc = [1., 0., 0., 0., 10000., 10000.]

    #
    # pf, pfl - point forces and their locations
    #
    # Point forces and their locations are entered as python lists.
    # Locations must be entered in the global coordinates.
    pf = []
    pfl = []

    #
    # q - distributed load
    #
    q = user_input.distributed_loads()

    # end user input
    # ----------------------------------------------------------------------- #

    # linear solution
    lin_sol = Solve('linear', 'static', nl, mp, q, pf, pfl, bc)
    ul = lin_sol.nodal_values()
    print "linear nodal values", ul

    # quadratic solution
    quad_sol = Solve('quadratic', 'static', nl, mp, q, pf, pfl, bc)
    uq = quad_sol.nodal_values()
    print "quadratic nodal values", uq

    if user_input.piecewise:
        pwl = lin_sol.piecewise_solution()
        print "piecewise solution as a function of x", pwl
        pwq = quad_sol.piecewise_solution()
        print "piecewise solution as a function of x", pwq

class force:
    def __init__(self,nodeLocs,distLoad,pointForces,pointForceLocs):

        self.nl = map(float,nodeLocs)    #nodal locations
        self.pf = map(float,pointForces) #array of point forces
        self.pfl= map(float,pointForceLocs)    #point force locations

        self.q = distLoad

    def linear(self):
        # Find the number of elements and nodes and material property
        nn = len(self.nl)
        ne = nn-1

        # number of point forces
        num_pf = len(self.pf)
        if len(self.pfl) != num_pf: sys.exit('bad point load input')

        # Initialize the force array
        F = np.zeros(nn,float)

        # Loop over elements
        for e in range(nn-1):

            # jacobian
            j = (self.nl[e+1] - self.nl[e])/2.

            # Evaluate the element contributions to the force vector
            x1=-math.sqrt(3.)/3.; x2=-x1
            w1=1.; w2=w1
            q1 = self.q(self.nl[e] + j*(1.+x1))
            q2 = self.q(self.nl[e] + j*(1.+x2))

            FLeft  = j/2.*(w1*q1*(1.-x1)+w2*q2*(1.-x2))
            FRight = j/2.*(w1*q1*(1.+x1)+w2*q2*(1.+x2))

            # Brute force adjustment for point forces
            if num_pf > 0:
                for ip in range(num_pf):
                    if self.pfl[ip] > self.nl[e] and \
                            self.pfl[ip] <= self.nl[e+1]:
                        h=self.nl[e+1]-self.nl[e]
                        x0=self.pfl[ip]-self.nl[e]
                        r=x0/h
                        FLeft = FLeft + self.pf[ip]*(1.-r)
                        FRight = FRight + self.pf[ip]*r

            # Incorporate the element contributions to the force vector
            F[e] = F[e] + FLeft
            F[e+1] = F[e+1] + FRight

        return F

    def quadratic(self):
        # Find the number of elements and nodes and material property
        nn = 2*len(self.nl) - 1
        ne = len(self.nl)-1

        # number of point forces
        num_pf = len(self.pf)
        if len(self.pfl) != num_pf: sys.exit('bad point load input')

        # Initialize the force array
        F = np.zeros(nn,float)

        # Loop over elements
        ee=0
        for e in range(ne):
            # jacobian
            j = (self.nl[e+1] - self.nl[e])/2.

            # Evaluate the element contributions to the force vector
            x1=-math.sqrt(3.)/3.; x2=-x1
            w1=1.; w2=w1
            q1 = self.q(self.nl[e] + j*(1.+x1))
            q2 = self.q(self.nl[e] + j*(1.+x2))

            FLeft   = j/2.*(w1*q1*x1*(x1-1.)   + w2*q2*x2*(x2-1.))
            FMiddle = j*(w1*q1*(1.+x1)*(1.-x1) + w2*q2*(1.+x2)*(1.-x2))
            FRight  = j/2.*(w1*q1*x1*(x1+1.)   + w2*q2*x2*(x2+1.))

            # Brute force adjustment for point forces
            for ip in range(num_pf):
                if self.pfl[ip] > self.nl[e] and self.pfl[ip] <= self.nl[e+1]:
                    x0=self.pfl[ip]-self.nl[e]; h=self.nl[e+1]-self.nl[e]; r=x0/h
                    FLeft   = FLeft   + self.pf[ip]*((r-1.)*(2.*r-1.))
                    FMiddle = FMiddle + self.pf[ip]*4.*r*(1.-r)
                    FRight  = FRight  + self.pf[ip]*r*(2.*r-1.)

            # Incorporate the element contributions to the force vector
            F[ee]   = F[ee]   + FLeft
            F[ee+1] = F[ee+1] + FMiddle
            F[ee+2] = F[ee+2] + FRight
            ee+=2
        return F


class Solve:
    def __init__(self, elid, simType, nodeList, matProps, distLoad,
                 pForces, pForceLocs, boundConds):

        self.id = elid.strip()
        self.st = simType.strip()
        self.nl = nodeList
        self.mp = map(float,matProps)
        self.q = distLoad[0]
        self.pf = map(float,pForces)
        self.pfl = map(float,pForceLocs)
        self.bc  =  map(float,boundConds)

        self.stiff = stiff(self.nl,self.mp)
        self.force = force(self.nl,self.q,self.pf,self.pfl)

        if self.id[:3] == "lin":
            self.k = self.stiff.linear()
            self.f = self.force.linear()

        elif self.id[:3] == "qua":
            self.k = self.stiff.quadratic()
            self.f = self.force.quadratic()

        else:
            sys.exit('only linear and quadratic element types are supported')

        # Boundary conditions
        TN = 1.E-12
        if self.bc[1] == 0.:
            self.bc[1] = TN
        if self.bc[4] == 0.:
            self.bc[4] = TN

    def nodal_values(self):
        # apply b.c.s to stiffness
        self.k[0,0] = self.k[0,0] - self.bc[0] / self.bc[1]
        self.k[-1,-1] = self.k[-1,-1] + self.bc[3] / self.bc[4]

        self.f[0]  = self.f[0]  - self.bc[2] / self.bc[1]
        self.f[-1] = self.f[-1] + self.bc[5] / self.bc[4]

        return np.dot(np.linalg.inv(self.k),self.f)

    def piecewise_solution(self):
        nVals = self.nodal_values()
        ne = len(self.nl) - 1
        elSol = []

        if self.id[:3] == "lin":
            for e in range(ne):
                h = self.nl[e + 1] - self.nl[e]
                r = "{0} / {h}".format("x", h=h)
                basisLeft = "1. - {0}".format(r)
                basisRight = r
                sol = "{0} * {1} + {2} * {3}".format(
                    nVals[e], basisLeft, nVals[e + 1], basisRight)
                elSol.append(sol)

        elif self.id[:3] == "qua":
            ee = 0
            for e in range(ne):
                h = self.nl[e + 1] - self.nl[e]
                r = "{0} / {1}".format("x", h)
                basisLeft = "({r} - 1.) * (2. * {r} - 1.)".format(r=r)
                basisMiddle = "4. * {r} * (1. - {r})".format(r=r)
                basisRight = "{r} * (2. * {r} - 1.)".format(r=r)
                sol = "{0} * {1} + {2} * {3} + {4} * {5}".format(
                    nVals[ee], basisLeft,
                    nVals[ee + 1], basisMiddle,
                    nVals[ee + 2], basisRight)
                elSol.append(sol)
                ee += 2

        return elSol


class stiff:

    def __init__(self,nodeLocs,matProps):

        self.nl=map(float,nodeLocs)    #nodal locations
        self.mp=map(float,matProps)    #material properties

    def linear(self):
        ''' generate finite element stiffness matrix '''

        # Find the number of elements and nodes and material property
        nn = len(self.nl)
        ne = nn-1

        # Check for quality of input - k[[i]] cannot be less than zero
        if self.mp <= 0:
            sys.exit('stiffness must be positive')

            # Initialize the stiffness matrix
        K = np.zeros((nn,nn),float)

        # Loop over each element, building the stiffness matrix
        for e in range(ne):
            # Element length
            h = self.nl[e+1]-self.nl[e];

            # Incorporate the element contributions to the stiffness matrix
            K[e,e]     = K[e,e]     + self.mp[0]/h
            K[e+1,e+1] = K[e+1,e+1] + self.mp[0]/h
            K[e,e+1]   = K[e,e+1]   - self.mp[0]/h
            K[e+1,e]   = K[e+1,e]   - self.mp[0]/h

        return K

    def quadratic(self):
        ''' generate finite element stiffness matrix '''

        # Find the number of elements and nodes and material property
        nn = 2*len(self.nl) - 1
        ne = len(self.nl)-1

        # Check for quality of input - k[[i]] cannot be less than zero
        if self.mp <= 0:
            sys.exit('stiffness must be positive')

        # Initialize the stiffness matrix
        K = np.zeros((nn,nn),float)

        # Loop over each element, building the stiffness matrix
        ee=0
        for e in range(ne):
            # Element length
            th = 3.*(self.nl[e+1]-self.nl[e]);

            # Incorporate the element contributions to the stiffness matrix
            K[ee,ee]     = K[ee,ee]     + 7.*self.mp[0]/th
            K[ee,ee+1]   = K[ee,ee+1]   - 8.*self.mp[0]/th
            K[ee,ee+2]   = K[ee,ee+2]   + 1.*self.mp[0]/th

            K[ee+1,ee]   = K[ee+1,ee]   - 8.*self.mp[0]/th
            K[ee+1,ee+1] = K[ee+1,ee+1] + 16.*self.mp[0]/th
            K[ee+1,ee+2] = K[ee+1,ee+2] - 8.*self.mp[0]/th

            K[ee+2,ee]   = K[ee+2,ee]   + 1.*self.mp[0]/th
            K[ee+2,ee+1] = K[ee+2,ee+1] - 8.*self.mp[0]/th
            K[ee+2,ee+2] = K[ee+2,ee+2] + 7.*self.mp[0]/th
            ee+=2
        return K
