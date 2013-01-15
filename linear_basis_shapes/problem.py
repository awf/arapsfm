# problem.py

# Imports
import os
import numpy as np
from time import time
from linear_basis_shapes.lbs_lm_solvers import solve_multiple
from misc.pickle_ import dump
from misc.bunch import Bunch

from operator import mul, add
from geometry import axis_angle, quaternion
from mesh import faces, geometry


# DEFAULT_SOLVER_OPTIONS
DEFAULT_SOLVER_OPTIONS = dict(
    maxIterations=100, 
    gradientThreshold=1e-6,
    updateThreshold=1e-7,
    improvementThreshold=1e-6,
    verbosenessLevel=1)

# BareLBSSolver
class BareLBSSolver(object):
    def __init__(self, T, V0, S, SN, C, P, frames):
        self.T = T
        self.V0 = V0
        self.S = S
        self.SN = SN
        self.C = C
        self.P = P
        self.frames = frames
        self.N = len(frames)

    def setup(self, U, L, D=0):
        self._s = Bunch()
        self._s.Vb = ([self.V0.copy()] + 
            map(lambda i: np.zeros_like(self.V0), xrange(D)))

        self._s.s = map(lambda i: np.r_[1.].reshape(-1,1), xrange(self.N))
        self._s.Xg = map(lambda i: np.r_[0., 0., 0.].reshape(-1,3), xrange(self.N))
        self._s.Vd = map(lambda i: np.r_[0., 0., 0.].reshape(-1,3), xrange(self.N))
        self._s.y = map(lambda i: np.ones(D, dtype=np.float64).reshape(-1, 1),
                        xrange(self.N))
        self._s.U = map(np.copy, U)
        self._s.L = map(np.copy, L)

    def add_basis(self):
        self._s.Vb.append(np.zeros_like(self.V0))
        y1 = []
        for y in self._s.y:
            y1.append(np.r_['0,2', y, [0.]])
        self._s.y = y1

    def _copy_state(self):
        s = {}
        for key, list_ in self._s.iteritems():
            s[key] = map(np.copy, list_)

        return s
        
    def __call__(self, lambdas, preconditioners, max_restarts=10, narrowband=2, 
                 **kwargs):

        solver_options = DEFAULT_SOLVER_OPTIONS.copy()
        solver_options.update(kwargs)

        states = []
        def save_state_callback(iteration, computeDerivatives):
            if not computeDerivatives:
                return
            states.append(self._copy_state())

        t1 = time()

        status = None

        for i in xrange(max_restarts):
            status = solve_multiple(
                self.T, 
                self._s.Vb, 
                self._s.s, self._s.Xg, self._s.Vd, 
                self._s.y, self._s.U, self._s.L, 
                self.C, self.P, self.S, self.SN, 
                np.require(lambdas, dtype=np.float64), 
                np.require(preconditioners, dtype=np.float64), 
                narrowband,
                debug=False,
                callback=save_state_callback,
                **kwargs)

            print status[1]

            if status[0] not in (0, 4):
                break

        t2 = time()

        return status[0], states, t2 - t1

    def export_states(self, states, output_dir):
        Z = map(lambda i: [], xrange(self.N)) 

        for n, s in enumerate(states):
            s = Bunch(s)
            for i in xrange(self.N):
                V = s.Vb[0]
                if s.y[i].size > 0:
                    V = V + reduce(add, map(mul, s.y[i], s.Vb[1:]))

                q = quaternion.quat(s.Xg[i].ravel())
                R = quaternion.rotationMatrix(q)
                Rt = np.transpose(R)
                V = s.s[i][0][0] * np.dot(V, Rt) + s.Vd[i][0]
                    
                Q = geometry.path2pos(V, self.T, s.L[i], s.U[i])

                d = dict(T=self.T, V=V,
                         C=self.C[i], P=self.P[i],
                         Q=Q, S=self.S[i],
                         image=self.frames[i],
                         s=s.s[i], Xg=s.Xg[i].ravel())

                Z[i].append(d)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, z in enumerate(Z):
            output_path = os.path.join(output_dir, '%d.dat' % i)
            print '-> %s' % output_path
            dump(output_path, dict(has_states=True, states=z))

