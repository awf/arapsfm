# problem.py

# Imports
import numpy as np
from misc.bunch import Bunch

from core_recovery import lm_alt_solvers2 as lm
from core_recovery.arap.sectioned import parse_k
from core_recovery.silhouette import solve_silhouette

from geometry.axis_angle import axAdd
from mesh import faces, geometry

from time import time

# DEFAULT_SOLVER_OPTIONS 
DEFAULT_SOLVER_OPTIONS = dict(maxIterations=100, 
                              gradientThreshold=1e-6,
                              updateThreshold=1e-8,
                              improvementThreshold=1e-6,
                              verbosenessLevel=1)

# Functions

# make_list_of_arrays
def make_list_of_arrays(s, n, dtype=np.float64, value=0.):
    b = []
    for i in xrange(n):
        a = np.empty(s, dtype=dtype)
        a.fill(value)
        b.append(a)
    return b

# safe_index_list
def safe_index_list(l, i):
    if i not in l:
        l.append(i)

    return l.index(i)
    
# CoreRecoverySolver
class CoreRecoverySolver(object):
    def __init__(self, 
                 lambdas,
                 preconditioners,
                 solver_options={},
                 narrowband=2,
                 uniform_weights=True,
                 max_restarts=10,
                 outer_loops=20,
                 candidate_radius=None):

        self.lambdas = lambdas
        self.preconditioners = preconditioners
        self.solver_options = DEFAULT_SOLVER_OPTIONS.copy()
        self.solver_options.update(solver_options)
        self.narrowband = narrowband
        self.uniform_weights = uniform_weights
        self.max_restarts = max_restarts
        self.outer_loops = outer_loops
        self.candidate_radius = candidate_radius

    def set_mesh(self, T, V0, silhouette_info):
        self.T = T
        self.V0 = V0
        self.silhouette_info = silhouette_info

    def set_data(self, frames, C, P, S, SN):
        self.frames = frames
        self.C = C
        self.P = P
        self.S = S
        self.SN = SN
        self.n = len(self.C)

    def setup(self, **kwargs):
        self._s = Bunch()
        self._s.V = self.V0.copy()
        self._s.V1 = make_list_of_arrays(self._s.V.shape, self.n)

        for v in self._s.V1:
            v.flat = self.V0.flat

        self._setup_global_rotations(**kwargs)
        self._setup_rotations(**kwargs)

        self._s.s = make_list_of_arrays((1, 1), self.n, value=1.0)

        self._s.U = []
        self._s.L = []
        for s in self.S:
            self._s.U.append(np.zeros((s.shape[0], 2), dtype=np.float64))
            self._s.L.append(np.zeros(s.shape[0], dtype=np.int32))

        self._setup_lambdas()

    def _setup_global_rotations(self, **kwargs):
        self.kg = kwargs.pop('kg')
        self.initial_Xgb = kwargs.pop('initial_Xgb', None)

        kg_inst, kg_basis, kg_coeff, kg_lookup = parse_k(self.kg)

        self.kg_info = Bunch(inst=kg_inst, 
                             basis=kg_basis, 
                             coeff=kg_coeff,
                             lookup=kg_lookup)

        # initialise Xg, Xgb, yg to 0.
        self._s.Xg = make_list_of_arrays((1, 3), len(kg_inst))
        self._s.Xgb = make_list_of_arrays((1, 3), len(kg_basis))
        self._s.yg = make_list_of_arrays((1, 1), len(kg_coeff))

        if self.initial_Xgb is not None:
            if self.initial_Xgb.shape != (len(kg_basis), 3):
                raise ValueError

            for i, xgb in enumerate(self.initial_Xgb):
                self._s.Xgb[i].flat = xgb.flat

    def _setup_rotations(self, **kwargs):
        self.ki = kwargs.pop('ki')
        self.initial_Xb = kwargs.pop('initial_Xb', None)

        ki_inst, ki_basis, ki_coeff, ki_lookup = parse_k(self.ki)

        self.ki_info = Bunch(inst=ki_inst, 
                             basis=ki_basis, 
                             coeff=ki_coeff,
                             lookup=ki_lookup)
        
        # initialise X, Xb, y to 0.
        self._s.X = make_list_of_arrays((len(ki_inst), 3), self.n)
        self._s.Xb = np.zeros((len(ki_basis), 3), dtype=np.float64)
        self._s.y = make_list_of_arrays((len(ki_coeff), 1), self.n)
        
        if self.initial_Xb is not None:
            if self.initial_Xb.shape != (len(ki_basis), 3):
                raise ValueError

            for i, xb in enumerate(self.initial_Xb):
                self._s.Xb[i].flat = xb.flat

    def _setup_lambdas(self):
        self.core_lambdas = np.r_[self.lambdas[3],    # as-rigid-as-possible
                                  self.lambdas[6],    # global scale acceleration penalty            
                                  self.lambdas[7],    # global rotations acceleration penalty
                                  self.lambdas[8],    # frame-to-frame acceleration penalty
                                  self.lambdas[5],    # rigid-arap to template 
                                  self.lambdas[9],    # global scale velocity penalty
                                  self.lambdas[10],   # global rotations velocity penalty
                                  self.lambdas[11]]   # frame-to-frame velocity penalty

        self.core_preconditioners = np.r_[self.preconditioners[0], # V
                                          self.preconditioners[1], # X/Xg
                                          self.preconditioners[2], # s
                                          self.preconditioners[4]] # y

        self.instance_lambdas = np.r_[self.lambdas[3],    # as-rigid-as-possible
                                      self.lambdas[1:3],  # silhouette
                                      self.lambdas[4],    # projection
                                      self.lambdas[6],    # global scale acceleration penalty
                                      self.lambdas[7],    # global rotations acceleration penalty
                                      self.lambdas[8],    # frame-to-frame acceleration penalty
                                      self.lambdas[9],    # global scale velocity penalty
                                      self.lambdas[10],   # global rotations velocity penalty
                                      self.lambdas[11]]   # frame-to-frame velocity penalty

        self.instance_preconditioners = np.r_[self.preconditioners[0], # V
                                              self.preconditioners[1], # X/Xg
                                              self.preconditioners[2], # s
                                              self.preconditioners[3], # U
                                              self.preconditioners[4]] # y

        self.silhouette_lambdas = self.lambdas[:3]

    def solve_silhouette(self, i, lambdas=None, candidate_radius=None):
        if lambdas is None:
            lambdas = self.silhouette_lambdas

        if candidate_radius is None:
            candidate_radius = self.candidate_radius

        t1 = time()
        u, l = solve_silhouette(
            self._s.V1[i],
            self.T,
            self.S[i], self.SN[i],
            self.silhouette_info['SilCandDistances'],
            self.silhouette_info['SilEdgeCands'],
            self.silhouette_info['SilEdgeCandParam'],
            self.silhouette_info['SilCandAssignedFaces'],
            self.silhouette_info['SilCandU'],
            lambdas,
            radius=candidate_radius,
            verbose=True)
        t2 = time()

        self._s.U[i].flat = u.flat
        self._s.L[i].flat = l.flat

        return t2 - t1

    def silhouette_preimages(self, i):
        return geometry.path2pos(self._s.V1[i], 
                                 self.T, 
                                 self._s.L[i], 
                                 self._s.U[i])

    def solve_instance(self, i, fixed_global_rotation=True, 
                       fixed_scale=False,
                       no_silhouette=False,
                       lambdas=None,
                       callback=None):

        t1 = time()

        kgi, Xgbi, ygi, Xgi, sm1, ym1, Xm1 = self._setup_subproblem(i)

        if lambdas is None:
            lambdas = self.instance_lambdas

        for j in xrange(self.max_restarts):
            print ' [%d] s[%d]: %.3f' % (j, i, self._s.s[i])
        
            status = lm.solve_instance(
                self.T, 
                self._s.V, 
                self._s.s[i],
                kgi, Xgbi, ygi, Xgi,
                self.ki, self._s.Xb, self._s.y[i], self._s.X[i], ym1, Xm1, sm1,
                self._s.V1[i], self._s.U[i], self._s.L[i],
                self.S[i], self.SN[i],
                self.C[i], self.P[i],
                lambdas,
                self.instance_preconditioners,
                np.r_[0., 2.],      # piecewise_polynomial
                self.narrowband,
                self.uniform_weights,
                fixed_scale,
                fixed_global_rotation,
                no_silhouette,
                callback=callback,
                **self.solver_options)

            print status[1]
            print ' [%d] s[%d]: %.3f' % (j, i, self._s.s[i])

            if status[0] not in (0, 4):
                break

        t2 = time()

        return t2 - t1

    def _setup_subproblem(self, i):
        if i > 0:
            sm1 = [self._s.s[i-1]]
            ym1 = [self._s.y[i-1]]
            Xm1 = [self._s.X[i-1]]
            subproblem = (i, i - 1)
        else:
            sm1 = []
            ym1 = []
            Xm1 = []
            subproblem = (i,)

        if i > 1:
            sm1.append(self._s.s[i-2])
            ym1.append(self._s.y[i-2])
            Xm1.append(self._s.X[i-2])
            subproblem = subproblem + (i - 2,)

        used_Xgb, used_yg, used_Xg = [], [], []
        kgi = []

        for ii in subproblem:
            m = self.kg_info.lookup[ii]
            n = self.kg[m]
            kgi.append(n)

            if n == 0:
                # fixed global rotation
                pass
            elif n == -1:
                # independent global rotation
                # NOTE: Potential problem here if multiple
                # instances share the same global rotation it will be changed
                # as each instance is processed
                kgi.append(safe_index_list(used_Xg, self.kg[m+1]))
            else:
                # n-basis rotation
                # NOTE: Potential problem here if multiple
                # instances share the same basis coefficient as it will be
                # changed as each instance is processed
                for ii in xrange(n):
                    kgi.append(safe_index_list(used_Xgb, 
                                               self.kg[m + 1 + 2*ii]))
                    kgi.append(safe_index_list(used_yg, 
                                               self.kg[m + 1 + 2*ii + 1]))

        kgi = np.require(kgi, dtype=np.int32)
        Xgbi = map(lambda i: self._s.Xgb[i], used_Xgb)
        ygi = map(lambda i: self._s.yg[i], used_yg)
        Xgi = map(lambda i: self._s.Xg[i], used_Xg)

        return kgi, Xgbi, ygi, Xgi, sm1, ym1, Xm1

    def get_instance(self, i):
        m = self.kg_info.lookup[i]
        n = self.kg[m]

        if n == 0:
            xg = np.r_[0., 0., 0.].reshape(1, 3)
        elif n == -1:
            xg = self._s.Xg[self.kg[m + 1]]
        else:
            xg = np.r_[0., 0., 0.]
            for j in xrange(n):
                yg = self._s.yg[self.kg[m + 1 + 2*j + 1]].ravel()
                xgb = self._s.Xgb[self.kg[m + 1 + 2*j]].ravel()
                xg = axAdd(yg * xgb, xg)

            xg = xg.reshape(1, 3)

        ret = dict(T=self.T, 
                   V=self._s.V1[i], 
                   C=self.C[i],
                   P=self.P[i],
                   s=self._s.s[i],
                   Xg=xg,
                   ki=self.ki,
                   Xb=self._s.Xb,
                   y=self._s.y[i],
                   X=self._s.X[i],
                   image=self.frames[i],
                   S=self.S[i],
                   Q=self.silhouette_preimages(i), 
                   L=self._s.L[i],
                   U=self._s.U[i])

        for key, value in ret.iteritems():
            if isinstance(value, np.ndarray):
                ret[key] = value.copy()

        return ret
                
    def solve_instance_callback(self, i, r=None):
        if r is None:
            r = dict(has_states=True, states=[])

        def solve_instance_callback(solver_iteration, computeDerivatives):
            if not computeDerivatives:
                return

            r['states'].append(self.get_instance(i))

        return solve_instance_callback, r
            
    def solve_core(self, fixed_Xgb=False, callback=None):
        t1 = time()

        for j in xrange(self.max_restarts):
            status = lm.solve_core(
                self.T, self._s.V, self._s.s,
                self.kg, self._s.Xgb, self._s.yg, self._s.Xg,
                self.ki, self._s.Xb, self._s.y, self._s.X, self._s.V1,
                self.core_lambdas,
                self.core_preconditioners,
                self.uniform_weights,
                fixed_Xgb,
                callback=callback,
                **self.solver_options)

            if status[0] not in (0, 4):
                break

        t2 = time()

        return t2 - t1

    def solve_core_callback(self, r=None):
        if r is None:
            r = dict(has_states=True, states=[])

        def solve_core_callback(solver_iteration, computeDerivatives):
            if not computeDerivatives:
                return

            r['states'].append(self.get_core())

        return solve_core_callback, r

    def get_core(self):
        cp = lambda l: map(np.copy, l)

        return dict(T=self.T, V=self._s.V.copy(),
                    s=cp(self._s.s),
                    kg=self.kg,
                    Xgb=cp(self._s.Xgb),
                    yg=cp(self._s.yg),
                    Xg=cp(self._s.Xg),
                    ki=self.ki,
                    Xb=self._s.Xb.copy(),
                    y=cp(self._s.y),
                    X=cp(self._s.X))

    # def __getstate__(self):
    #     return (self.__dict__,)

    # def __setstate__(self, state):
    #     self.__dict__.update(**state)

