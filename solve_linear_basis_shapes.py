# solve_linear_basis_shapes.py

# Imports
import os, argparse
import numpy as np
from linear_basis_shapes.lbs_lm_solvers import solve_single_silhouette

from operator import mul, add
from geometry import axis_angle, quaternion
from mesh import faces, geometry

from visualise.visualise import *
from pprint import pprint

# Constants
FINAL_SOLVER_OPTIONS = dict(maxIterations=100, 
                            gradientThreshold=1e-6,
                            updateThreshold=1e-7,
                            improvementThreshold=1e-6,
                            verbosenessLevel=1)

# parse_args:
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('solver')
    parser.add_argument('lambdas', type=str)
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('--preconditioners', type=str, default='None')
    parser.add_argument('--max_restarts', type=int, default=10)
    parser.add_argument('--narrowband', type=int, default=2)
    parser.add_argument('--solver_options', type=str, default='{}')
    args = parser.parse_args()

    for key in ['lambdas',
                'preconditioners',
                'solver_options']:
        setattr(args, key, eval(getattr(args, key)))

    if args.preconditioners is None:
        args.preconditioners = [1.0] * 5

    for key, dtype in [('lambdas', np.float64),
                       ('preconditioners', np.float64)]:
        setattr(args, key, np.asarray(getattr(args, key), dtype=dtype))

    solver_options = FINAL_SOLVER_OPTIONS.copy()
    solver_options.update(args.solver_options)
    args.solver_options = solver_options

    return args

# main
def main():
    args = parse_args()

    arap_solver = np.load(args.solver)

    T = arap_solver.T
    Vb = [arap_solver._s.V1[args.i].copy()]

    s = np.r_[1.].reshape(-1,1)
    xg = np.r_[0., 0., 0.].reshape(-1,3)
    vd = np.r_[0., 0., 0.].reshape(-1,3)

    y = np.array(tuple(), dtype=np.float64).reshape(0,1)

    C = arap_solver.C[args.i]
    P = arap_solver.P[args.i]
    S = arap_solver.S[args.i]
    SN = arap_solver.SN[args.i]

    U = arap_solver._s.U[args.i].copy()
    L = arap_solver._s.L[args.i].copy()

    for i in xrange(args.max_restarts):
        status = solve_single_silhouette(
            T, Vb, s, xg, vd, y, U, L, 
            C, P, S, SN, args.lambdas, args.preconditioners,
            args.narrowband,
            debug=False,
            **args.solver_options)
        print status[1]

        if status[0] not in (0, 4):
            break

    V = Vb[0]
    if y.size > 0:
        V = V + reduce(add, map(mul, y, Vb[1:]))

    q = quaternion.quat(xg[0])
    R = quaternion.rotationMatrix(q)
    Rt = np.transpose(R)
    V = s[0][0] * np.dot(V, Rt) + vd[0]


    vis = VisualiseMesh()
    vis.add_mesh(V, T) # Vb[0], T)
    vis.add_image(arap_solver.frames[args.i])

    Q = geometry.path2pos(V, T, L, U)
    vis.add_projection(C, P)
    vis.add_quick_silhouette(Q, S)

    vis.camera_actions(('SetParallelProjection', (True,)))
    vis.execute()

if __name__ == '__main__':
    main()
