# export_deformations_on_core.py

# Imports
import os, argparse, sys
import numpy as np
from misc.pickle_ import dump, load
from solvers.arap import ARAPVertexSolver
from geometry.axis_angle import *
from geometry.quaternion import *

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_path')
    parser.add_argument('output_path')
    parser.add_argument('--N', type=int, default=-1)
    args = parser.parse_args()

    print '<- %s' % args.solver_path

    solver = load(args.solver_path)
    if args.N < 0:
        args.N = solver.n
    solve_arap = solver.get_arap_solver()

    print '... ',

    states = []
    for i in xrange(args.N):
        Xi = solver.get_instance_rotations(i)
        Ri = map(lambda x: rotationMatrix(quat(x)), Xi)
        Vi = solve_arap(Ri)
        s = np.array([[1.0]], dtype=np.float64)
        Xg = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        states.append(dict(T=solver.T, V=Vi, X=Xi, s=s, Xg=Xg,
                           image=solver.frames[i]))

        print '%d ' % i,
        sys.stdout.flush()

    sys.stdout.write('\n')

    print '-> %s' % args.output_path
    dump(args.output_path, dict(states=states, has_states=True))

if __name__ == '__main__':
    main()
