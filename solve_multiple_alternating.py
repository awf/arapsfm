# solve_multiple_alternating.py

# Imports
from util.cmdline import * 
from itertools import count, izip

from core_recovery.lm_alt_solvers import \
    solve_instance as lm_solve_instance

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from time import time
from visualise import *
    
# Utilities

# load_formatted
def load_formatted(indices, stem, *keys):
    r = [[] for key in keys]

    for i in indices:
        full_path = stem % i

        try:
            i = load(full_path)
        except IOError:
            break

        print '<- %s' % full_path

        if not keys:
            r.append(i)
        else:
            for l, key in enumerate(keys):
                r[l].append(i[key])

    return r

VIS_CACHE = []

# quick_visualise
def quick_visualise(V, T, L, U, S, frame=None):
    vis = VisualiseMesh()
    vis.add_mesh(V, T, L)
    if frame:
        vis.add_image(frame)
    Q = geometry.path2pos(V, T, L, U)
    N = Q.shape[0]
    vis.add_silhouette(Q, np.arange(N), [0, N-1], S)
    vis.camera_actions(('SetParallelProjection', (True,)))
    vis.execute()

    VIS_CACHE.append(vis)
    
# Main

# main
def main():
    # setup parser
    # ------------------------------------------------------------------------ 
    parser = argparse.ArgumentParser(
        description='Solve multiple frame core recovery problem')

    # core
    parser.add_argument('mesh', type=str)
    parser.add_argument('core_initialisation', type=str)
    parser.add_argument('--use_linear_transform', 
                        action='store_true',
                        default=False)

    # instances
    parser.add_argument('indices', type=str)
    parser.add_argument('instance_initialisations', type=str)

    # projection energy
    parser.add_argument('user_constraints', type=str)

    # silhouette energy
    parser.add_argument('silhouette', type=str)
    parser.add_argument('silhouette_info', type=str)

    # spillage 
    parser.add_argument('spillage', type=str)

    # general
    parser.add_argument('lambdas', type=str)
    parser.add_argument('preconditioners', type=str)

    parser.add_argument('--output', type=str)

    # visualisation
    parser.add_argument('--frames', type=str, default=None)

    # solving
    parser.add_argument('--solver_options', type=str, default=None)
    parser.add_argument('--find_circular_path', 
                        action='store_true',
                        default=False)
    parser.add_argument('--narrowband', type=int, default=3)
    parser.add_argument('--uniform_weights', 
                         action='store_true',
                         default=False)
    parser.add_argument('--max_restarts', type=int, default=5)

    # parse the arguments
    args = parser.parse_args()

    # parse argments and load variables
    # ------------------------------------------------------------------------ 

    # load the core geometry and input mesh 
    V = load_input_geometry(args.core_initialisation, 
                            args.use_linear_transform)
    T = load_input_mesh(args.mesh)
    print 'core geometry:'
    print 'V.shape:', V.shape
    print 'T.shape:', T.shape

    # parse the instance indices
    indices = eval(args.indices)
    load_instance_variables = lambda *a: load_formatted(indices, *a)
    print 'indices:', indices

    # load instance initial vertex positions
    print 'initialisations:'
    (V1,) = load_instance_variables(args.instance_initialisations, 'V')

    # load user constraints
    print 'user_constraints:'
    C, P = load_instance_variables(args.user_constraints, 'C', 'P')

    # load silhouette
    print 'silhouette:'
    S, SN = load_instance_variables(args.silhouette, 'S', 'SN')

    # load silhouette information
    print 'silhouette information:'
    silhouette_info = load_instance_variables(args.silhouette_info)

    # load spillage
    print 'spillage:'
    (R,) = load_instance_variables(args.spillage, 'R')
    Rx, Ry = map(list, izip(*R))

    # parse the lambdas and preconditioners
    lambdas = parse_float_string(args.lambdas)
    preconditioners = parse_float_string(args.preconditioners)
    print 'lambdas:', lambdas
    print 'preconditioners:', preconditioners

    # solver arguments
    solver_options = parse_solver_options(args.solver_options,
        maxIterations=20, 
        gradientThreshold=1e-6,
        updateThreshold=1e-6,
        improvementThreshold=1e-6,
        verbosenessLevel=1)
    print 'solver_options:', solver_options

    # setup output directory
    # ------------------------------------------------------------------------ 
    if not os.path.exists(args.output):
        print 'Creating directory:', args.output
        os.makedirs(args.output)

    # initialise all auxilarity variables
    # ------------------------------------------------------------------------ 

    # instance rotations (X)
    X = [np.zeros_like(v) for v in V1]

    # global rotations (X)
    Xg = [np.zeros((1, 3), dtype=np.float64) for v in V1]

    # instance scales (s)
    instScales = [np.ones((1, 1), dtype=np.float64) for v in V1]

    # initialise silhouette (U, L)
    print 'initialising silhouette information:'

    global_silhoutte_lambdas = lambdas[:3]
    print 'global_silhoutte_lambdas:', global_silhoutte_lambdas
    print 'isCircular:', args.find_circular_path

    U, L = [], []
    for l in xrange(len(indices)):
        s = silhouette_info[l]
        u, l = shortest_path_solve(V1[l], T, S[l], SN[l],
                                   s['SilCandDistances'],
                                   s['SilEdgeCands'],
                                   s['SilEdgeCandParam'],
                                   s['SilCandAssignedFaces'],
                                   s['SilCandU'],
                                   global_silhoutte_lambdas,
                                   isCircular=args.find_circular_path)
        U.append(u)
        L.append(l)

    # optional frames
    if args.frames is not None:
        frames = [args.frames % d for d in indices]
    else:
        frames = [None for d in indices]

    print 'frames:'
    pprint(frames)
        
    # alternating minimisation
    # ------------------------------------------------------------------------ 

    # multiple frame minimise
    instance_lambdas = np.r_[lambdas[3],    # as-rigid-as-possible
                             lambdas[1:3],  # silhouette
                             lambdas[4],    # spillage
                             # lambdas[6]]  # projection
                             ]

    print 'instance_lambdas:', instance_lambdas
    print 'preconditioners:', preconditioners
    print 'narrowband:', args.narrowband
    print 'uniform_weights:', args.uniform_weights

    def solve_instance(i):
        status = lm_solve_instance(T, V, 
            Xg[i], instScales[i], X[i], V1[i], U[i], L[i],
            S[i], SN[i], Rx[i], Ry[i], 
            instance_lambdas, 
            preconditioners,
            piecewisePolynomial,
            args.narrowband,
            args.uniform_weights,
            **solver_options)

        print 'LM Status (%d): ' % status[0], status[1]

        return status

    num_instances = len(V1)
    print 'num_instances:', num_instances
    print 'max_restarts:', args.max_restarts

    t1 = time()
    for i in xrange(args.max_restarts):
        inst_ret = map(solve_instance, xrange(num_instances))

        # break loop if any ret status is 0 (OPTIMIZER_TIMEOUT) 
        # or 4 (OPTIMIZER_CANT_BEGIN_ITERATION)
        for status in inst_ret:
            if status[0] in (0, 4):
                break
        else:
            break

    t2 = time()
    print 'time taken: %.3f' % (t2 - t1)

    for l in xrange(len(indices)):
        print 'viewing: %d' % indices[l]
        quick_visualise(V1[l], T, L[l], U[l], S[l], frames[l])

    vis = VisualiseMesh()
    vis.add_mesh(V, T)
    vis.execute()

if __name__ == '__main__':
    main()

