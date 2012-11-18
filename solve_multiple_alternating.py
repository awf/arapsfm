# solve_multiple_alternating.py

# Imports
from util.cmdline import * 
from itertools import count, izip

from core_recovery.lm_alt_solvers import \
    solve_instance as lm_solve_instance, \
    solve_core as lm_solve_core

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from time import time
from visualise import *
from misc.bunch import Bunch

import multiprocessing as mp
from itertools import izip_longest
from misc.numpy_ import mparray
    
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
    
# Multiprocessing

# grouper (from `itertools`)
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

# async_exec
def async_exec(f, iterable, n=None, poll=1., chunksize=1):
    def _async_exec_process(args_list):
        for args in args_list:
            if args is None:
                return

            f(*args)

    if n is None:
        n = mp.cpu_count()

    if n == 1:
        return _async_exec_process(iterable)

    active_processes = []

    for args_list in grouper(chunksize, iterable, None):
        while len(active_processes) >= n:
            for i, p in enumerate(active_processes):
                p.join(poll)
                if not p.is_alive():
                    del active_processes[i]
                    break
        
        # launch next process
        p = mp.Process(target=_async_exec_process, args=(args_list, ))
        p.start()
        active_processes.append(p)

    for p in active_processes:
        p.join()

# Main

# save_state
def save_state(output_dir, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    make_path = lambda f: os.path.join(output_dir, f)
    output_file = make_path('core.npz')

    b = Bunch(kwargs)

    print '-> %s' % output_file

    pickle_.dump(output_file,
                dict(T=b.T, 
                     V=b.V,
                     lambdas=b.lambdas,
                     preconditioners=b.preconditioners,
                     piecewise_polynomial=b.piecewise_polynomial,
                     solver_options=b.solver_options,
                     max_restarts=b.max_restarts,
                     narrowband=b.narrowband,
                     uniform_weights=b.uniform_weights,
                     find_circular_path=b.find_circular_path,
                     frames=b.frames,
                     indices=b.indices))

    for l, index in enumerate(b.indices):
        Q = geometry.path2pos(b.V1[l], b.T, b.L[l], b.U[l])

        d = dict(T=b.T,
                 V=b.V1[l],
                 Xg=b.Xg[l],
                 s=b.instScales[l],
                 X=b.X[l],
                 L=b.L[l],
                 U=b.U[l],
                 Q=Q,
                 S=b.S[l],
                 C=b.C[l],
                 P=b.P[l],
                 lambdas=b.lambdas,
                 preconditioners=b.preconditioners,
                 index=index)

        if b.frames[l] is not None:
            d['image'] = b.frames[l]

        output_file = make_path('%d.npz' % index)

        print '-> %s' % output_file
        pickle_.dump(output_file, d)

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
    parser.add_argument('piecewise_polynomial', type=str)

    parser.add_argument('--output', type=str, default=None)

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
    parser.add_argument('--outer_loops', type=int, default=5)
    parser.add_argument('--num_processes', type=int, default=1)

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
    num_instances = len(V1)

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
    piecewise_polynomial = parse_float_string(args.piecewise_polynomial)
    print 'lambdas:', lambdas
    print 'preconditioners:', preconditioners
    print 'piecewise_polynomial:', piecewise_polynomial

    # solver arguments
    solver_options = parse_solver_options(args.solver_options,
        maxIterations=20, 
        gradientThreshold=1e-6,
        updateThreshold=1e-6,
        improvementThreshold=1e-6,
        verbosenessLevel=1)
    print 'solver_options:', solver_options

    num_processes = args.num_processes
    if num_processes < 0:
        num_processes = mp.cpu_count()

    print 'num_processes:', num_processes

    # setup output directory
    # ------------------------------------------------------------------------ 
    output = args.output
    if output == 'default':
        make_str = lambda a: ",".join('%.4g' % a_ for a_ in a)
        mesh_file = os.path.split(args.mesh)[1]
        output = '%s_%s_%s_%s_%s' % (os.path.splitext(mesh_file)[0],
                                     make_str(indices),
                                     make_str(lambdas), 
                                     make_str(preconditioners),
                                     make_str(piecewise_polynomial))

    print 'output:', output

    if output is not None and not os.path.exists(output):
        print 'Creating directory:', output
        os.makedirs(output)

    # transfer vertices to shared memory
    # ------------------------------------------------------------------------
    def to_shared(a):
        b = mparray.empty_like(a)
        b.flat = a.flat
        return b

    V = to_shared(V)
    V1 = map(to_shared, V1)

    # initialise all auxilarity variables
    # ------------------------------------------------------------------------ 

    # instance rotations (X)
    X = [mparray.zeros_like(v) for v in V1]

    # global rotations (X)
    Xg = [mparray.zeros((1, 3), dtype=np.float64) for v in V1]

    # instance scales (s)
    instScales = [mparray.ones((1, 1), dtype=np.float64) for v in V1]

    # initialise silhouette (U, L)
    print 'initialising silhouette information:'

    global_silhoutte_lambdas = lambdas[:3]
    print 'global_silhoutte_lambdas:', global_silhoutte_lambdas
    print 'isCircular:', args.find_circular_path

    U = [mparray.empty((s.shape[0], 2), np.float64) for s in S]
    L = [mparray.empty(s.shape[0], np.int32) for s in S]

    def solve_silhouette_info(i):
        s = silhouette_info[i]
        u, l = shortest_path_solve(V1[i], T, S[i], SN[i],
                                   s['SilCandDistances'],
                                   s['SilEdgeCands'],
                                   s['SilEdgeCandParam'],
                                   s['SilCandAssignedFaces'],
                                   s['SilCandU'],
                                   global_silhoutte_lambdas,
                                   isCircular=args.find_circular_path)

        U[i].flat = u.flat
        L[i].flat = l.flat

    async_exec(solve_silhouette_info,
               ((i,) for i in xrange(num_instances)),
               n=num_processes,
               chunksize=max(1, num_instances / num_processes))

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
                                            # projection
                             ]

    core_lambdas = np.r_[lambdas[3], lambdas[6]]
    core_preconditioners = np.r_[preconditioners[:3],
                                 preconditioners[4]]

    print 'instance_lambdas:', instance_lambdas
    print 'core_lambdas:', core_lambdas

    print 'preconditioners:', preconditioners
    print 'narrowband:', args.narrowband
    print 'uniform_weights:', args.uniform_weights

    print 'num_instances:', num_instances
    print 'max_restarts:', args.max_restarts
    print 'outer_loops:', args.outer_loops

    t1 = time()
    for l in xrange(args.outer_loops):
        if l == 0:
            instance_lambdas[0] = lambdas[5]
        else:
            instance_lambdas[0] = lambdas[3]

        statuses = mparray.empty(num_instances + 1, dtype=np.int32)

        def solve_instance(i):
            for j in xrange(args.max_restarts):
                print '# solve_instance:', i
                status = lm_solve_instance(T, V, 
                    Xg[i], instScales[i], X[i], V1[i], U[i], L[i],
                    S[i], SN[i], Rx[i], Ry[i], 
                    instance_lambdas, 
                    preconditioners,
                    piecewise_polynomial,
                    args.narrowband,
                    args.uniform_weights,
                    i == 0,
                    **solver_options)

                print 'LM Status (%d): ' % status[0], status[1]

                # break loop if any ret status is:
                #   0 (OPTIMIZER_TIMEOUT) 
                #   4 (OPTIMIZER_CANT_BEGIN_ITERATION)
                if status[0] not in (0, 4):
                    break

            statuses[i] = status[0]

        async_exec(solve_instance, 
                   ((i,) for i in xrange(num_instances)), 
                   n=num_processes, 
                   chunksize=max(1, num_instances / num_processes))

        print instScales

        for j in xrange(args.max_restarts):
            core_opts = solver_options.copy()
            # core_opts['verbosenessLevel'] = 2

            print '# solve_core:'
            status = lm_solve_core(T, V, Xg, instScales, X, V1, 
                                   core_lambdas,
                                   core_preconditioners,
                                   args.narrowband,
                                   args.uniform_weights,
                                   **core_opts)

            print 'LM Status (%d): ' % status[0], status[1]

            if status[0] not in (0, 4):
                break

        print instScales

        statuses[-1] = status[0]

        # if not np.any((statuses == 0) | (statuses == 4)):
        #     break

    t2 = time()
    print 'time taken: %.3f' % (t2 - t1)

    if output is not None:
        scope = args.__dict__.copy()
        scope.update(locals())
        save_state(output, **scope)
    else:
        for l in xrange(len(indices)):
            print 'viewing: %d' % indices[l]
            quick_visualise(V1[l], T, L[l], U[l], S[l], frames[l])

        vis = VisualiseMesh()
        vis.add_mesh(V, T)
        vis.execute()

if __name__ == '__main__':
    main()

