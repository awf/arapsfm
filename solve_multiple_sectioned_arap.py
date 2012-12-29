# solve_multiple_sectioned_arap.py

# Imports
from util.cmdline import * 
from itertools import count, izip

from core_recovery.lm_alt_solvers import \
    solve_instance_sectioned_arap_temporal as lm_solve_instance, \
    solve_core_sectioned_arap as lm_solve_core, \
    solve_two_source_arap_proj

from core_recovery.lm_solvers import solve_single_arap_proj

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
def async_exec(f, iterable, n=None, timeout=5., chunksize=1):
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

    def wait_for_active_processes(n):
        while len(active_processes) >= n:
            print 'active_processes:'
            pprint(active_processes)
            for i, p in enumerate(active_processes):
                'join: %d' % p.pid
                p.join(timeout)

                if not p.is_alive():
                    '%d is NOT alive' % p.pid
                    break
            else:
                continue

            print 'deleting: %d' % p.pid
            del active_processes[i]

    for args_list in grouper(chunksize, iterable, None):
        wait_for_active_processes(n)
        
        # launch next process
        p = mp.Process(target=_async_exec_process, args=(args_list, ))
        p.start()
        active_processes.append(p)

    wait_for_active_processes(1)

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
                     K=b.K,
                     Xb=b.Xb,
                     X=b.X,
                     y=b.y,
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
                 K=b.K,
                 Xb=b.Xb,
                 y=b.y[l],
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

    # initial silhouette information
    parser.add_argument('core_silhouette_info', type=str)

    # arap sections
    parser.add_argument('arap_sections', type=str)

    # instances
    parser.add_argument('indices', type=str)

    # projection energy
    parser.add_argument('user_constraints', type=str)

    # silhouette energy
    parser.add_argument('silhouette', type=str)

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

    # optional
    parser.add_argument('--use_single_process', type=int, default=None)
    parser.add_argument('--quit_after_silhouette',
                        action='store_true',
                        default=False)

    # parse the arguments
    args = parser.parse_args()

    # setup output directory
    # ------------------------------------------------------------------------ 

    # parse the instance indices
    indices = eval(args.indices)
    load_instance_variables = lambda *a: load_formatted(indices, *a)
    print 'indices:', indices

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

    output = args.output

    if '{default}' in output:
        make_str = lambda a: ",".join('%.4g' % a_ for a_ in a)
        mesh_file = os.path.split(args.mesh)[1]

        if len(indices) > 20:
            indices_str = (make_str(indices[:10]) + '---' +
                           make_str(indices[-10:]))
        else:
            indices_str = make_str(indices)

        default_directory = '%s_%s_%s_%s_%s' % (
            os.path.splitext(mesh_file)[0],
            indices_str,
            make_str(lambdas), 
            make_str(preconditioners),
            make_str(piecewise_polynomial))

        output = output.format(default=default_directory)

    print 'output:', output

    if output is not None and not os.path.exists(output):
        print 'Creating directory:', output
        os.makedirs(output)

    # parse argments and load variables
    # ------------------------------------------------------------------------ 

    # load the core geometry and input mesh 
    V = load_input_geometry(args.core_initialisation, 
                            args.use_linear_transform)
    T = load_input_mesh(args.mesh)
    print 'core geometry:'
    print 'V.shape:', V.shape
    print 'T.shape:', T.shape

    # load user constraints
    print 'user_constraints:'
    C, P = load_instance_variables(args.user_constraints, 'C', 'P')
    num_instances = len(C)

    # load silhouette
    print 'silhouette:'
    S, SN = load_instance_variables(args.silhouette, 'S', 'SN')

    # load silhouette information
    print 'silhouette information:'
    silhouette_info = np.load(args.core_silhouette_info)
    print '<- %s' % args.core_silhouette_info

    # load spillage
    print 'spillage:'
    (R,) = load_instance_variables(args.spillage, 'R')
    Rx, Ry = map(list, izip(*R))

    # transfer vertices to shared memory
    # ------------------------------------------------------------------------
    def to_shared(a):
        b = mparray.empty_like(a)
        b.flat = a.flat
        return b

    V = to_shared(V)
    V1 = map(to_shared, num_instances * [V])

    # initialise all auxilarity variables
    # ------------------------------------------------------------------------ 

    # global rotations (X)
    Xg = [mparray.zeros((1, 3), dtype=np.float64) for v in V1]

    # instance scales (s)
    instScales = [mparray.ones((1, 1), dtype=np.float64) for v in V1]

    # sectioned arap specification
    K = np.load(args.arap_sections)['K']

    # basis rotations shared across each instance (Xb)
    i = np.argwhere(K[:, 0] > 0).ravel()
    unique_Xbi = np.unique(K[i, 1])
    NXb = unique_Xbi.shape[0]
    if NXb > 0:
        assert NXb == np.amax(unique_Xbi) + 1

    Xb = mparray.zeros((NXb, 3), dtype=np.float64)

    # instance basis coefficients (y)
    unique_yi = np.unique(K[i, 0])
    Nyi = unique_yi.shape[0]
    if Nyi > 0:
        # no requirement for `+1` because y-indices are 1-based
        assert Nyi == np.amax(unique_yi)

    y = [mparray.ones((Nyi, 1), dtype=np.float64) for v in V1]

    # free (shared) rotations in each instance (X)
    i = np.argwhere(K[:, 0] < 0).ravel()
    unique_Xi = np.unique(K[i, 1])
    NX = unique_Xi.shape[0]
    if NX > 0:
        assert NX == np.amax(unique_Xi) + 1

    X = [mparray.zeros((NX, 3), dtype=np.float64) for v in V1]

    # initialise silhouette (U, L)
    print 'initialising silhouette information:'

    global_silhoutte_lambdas = lambdas[:3]
    print 'global_silhoutte_lambdas:', global_silhoutte_lambdas
    print 'isCircular:', args.find_circular_path

    U = [mparray.empty((s.shape[0], 2), np.float64) for s in S]
    L = [mparray.empty(s.shape[0], np.int32) for s in S]

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
                             lambdas[5],    # projection
                             lambdas[7],    # temporal ARAP penalty
                             lambdas[8],    # global rotations penalty
                             lambdas[9],    # global scale penalty 
                             lambdas[10]]   # regular rotations penalty

    core_lambdas = np.r_[lambdas[3],    # as-rigid-as-possible
                         lambdas[8],    # global rotations penalty
                         lambdas[6],    # laplacian
                        ]

    core_preconditioners = np.r_[preconditioners[0], # V
                                 preconditioners[1], # X
                                 preconditioners[2], # s
                                 preconditioners[4], # Xg
                                 preconditioners[5]] # y

    print 'instance_lambdas:', instance_lambdas
    print 'core_lambdas:', core_lambdas

    print 'preconditioners:', preconditioners
    print 'narrowband:', args.narrowband
    print 'uniform_weights:', args.uniform_weights

    print 'num_instances:', num_instances
    print 'max_restarts:', args.max_restarts
    print 'outer_loops:', args.outer_loops

    # solve for initialisations without the basis rotations
    initialisation_lambdas = np.r_[
        lambdas[5],   # projection
        lambdas[3],   # as-rigid-as-possible
        lambdas[7],   # temporal ARAP penalty
        lambdas[8],   # global rotations regularisation
        lambdas[9],   # global scale regularisation
        lambdas[10]]  # frame-to-frame rotations regularisation

    # initialisation preconditioners
    initialisation_preconditioners = np.r_[preconditioners[0], # V
                                           preconditioners[1], # X
                                           preconditioners[2], # s
                                           preconditioners[4]] # Xg
                         

    # initialise `iX` intermediate rotations
    iX = np.zeros_like(V1[0])
    empty3 = np.array(tuple(), dtype=np.float64).reshape(0, 3)

    # set initialisation solver options and update the maximum number of
    # iterations
    init_solver_options = solver_options.copy()
    init_solver_options['maxIterations']= 50

    # solve for V1[0]
    V1[0].flat = V.flat

    for j in xrange(args.max_restarts):
        status = solve_two_source_arap_proj(T, 
                                            V, Xg[0], instScales[0], iX,
                                            empty3, empty3, empty3, empty3,
                                            V1[0],
                                            C[0], P[0],
                                            initialisation_lambdas,
                                            initialisation_preconditioners,
                                            args.uniform_weights,
                                            **init_solver_options)

        if status[0] not in (0, 4):
            break

    # solve for V1[i] for i > 0 using user constraints
    def solve_initialisation(i):
        print '> solve_initialisation:', i

        # initialise to previous frame
        for a in (V1, Xg, instScales):
            a[i].flat = a[i-1].flat

        Vp = V1[i-1]
        Xgp = np.zeros((1, 3), dtype=np.float64)
        sp = np.ones((1, 1), dtype=np.float64)
        Xp = np.zeros_like(Vp)

        for j in xrange(args.max_restarts):
            status = solve_two_source_arap_proj(T, 
                                                V, Xg[i], instScales[i], iX,
                                                Vp, Xgp, sp, Xp,
                                                V1[i], 
                                                C[i], P[i],
                                                initialisation_lambdas,
                                                initialisation_preconditioners,
                                                args.uniform_weights,
                                                **init_solver_options)

            print status[1]

            if status[0] not in (0, 4):
                break

        print '< solve_initialisation:', i

    map(solve_initialisation, xrange(1, num_instances))

    # iterate by updating the core and then the instances
    t1 = time()
    for l in xrange(args.outer_loops):
        # save optimisation statuses
        statuses = mparray.empty(num_instances + 1, dtype=np.int32)

        # re-initialise U and L
        def solve_silhouette_info(i):
            print '> solve_silhouette_info:', i

            D = instScales[i] * silhouette_info['SilCandDistances']

            u, l = shortest_path_solve(V1[i], T, S[i], SN[i],
                                       D,
                                       silhouette_info['SilEdgeCands'],
                                       silhouette_info['SilEdgeCandParam'],
                                       silhouette_info['SilCandAssignedFaces'],
                                       silhouette_info['SilCandU'],
                                       global_silhoutte_lambdas,
                                       isCircular=args.find_circular_path)

            U[i].flat = u.flat
            L[i].flat = l.flat

            print '< solve_silhouette_info:', i

        if (args.use_single_process is None or 
            l < args.use_single_process):
            async_exec(solve_silhouette_info,
                       ((i,) for i in xrange(num_instances)),
                       n=num_processes,
                       chunksize=max(1, num_instances / num_processes))
        else:
            map(solve_silhouette_info, xrange(num_instances))

        if args.quit_after_silhouette:
            break

        # solve for the core geometry
        for j in xrange(args.max_restarts):
            print '# solve_core:'
            status = lm_solve_core(T, V, Xg, instScales, K, Xb, y, X, V1, 
                                   core_lambdas,
                                   core_preconditioners,
                                   args.narrowband,
                                   args.uniform_weights,
                                   **solver_options)

            print 'LM Status (%d): ' % status[0], status[1]

            if status[0] not in (0, 4):
                break

        print 'instScales:', instScales
        statuses[0] = status[0]

        # TODO Check if last `V1` can be used or if sequential is better
        # V10 = map(to_shared, V1)
        
        # require sequential optimisation
        V10 = V1

        # solve instances separately
        def solve_instance(i):
            print '> solve_instance:', i
            if i > 0:
                Vp = V10[i-1]
                Xgp = np.zeros((1, 3), dtype=np.float64)
                sp = np.ones((1, 1), dtype=np.float64)
                Xp = np.zeros_like(Vp)
            else:
                Vp = Xgp = sp = Xp = empty3

            for j in xrange(args.max_restarts):
                status = lm_solve_instance(T, V, 
                    Xg[i], instScales[i], 
                    K, Xb,
                    y[i], X[i], 
                    V1[i], U[i], L[i],
                    S[i], SN[i], 
                    Rx[i], Ry[i], 
                    C[i], P[i],
                    Vp, Xgp, sp, Xp,
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

            statuses[i + 1] = status[0]
            print '< solve_instance:', i

        # TODO Check if last `V1` can be used or if sequential is better
        # async_exec(solve_instance, 
        #            ((i,) for i in xrange(num_instances)), 
        #            n=num_processes, 
        #            chunksize=max(1, num_instances / num_processes))
        map(solve_instance, xrange(num_instances))

        print 'instScales:', instScales

        # save the intermediate results
        intermediate_output = os.path.join(output, str(l))
        if not os.path.exists(intermediate_output):
            os.makedirs(intermediate_output)

        scope = args.__dict__.copy()
        scope.update(locals())
        save_state(intermediate_output, **scope)

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

