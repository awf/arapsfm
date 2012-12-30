# solve_multiple_sectioned_arap.py

# Imports
from util.cmdline import * 
from itertools import count, izip

from core_recovery import lm_alt_solvers2 as lm2
from core_recovery import silhouette_global_solver as sil

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

# Multiprocessing

# grouper (from `itertools`)
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

# async_exec
def async_exec(f, iterable, n=None, timeout=5., chunksize=1, verbose=False):
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
            if verbose:
                print 'active_processes:'
                pprint(map(lambda p: p.pid, active_processes))
            for i, p in enumerate(active_processes):
                if verbose:
                    print 'join: %d' % p.pid
                p.join(timeout)

                if not p.is_alive():
                    break
            else:
                continue

            if verbose:
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
    # TODO: Ensure that enough state is saved for each output so that
    # EVERYTHING is recoverable
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    make_path = lambda f: os.path.join(output_dir, f)
    output_file = make_path('core.npz')

    b = Bunch(kwargs)

    print '-> %s' % output_file

    core_dictionary = b.args.__dict__.copy()
    core_dictionary.update(T=b.T, V=b.V)

    pickle_.dump(output_file, core_dictionary)

    for l, index in enumerate(b.indices):
        Q = geometry.path2pos(b.V1[l], b.T, b.L[l], b.U[l])

        output_dictionary = b.args.__dict__.copy()
        output_dictionary.update(T=b.T, V=b.V1[l], L=b.L[l], U=b.U[l], Q=Q,
                                 S=b.S[l], C=b.C[l], P=b.P[l])

        if b.frames[l] is not None:
            output_dictionary['image'] = b.frames[l]

        output_file = make_path('%d.npz' % index)

        print '-> %s' % output_file
        pickle_.dump(output_file, output_dictionary)

# parse_k
def parse_k(k):
    inst_info, basis_info, coeff_info, k_lookup = {}, {}, {}, []
    l, i = 0, 0
    while i < k.shape[0]:
        k_lookup.append(i)

        if k[i] == 0:
            # fixed global rotation
            i += 1
        elif k[i] < 0:
            # instance global rotation
            inst_info.setdefault(k[i+1], []).append(l)
            i += 2
        else:
            # basis rotation
            n = k[i]
            for j in xrange(n):
                basis_info.setdefault(k[i+1+2*j], []).append(l)
                coeff_info.setdefault(k[i+1+2*j+1], []).append(l)
            i += 2*n + 1

        l += 1

    k_lookup = np.asarray(k_lookup, dtype=np.int32)

    return inst_info, basis_info, coeff_info, k_lookup

# main
def main():
    # setup parser
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

    # global rotation basis configuration 
    parser.add_argument('global_rotation_config', type=str)

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

    # evaluate arguments which are passed in as Python strings
    for key in ['indices',
                'lambdas',
                'preconditioners',
                'piecewise_polynomial',
                'solver_options',
                'global_rotation_config']:
        setattr(args, key, eval(getattr(args, key)))

    # setup `solver_options`
    def_solver_options = dict(maxIterations=20, 
                              gradientThreshold=1e-6,
                              updateThreshold=1e-6,
                              improvementThreshold=1e-6,
                              verbosenessLevel=1)
    def_solver_options.update(args.solver_options)
    args.solver_options = def_solver_options

    # handle `num_processes`, `output`, and `frames`
    if args.num_processes < 0:
        args.num_processes = mp.cpu_count()

    if args.output is not None and '{default}' in args.output:
        make_str = lambda a: ",".join('%.4g' % a_ for a_ in a)
        mesh_file = os.path.split(args.mesh)[1]

        if len(args.indices) > 10:
            indices_str = (make_str(args.indices[:5]) + '---' +
                           make_str(args.indices[-5:]))
        else:
            indices_str = make_str(args.indices)

        default_directory = '%s_%s_%s_%s_%s' % (
            os.path.splitext(mesh_file)[0],
            indices_str,
            make_str(args.lambdas), 
            make_str(args.preconditioners),
            make_str(args.piecewise_polynomial))

        args.output = args.output.format(default=default_directory)

    if args.frames is not None:
        args.frames = [args.frames % d for d in args.indices]
    else:
        args.frames = [None for d in args.indices]

    args.piecewise_polynomial = np.require(args.piecewise_polynomial,
                                           dtype=np.float64)


    # output arguments
    pprint(args.__dict__)

    # setup output directory
    print '> %s' % args.output
    if args.output is not None and not os.path.exists(args.output):
        print 'Creating directory:', args.output
        os.makedirs(args.output)

    # parse argments and load data
    load_instance_variables = lambda *a: load_formatted(args.indices, *a)

    V = load_input_geometry(args.core_initialisation, 
                            args.use_linear_transform)
    T = load_input_mesh(args.mesh)

    C, P = load_instance_variables(args.user_constraints, 'C', 'P')
    num_instances = len(C)

    S, SN = load_instance_variables(args.silhouette, 'S', 'SN')

    silhouette_info = np.load(args.core_silhouette_info)
    print '<- %s' % args.core_silhouette_info

    (R,) = load_instance_variables(args.spillage, 'R')
    Rx, Ry = map(list, izip(*R))

    # transfer vertices to shared memory
    V_flat = V.flat
    V = mparray.empty_like(V)
    V.flat = V_flat

    def make_shared(s, n, dtype=np.float64, value=0.):
        b = []
        for i in xrange(n):
            a = mparray.empty(s, dtype=dtype)
            a.fill(value)
            b.append(a)
        return b

    V1 = make_shared(V.shape, num_instances)
    V1[0].flat = V.flat

    # parse `kg` to get the configuration of all of the global rotations
    kg = np.asarray(args.global_rotation_config, dtype=np.int32)
    kg_inst, kg_basis, kg_coeff, kg_lookup = parse_k(kg)

    # allocate global basis rotations
    Xg = make_shared((1, 3), len(kg_inst))
    Xgb = make_shared((1, 3), len(kg_basis))
    yg = make_shared((1, 1), len(kg_coeff), value=1.)

    # parse `ki` to get the configuration of all rotations at each instance
    ki = np.require(np.load(args.arap_sections)['k2'], dtype=np.int32)
    ki_inst, ki_basis, ki_coeff, ki_lookup = parse_k(ki)

    X = make_shared((len(ki_inst), 3), num_instances)
    Xb = mparray.empty((len(ki_basis), 3), dtype=np.float64)
    y = make_shared((len(ki_coeff), 1), num_instances, value=1.)

    # instance scales (s)
    instScales = make_shared((1, 1), num_instances)

    # initialise silhouette (U, L)
    U = [mparray.empty((s.shape[0], 2), np.float64) for s in S]
    L = [mparray.empty(s.shape[0], np.int32) for s in S]

    # solve for initialisations without the basis rotations (global or local)
    initialisation_lambdas = np.r_[
        args.lambdas[5],   # projection
        args.lambdas[3],   # as-rigid-as-possible
        args.lambdas[7],   # temporal ARAP penalty
        args.lambdas[8],   # global rotations regularisation
        args.lambdas[9],   # global scale regularisation
        args.lambdas[10]]  # frame-to-frame rotations regularisation

    # initialisation preconditioners
    initialisation_preconditioners = np.r_[args.preconditioners[0], # V
                                           args.preconditioners[1], # X
                                           args.preconditioners[2]] # s

    # temporary variables
    iX = np.zeros_like(V)
    empty3 = np.array(tuple(), dtype=np.float64).reshape(0, 3)
    iXg = np.zeros((1, 3), dtype=np.float64)

    initialisation_solver_options = args.solver_options.copy()
    initialisation_solver_options['maxIterations'] = 50 

    # perform initialisation
    V1[0].flat = V.flat

    # solve for V1[:]
    def solve_initialisation(i):
        print '[-1] %d:' % i

        if i > 0:
            # NOTE: use iXg from previous frame
            V1[i].flat = V1[i-1].flat
            instScales[i][0] = instScales[i-1][0]

            Vp = V1[i-1]
            Xgp = np.zeros((1, 3), dtype=np.float64)
            sp = np.ones((1, 1), dtype=np.float64)
            Xp = np.zeros_like(Vp)
        else:
            Vp = Xgp = sp = Xp = empty3

        for j in xrange(args.max_restarts):
            status = lm2.solve_two_source_arap_proj(T, 
                                                    V, instScales[i], iXg, iX,
                                                    Vp, sp, Xgp, Xp,
                                                    V1[i], 
                                                    C[i], P[i],
                                                    initialisation_lambdas,
                                                    initialisation_preconditioners,
                                                    args.uniform_weights,
                                                    **initialisation_solver_options)

            print status[1]
            if status[0] not in (0, 4):
                break

    map(solve_initialisation, xrange(num_instances))

    # setup solving for V, X, ... and V1
    core_lambdas = np.r_[args.lambdas[3],    # as-rigid-as-possible
                         args.lambdas[6]]    # laplacian

    core_preconditioners = np.r_[args.preconditioners[0], # V
                                 args.preconditioners[1], # X/Xg
                                 args.preconditioners[2], # s
                                 args.preconditioners[4]] # y

    instance_lambdas = np.r_[args.lambdas[3],    # as-rigid-as-possible
                             args.lambdas[1:3],  # silhouette
                             args.lambdas[4],    # spillage
                             args.lambdas[5],    # projection
                             args.lambdas[7],    # temporal ARAP penalty
                             args.lambdas[8],    # global rotations penalty
                             args.lambdas[9],    # global scale penalty 
                             args.lambdas[10]]   # regular rotations penalty

    instance_preconditioners = np.r_[args.preconditioners[0], # V
                                     args.preconditioners[1], # X/Xg
                                     args.preconditioners[2], # s
                                     args.preconditioners[3], # U
                                     args.preconditioners[4]] # y


    core_solver_options = args.solver_options.copy()
    core_solver_options['verbosenessLevel'] = 1

    instance_solver_options = args.solver_options.copy()
    instance_solver_options['verbosenessLevel'] = 1

    silhouette_lambdas = np.require(args.lambdas[:3], dtype=np.float64)

    for l in xrange(args.outer_loops):
        print '[%d] `solve_core`:' % l

        for j in xrange(args.max_restarts):
            status = lm2.solve_core(T, V, instScales, 
                                    kg, Xgb, yg, Xg,
                                    ki, Xb, y, X, V1,
                                    core_lambdas,
                                    core_preconditioners,
                                    args.narrowband,
                                    args.uniform_weights,
                                    **core_solver_options)
            print status[1]
            if status[0] not in (0, 4):
                break

        # update_silhouette
        def update_silhouette(i):
            print '[%d] `update_silhouette`:' % l, i
            
            D = instScales[i] * silhouette_info['SilCandDistances']

            u, l_ = sil.shortest_path_solve(
                V1[i], T, S[i], SN[i], D,
                silhouette_info['SilEdgeCands'],
                silhouette_info['SilEdgeCandParam'],
                silhouette_info['SilCandAssignedFaces'],
                silhouette_info['SilCandU'],
                silhouette_lambdas,
                isCircular=args.find_circular_path)

            U[i].flat = u.flat
            L[i].flat = l_.flat

        if (args.use_single_process is None or 
            l < args.use_single_process):
            async_exec(update_silhouette,
                       ((i,) for i in xrange(num_instances)),
                       n=args.num_processes,
                       chunksize=max(1, num_instances / args.num_processes),
                       verbose=True)
        else:
            map(update_silhouette, xrange(num_instances))

        # solve_instance
        def solve_instance(i):
            print '[%d] `solve_instance`:' % l, i

            if i > 0:
                Vp = V1[i-1]
                Xgp = np.zeros((1, 3), dtype=np.float64)
                sp = np.ones((1, 1), dtype=np.float64)
                Xp = np.zeros_like(Vp)
            else:
                Vp = Xgp = sp = Xp = empty3

            m = kg_lookup[i]
            n = kg[m]

            Xgbi = []
            ygi = []
            Xgi = empty3

            if n == 0:  # fixed global rotation
                pass
            elif n == -1:   # independent global rotation
                # NOTE: Potential problem here if multiple
                # instances share the same global rotation it will be changed
                # as each instance is processed
                Xgi = Xg[kg[m+1]]
            else:
                # n-basis rotation
                # NOTE: Potential problem here if multiple
                # instances share the same basis coefficient as it will be
                # changed as each instance is processed
                for ii in xrange(n):
                    Xgbi.append(Xgb[kg[m + 1 + 2*ii]])
                    ygi.append(yg[kg[m + 1 + 2*ii + 1]])
                
            fixed_scale = i == 0 

            for j in xrange(args.max_restarts):
                status = lm2.solve_instance(T, V, instScales[i],
                                            n, Xgbi, ygi, Xgi,
                                            ki, Xb, y[i], X[i],
                                            Vp, sp, Xgp, Xp,
                                            V1[i], U[i], L[i],
                                            S[i], SN[i],
                                            Rx[i], Ry[i], 
                                            C[i], P[i],
                                            instance_lambdas,
                                            instance_preconditioners,
                                            args.piecewise_polynomial,
                                            args.narrowband,
                                            args.uniform_weights,
                                            fixed_scale,
                                            **instance_solver_options)


                print status[1]

                if status[0] not in (0, 4):
                    break

        map(solve_instance, xrange(num_instances))

        intermediate_output = os.path.join(args.output, str(l))
        if not os.path.exists(intermediate_output):
            os.makedirs(intermediate_output)

        scope = args.__dict__.copy()
        scope.update(locals())
        save_state(intermediate_output, **scope)

    print 'instScales:', instScales
    print 'kg:', kg
    print 'Xgb:', Xgb
    print 'yg:', yg
    print 'Xg:', Xg

    if args.output is not None:
        scope = args.__dict__.copy()
        scope.update(locals())
        save_state(args.output, **scope)

if __name__ == '__main__':
    main()

