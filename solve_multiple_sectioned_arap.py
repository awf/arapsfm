# solve_multiple_sectioned_arap.py

# Imports
from util.cmdline import * 
from itertools import count, izip

from core_recovery import lm_alt_solvers2 as lm
from core_recovery.arap.sectioned import parse_k
from core_recovery.silhouette import solve_silhouette

from operator import add
from time import time
from misc.bunch import Bunch
from misc.numpy_ import mparray

from scipy.linalg import norm
    
# Utilities

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
    core_dictionary.update(T=b.T, V=b.V, 
                           s=b.instScales, 
                           kg=b.kg, Xgb=b.Xgb, yg=b.yg, Xg=b.Xg,
                           ki=b.ki, Xb=b.Xb, y=b.y, X=b.X)

    pickle_.dump(output_file, core_dictionary)

    for l, index in enumerate(b.indices):
        Q = geometry.path2pos(b.V1[l], b.T, b.L[l], b.U[l])

        m = b.kg_lookup[l]
        n = b.kg[m]

        if n == 0:
            Xg = np.r_[0., 0., 0.].reshape(1, 3)
        elif n == -1:
            Xg = b.Xg[b.kg[m + 1]]
        else:
            Xg = reduce(add, 
                        (b.yg[b.kg[m + 1 + 2*ii + 1]] * 
                         b.Xgb[b.kg[m + 1 + 2*ii]] for ii in xrange(n))).reshape(1, 3)

        output_dictionary = b.args.__dict__.copy()
        output_dictionary.update(T=b.T, V=b.V1[l], 
                                 S=b.S[l], Q=Q, L=b.L[l], U=b.U[l],
                                 C=b.C[l], P=b.P[l],
                                 s=b.instScales[l], Xg=Xg,
                                 ki=b.ki, Xb=b.Xb, y=b.y[l], X=b.X[l])

        if b.frames[l] is not None:
            output_dictionary['image'] = b.frames[l]

        output_file = make_path('%d.npz' % index)

        print '-> %s' % output_file
        pickle_.dump(output_file, output_dictionary)

# safe_index_list
def safe_index_list(l, i):
    if i not in l:
        l.append(i)

    return l.index(i)

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

    parser.add_argument('output', type=str)

    # visualisation
    parser.add_argument('--frames', type=str, default=None)

    # solving
    parser.add_argument('--solver_options', type=str, default=None)
    parser.add_argument('--narrowband', type=int, default=3)
    parser.add_argument('--no_uniform_weights', dest='uniform_weights',
                        default=True,
                        action='store_false')
    parser.add_argument('--max_restarts', type=int, default=5)
    parser.add_argument('--outer_loops', type=int, default=5)
    parser.add_argument('--candidate_radius', type=float, default=None)

    # optional
    parser.add_argument('--quit_after_silhouette',
                        action='store_true',
                        default=False)

    parser.add_argument('--use_z_components', type=str, default='None')
    parser.add_argument('--use_z_components_for_init_only', 
                        action='store_true',
                        default=False)

    parser.add_argument('--initial_Xgb', type=str, default='None')
    parser.add_argument('--fixed_Xgb', 
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
                'global_rotation_config',
                'use_z_components',
                'initial_Xgb'
                ]:
        setattr(args, key, eval(getattr(args, key)))

    # setup `solver_options`
    def_solver_options = dict(maxIterations=20, 
                              gradientThreshold=1e-6,
                              updateThreshold=1e-6,
                              improvementThreshold=1e-6,
                              verbosenessLevel=1)
    def_solver_options.update(args.solver_options)
    args.solver_options = def_solver_options

    if '{default}' in args.output:
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

    if args.initial_Xgb is not None:
        args.initial_Xgb = np.asarray(args.initial_Xgb, dtype=np.float64)

    # output arguments
    pprint(args.__dict__)

    # setup output directory
    print '> %s' % args.output
    if args.output is not None and not os.path.exists(args.output):
        print 'Creating directory:', args.output
        os.makedirs(args.output)

    # parse argments and load data
    load_instance_variables = lambda *a: load_formatted(args.indices, *a, verbose=True)

    V0 = load_input_geometry(args.core_initialisation, 
                             args.use_linear_transform)
    T = load_input_mesh(args.mesh)

    C, P = load_instance_variables(args.user_constraints, 'C', 'P')
    num_instances = len(C)

    S, SN = load_instance_variables(args.silhouette, 'S', 'SN')

    silhouette_info = np.load(args.core_silhouette_info)
    print '<- %s' % args.core_silhouette_info

    (R,) = load_instance_variables(args.spillage, 'R')
    Rx, Ry = map(list, izip(*R))

    # initial projection constraints are a direct shallow copy
    Pi = P[:]

    # augment projection constraints with z-axis constraints for the required
    # frames
    if args.use_z_components is not None:
        assert len(args.use_z_components) == num_instances

        Vi_cache = {}
        for i, index in enumerate(args.use_z_components):
            if index < 0:
                continue

            if index not in Vi_cache:
                Vi_cache[index] = load_input_geometry(
                    args.user_constraints % args.indices[index], 
                    args.use_linear_transform)
            
            Vi = Vi_cache[index]
            pz = Vi[C[i], 2]
            Pi[i] = np.require(np.c_[P[i], pz[:, np.newaxis]], np.float64,
                               requirements='C')
        del Vi_cache

        if not args.use_z_components_for_init_only:
            P = Pi
                                
    # transfer vertices to shared memory 
    # (`V0` is fixed, no need for shared memory)
    V = mparray.empty_like(V0)
    V.flat = V0.flat

    def make_shared(s, n, dtype=np.float64, value=0.):
        b = []
        for i in xrange(n):
            a = mparray.empty(s, dtype=dtype)
            a.fill(value)
            b.append(a)
        return b

    # reset `V0` to the origin and initialise `d0`
    d0 = np.atleast_2d(np.mean(V0, axis=0))
    d0 = np.require(d0, dtype=np.float64, 
                    requirements='C')
    V0 -= d0

    # initialise `s0`, `xg0`
    s0 = np.array([[1.]], dtype=np.float64)
    xg0 = np.array([[0., 0., 0.]], dtype=np.float64)

    # initialise `V1`
    V1 = make_shared(V.shape, num_instances)
    V1[0].flat = V.flat

    # parse `kg` to get the configuration of all of the global rotations
    kg = np.asarray(args.global_rotation_config, dtype=np.int32)
    kg_inst, kg_basis, kg_coeff, kg_lookup = parse_k(kg)

    # allocate global basis rotations
    Xg = make_shared((1, 3), len(kg_inst))
    Xgb = make_shared((1, 3), len(kg_basis))
    yg = make_shared((1, 1), len(kg_coeff), value=1.)

    if args.initial_Xgb is not None:
        assert args.initial_Xgb.shape == (len(kg_basis), 3)
        for i, xgb in enumerate(args.initial_Xgb):
            Xgb[i].flat = xgb.flat

    # parse `ki` to get the configuration of all rotations at each instance
    ki = np.require(np.load(args.arap_sections)['k2'], dtype=np.int32)
    ki_inst, ki_basis, ki_coeff, ki_lookup = parse_k(ki)

    X = make_shared((len(ki_inst), 3), num_instances)
    Xb = mparray.zeros((len(ki_basis), 3), dtype=np.float64)
    y = make_shared((len(ki_coeff), 1), num_instances, value=1.)

    # instance scales (s)
    instScales = make_shared((1, 1), num_instances)

    # initialise silhouette (U, L)
    U = [mparray.empty((s.shape[0], 2), np.float64) for s in S]
    L = [mparray.empty(s.shape[0], np.int32) for s in S]

    # temporary variables
    iX = np.zeros_like(V)
    empty3 = np.array(tuple(), dtype=np.float64).reshape(0, 3)
    iXg = np.zeros((1, 3), dtype=np.float64)

    # setup lambdas and preconditioners
    initialisation_lambdas = np.r_[
        args.lambdas[5],   # projection
        args.lambdas[3],   # as-rigid-as-possible
        args.lambdas[7],   # temporal ARAP penalty
        args.lambdas[8],   # global rotations regularisation
        args.lambdas[9],   # global scale regularisation
        args.lambdas[10]]  # frame-to-frame rotations regularisation

    initialisation_preconditioners = np.r_[args.preconditioners[0], # V
                                           args.preconditioners[1], # X
                                           args.preconditioners[2]] # s

    core_lambdas = np.r_[args.lambdas[3],    # as-rigid-as-possible
                         args.lambdas[8],    # global rotations penalty
                         args.lambdas[9],    # global scale penalty 
                         args.lambdas[10],   # frame-to-frame rotations regularisation
                         args.lambdas[6],    # laplacian (NOTE should be scaled by `len(V1)`)
                         args.lambdas[11],   # rigid-registration of core to initialisation
                         len(V1) * args.lambdas[12]]   # penalty of the global rotation of the registration

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
                             args.lambdas[10]]   # frame-to-frame rotations regularisation

    instance_preconditioners = np.r_[args.preconditioners[0], # V
                                     args.preconditioners[1], # X/Xg
                                     args.preconditioners[2], # s
                                     args.preconditioners[3], # U
                                     args.preconditioners[4]] # y

    silhouette_lambdas = np.require(args.lambdas[:3], dtype=np.float64)

    core_solver_options = args.solver_options.copy()
    # core_solver_options['verbosenessLevel'] = 1
    instance_solver_options = args.solver_options.copy()
    # instance_solver_options['verbosenessLevel'] = 1
    init_solver_options = args.solver_options.copy()
    init_solver_options['maxIterations'] = max(50, 
        init_solver_options['maxIterations'])

    # setup solving functions

    # solve_initialisation
    def solve_initialisation(i):
        print '[-1] `solve_initialisation` (%d):' % i

        t1 = time()

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
            status = lm.solve_two_source_arap_proj(T, 
                                                   V, instScales[i], iXg, iX,
                                                   Vp, sp, Xgp, Xp,
                                                   V1[i], 
                                                   C[i], Pi[i],
                                                   initialisation_lambdas,
                                                   initialisation_preconditioners,
                                                   args.uniform_weights,
                                                   **init_solver_options)

            print status[1]
            if status[0] not in (0, 4):
                break

        t2 = time()

        print '[-1] `solve_initialisation` (%d): %.3fs' % (i, t2 - t1)

    # update_silhouette
    def update_silhouette(i):
        print '[%d] `update_silhouette` (%d):' % (l, i)
        
        t1 = time()

        u, l_ = solve_silhouette(
            V1[i], T, S[i], SN[i], 
            silhouette_info['SilCandDistances'],
            silhouette_info['SilEdgeCands'],
            silhouette_info['SilEdgeCandParam'],
            silhouette_info['SilCandAssignedFaces'],
            silhouette_info['SilCandU'],
            silhouette_lambdas,
            radius=args.candidate_radius,
            verbose=True)

        U[i].flat = u.flat
        L[i].flat = l_.flat

        t2 = time()
        print '[%d] `update_silhouette (%d)`: %.3fs' % (l, i, t2 - t1)

    # solve_instance
    def solve_instance(i):
        print '[%d] `solve_instance` (%d):' % (l, i)

        t1 = time()

        if i > 0:
            Vm1 = V1[i-1]
            sp = np.ones((1, 1), dtype=np.float64)
            Xgp = np.zeros((1, 3), dtype=np.float64)
            Xp = np.zeros_like(Vm1)
        else:
            Vm1 = sp = Xgp = Xp = empty3

        if i > 1:
            sm1 = [instScales[i-1], instScales[i-2]]
            ym1 = [y[i-1], y[i-2]]
            Xm1 = [X[i-1], X[i-2]]
            subproblem = (i, i - 1, i - 2)
        else:
            sm1 = []
            ym1 = []
            Xm1 = []
            subproblem = (i,)

        used_Xgb, used_yg, used_Xg = [], [], []
        kgi = []

        for ii in subproblem:
            m = kg_lookup[ii]

            n = kg[m]
            kgi.append(n)

            if n == 0:
                # fixed global rotation
                pass
            elif n == -1:
                # independent global rotation
                # NOTE: Potential problem here if multiple
                # instances share the same global rotation it will be changed
                # as each instance is processed
                kgi.append(safe_index_list(used_Xg, kg[m+1]))
            else:
                # n-basis rotation
                # NOTE: Potential problem here if multiple
                # instances share the same basis coefficient as it will be
                # changed as each instance is processed
                for ii in xrange(n):
                    kgi.append(safe_index_list(used_Xgb, 
                                               kg[m + 1 + 2*ii]))
                    kgi.append(safe_index_list(used_yg, 
                                               kg[m + 1 + 2*ii + 1]))

        kgi = np.require(kgi, dtype=np.int32)
        Xgbi = map(lambda i: Xgb[i], used_Xgb)
        ygi = map(lambda i: yg[i], used_yg)
        Xgi = map(lambda i: Xg[i], used_Xg)

        # fixed_scale = i == 0 
        fixed_scale = False
        fixed_global_rotation = True

        for j in xrange(args.max_restarts):
            print ' [%d] s[%d]: %.3f' % (j, i, instScales[i])

            status = lm.solve_instance(T, V, instScales[i],
                                       kgi, Xgbi, ygi, Xgi,
                                       ki, Xb, y[i], X[i], ym1, Xm1,
                                       Vm1, sp, Xgp, Xp, sm1,
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
                                       fixed_global_rotation,
                                       **instance_solver_options)


            print status[1]

            print ' [%d] s[%d]: %.3f' % (j, i, instScales[i])

            if status[0] not in (0, 4):
                break

        t2 = time()

        print '[%d] `solve_instance` (%d): %.3fs' % (l, i, t2 - t1)

    # start complete timing
    t1_complete = time()

    # perform initialisation
    V1[0].flat = V.flat

    map(solve_initialisation, xrange(num_instances))

    for l in xrange(args.outer_loops):
        # update_silhouette immediately if `quit_after_silhouette`
        if args.quit_after_silhouette:
            map(update_silhouette, xrange(num_instances))
            break

        # solve_core
        print '[%d] `solve_core`:' % l

        t1 = time()
        for j in xrange(args.max_restarts):
            status = lm.solve_core(T, V, instScales, 
                                   kg, Xgb, yg, Xg,
                                   ki, Xb, y, X, V1,
                                   V0, s0, xg0, d0,
                                   core_lambdas,
                                   core_preconditioners,
                                   args.uniform_weights,
                                   args.fixed_Xgb,
                                   **core_solver_options)
            print status[1]
            if status[0] not in (0, 4):
                break
        t2 = time()
        print '[%d] `solve_core`: %.3fs' % (l, t2 - t1)

        print 's0:', np.squeeze(s0)
        l_xg0 = norm(xg0[0][0])
        print 'xg0 (%.3f / %.1f):' % (l_xg0, np.rad2deg(l_xg0)), np.squeeze(xg0)
        print 'd0:', np.squeeze(d0)

        print 'scales:', np.squeeze(instScales)

        print 'kg:'
        pprint(kg)
        print 'Xgb:'
        pprint(Xgb)
        print 'yg:'
        pprint(yg)
        print 'Xg:'
        pprint(Xg)
        print 'Xb:'
        pprint(Xb)

        # save update from the core
        intermediate_output = os.path.join(args.output, str(l) + 'A')
        if not os.path.exists(intermediate_output):
            os.makedirs(intermediate_output)

        scope = args.__dict__.copy()
        scope.update(locals())
        save_state(intermediate_output, **scope)

        # don't run `solve_instance` on the last iteration
        if l >= args.outer_loops - 1:
            break

        # update the silhouette
        map(update_silhouette, xrange(num_instances))

        # update each instance
        map(solve_instance, xrange(num_instances))

        intermediate_output = os.path.join(args.output, str(l))
        if not os.path.exists(intermediate_output):
            os.makedirs(intermediate_output)

        scope = args.__dict__.copy()
        scope.update(locals())
        save_state(intermediate_output, **scope)

    t2_complete = time()

    print '[+] time taken: %.3fs' % (t2_complete - t1_complete)

    scope = args.__dict__.copy()
    scope.update(locals())
    save_state(args.output, **scope)

if __name__ == '__main__':
    main()

