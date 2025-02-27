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

from geometry import axis_angle

# Utilities

# Main

# save_state
def save_state(_output_dir, **kwargs):
    # TODO: Ensure that enough state is saved for each output so that
    # EVERYTHING is recoverable
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)

    make_path = lambda f: os.path.join(_output_dir, f)
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
            Xg = reduce(safe_ax_add,
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

# safe_ax_add
def safe_ax_add(x, y):
    return axis_angle.axAdd(x.ravel(), y.ravel())

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

    parser.add_argument('--initial_Xb', type=str, default='None')

    parser.add_argument('--silhouette_projection_lambdas', type=str,
                        default='None')

    parser.add_argument('--save_optimisation_progress', type=str, default=None)

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
                'initial_Xgb',
                'initial_Xb',
                'silhouette_projection_lambdas',
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

    if args.initial_Xb is not None:
        args.initial_Xb = np.asarray(args.initial_Xb, dtype=np.float64)

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

    # initialise `V1`
    V1 = make_shared(V.shape, num_instances)
    V1[0].flat = V.flat

    # parse `kg` to get the configuration of all of the global rotations
    kg = np.asarray(args.global_rotation_config, dtype=np.int32)
    kg_inst, kg_basis, kg_coeff, kg_lookup = parse_k(kg)

    # allocate global basis rotations
    Xg = make_shared((1, 3), len(kg_inst))
    Xgb = make_shared((1, 3), len(kg_basis))
    yg = make_shared((1, 1), len(kg_coeff), value=0.)
    
    if args.initial_Xgb is not None:
        if args.initial_Xgb.shape != (len(kg_basis), 3):
            raise ValueError('args.initial_Xgb.shape != (%d, 3)' %
                             len(kg_basis))
        for i, xgb in enumerate(args.initial_Xgb):
            Xgb[i].flat = xgb.flat

    # parse `ki` to get the configuration of all rotations at each instance
    ki = np.require(np.load(args.arap_sections)['k2'], dtype=np.int32)
    ki_inst, ki_basis, ki_coeff, ki_lookup = parse_k(ki)

    X = make_shared((len(ki_inst), 3), num_instances)
    Xb = mparray.zeros((len(ki_basis), 3), dtype=np.float64)
    y = make_shared((len(ki_coeff), 1), num_instances, value=0.)

    if args.initial_Xb is not None:
        if args.initial_Xb.shape != (len(ki_basis), 3):
            raise ValueError('args.initial_Xb.shape != (%d, 3)' %
                             len(ki_basis))

        for i, xb in enumerate(args.initial_Xb):
            Xb[i].flat = xb.flat

    # instance scales (s)
    instScales = make_shared((1, 1), num_instances, value=1.)

    # initialise silhouette (U, L)
    U = [mparray.empty((s.shape[0], 2), np.float64) for s in S]
    L = [mparray.empty(s.shape[0], np.int32) for s in S]

    # temporary variables
    empty3 = np.array(tuple(), dtype=np.float64).reshape(0, 3)

    # setup lambdas and preconditioners
    core_lambdas = np.r_[args.lambdas[3],    # as-rigid-as-possible
                         args.lambdas[6],    # global scale acceleration penalty
                         args.lambdas[7],    # global rotations acceleration penalty
                         args.lambdas[8],    # frame-to-frame acceleration penalty
                         args.lambdas[5],    # laplacian 
                         args.lambdas[9],    # global scale velocity penalty
                         args.lambdas[10],   # global rotations velocity penalty
                         args.lambdas[11]]   # frame-to-frame velocity penalty


    core_preconditioners = np.r_[args.preconditioners[0], # V
                                 args.preconditioners[1], # X/Xg
                                 args.preconditioners[2], # s
                                 args.preconditioners[4]] # y

    instance_lambdas = np.r_[args.lambdas[3],    # as-rigid-as-possible
                             args.lambdas[1:3],  # silhouette
                             args.lambdas[4],    # projection
                             args.lambdas[6],    # global scale acceleration penalty
                             args.lambdas[7],    # global rotations acceleration penalty
                             args.lambdas[8],    # frame-to-frame acceleration penalty
                             args.lambdas[9],    # global scale velocity penalty
                             args.lambdas[10],   # global rotations velocity penalty
                             args.lambdas[11]]   # frame-to-frame velocity penalty

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

    if args.save_optimisation_progress is not None:
        save_optimisation_root = os.path.join(
            args.output, args.save_optimisation_progress)

        if not os.path.exists(save_optimisation_root):
            os.makedirs(save_optimisation_root)

    # make_solve_instance_callback
    def make_solve_instance_callback(l, output_path):
        all_states = []

        # solve_instance_callback
        def solve_instance_callback(solver_iteration, computeDerivatives):
            if not computeDerivatives:
                return

            m = kg_lookup[l]
            n = kg[m]

            if n == 0:
                _Xg = np.r_[0., 0., 0.].reshape(1, 3)
            elif n == -1:
                _Xg = Xg[kg[m + 1]]
            else:
                _Xg = reduce(safe_ax_add,
                            (yg[kg[m + 1 + 2*ii + 1]] * 
                             Xgb[kg[m + 1 + 2*ii]] for ii in xrange(n))).reshape(1, 3)

            # NOTE copies are required because it is not written to disk
            # immediately
            # NOTE assuming `Xb` is not being updated (true in the alternation
            # scheme)
            output_dict = dict(T=T, V=V1[l].copy(), C=C[l], P=P[l],
                               s=instScales[l].copy(), Xg=_Xg.copy(),
                               ki=ki, Xb=Xb, y=y[l].copy(), X=X[l].copy(),
                               solver_iteration=solver_iteration)

            if args.frames[l] is not None:
                output_dict['image'] = args.frames[l]

            if l >= 0:
                Q = geometry.path2pos(V1[l], T, L[l], U[l])
                output_dict.update(S=S[l], Q=Q, L=L[l].copy(), U=U[l].copy())

            all_states.append(output_dict)

        # save_solve_instance_states
        def save_solve_instance_states():
            print '-> %s' % output_path
            pickle_.dump(output_path, 
                         {'has_states' : True, 'states' : all_states})

        return solve_instance_callback, save_solve_instance_states

    # make_solve_core_callback
    def make_solve_core_callback(output_path):
        all_states = []

        # solve_core_callback
        def solve_core_callback(solver_iteration, computeDerivatives):
            if not computeDerivatives:
                return

            copyl = lambda l: map(np.copy, l)

            core_dictionary = dict(T=T, V=V.copy(), 
                                   s=copyl(instScales), 
                                   kg=kg, Xgb=copyl(Xgb), 
                                   yg=copyl(yg), Xg=copyl(Xg),
                                   ki=ki, Xb=Xb.copy(), y=copyl(y), X=copyl(X),
                                   solver_iteration=solver_iteration)

            all_states.append(core_dictionary)
            
        # save_solve_core_states
        def save_solve_core_states():
            print '-> %s' % output_path
            pickle_.dump(output_path, 
                         {'has_states' : True, 'states' : all_states})

        return solve_core_callback, save_solve_core_states

    # setup solving functions

    # update_silhouette
    def update_silhouette(i):
        print '[%d] `update_silhouette` (%d):' % (l, i)
        
        lambdas = silhouette_lambdas.copy()
        if (args.silhouette_projection_lambdas is not None and
            l < len(args.silhouette_projection_lambdas)):
            lambdas[1] = args.silhouette_projection_lambdas[l]

        t1 = time()

        u, l_ = solve_silhouette(
            V1[i], T, S[i], SN[i], 
            silhouette_info['SilCandDistances'],
            silhouette_info['SilEdgeCands'],
            silhouette_info['SilEdgeCandParam'],
            silhouette_info['SilCandAssignedFaces'],
            silhouette_info['SilCandU'],
            lambdas,
            radius=args.candidate_radius,
            verbose=True)

        U[i].flat = u.flat
        L[i].flat = l_.flat

        t2 = time()
        print '[%d] `update_silhouette (%d)`: %.3fs' % (l, i, t2 - t1)

    # solve_instance
    def solve_instance(l, i, fixed_global_rotation=True, 
                       fixed_scale=False,
                       no_silhouette=False):
        print '[%d] `solve_instance` (%d):' % (l, i)

        t1 = time()

        if i > 0:
            sm1 = [instScales[i-1]]
            ym1 = [y[i-1]]
            Xm1 = [X[i-1]]

            subproblem = (i, i - 1)
        else:
            sm1 = []
            ym1 = []
            Xm1 = []
            subproblem = (i,)

        if i > 1:
            sm1.append(instScales[i-2])
            ym1.append(y[i-2])
            Xm1.append(X[i-2])
            subproblem = subproblem + (i - 2,)

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

        lambdas = instance_lambdas.copy()
        if (args.silhouette_projection_lambdas is not None and
            l < len(args.silhouette_projection_lambdas)):
            lambdas[2] = args.silhouette_projection_lambdas[l]

        if args.save_optimisation_progress is not None:
            output_path = os.path.join(
                args.output, 
                args.save_optimisation_progress, 
                str(l), 
                '%d.dat' % i)

            callback, write_states = make_solve_instance_callback(
                i, output_path)

        else:
            callback = None

        for j in xrange(args.max_restarts):
            print ' [%d] s[%d]: %.3f' % (j, i, instScales[i])

            status = lm.solve_instance(T, V, instScales[i],
                                       kgi, Xgbi, ygi, Xgi,
                                       ki, Xb, y[i], X[i], ym1, Xm1, sm1,
                                       V1[i], U[i], L[i],
                                       S[i], SN[i],
                                       C[i], P[i],
                                       lambdas,
                                       instance_preconditioners,
                                       args.piecewise_polynomial,
                                       args.narrowband,
                                       args.uniform_weights,
                                       fixed_scale,
                                       fixed_global_rotation,
                                       no_silhouette,
                                       callback=callback,
                                       **instance_solver_options)


            print status[1]

            print ' [%d] s[%d]: %.3f' % (j, i, instScales[i])

            if status[0] not in (0, 4):
                break

        t2 = time()

        if callback is not None:
            write_states()

        print '[%d] `solve_instance` (%d): %.3fs' % (l, i, t2 - t1)

    # start complete timing
    t1_complete = time()

    # perform initialisation
        
    V1[0].flat = V.flat

    if args.save_optimisation_progress is not None:
        output_dir = os.path.join(
            args.output, 
            args.save_optimisation_progress, 
            '-1')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Improved initialisation using basis 
    for i in xrange(num_instances):
        if i > 0:
            V1[i].flat = V1[i-1].flat

        # NOTE Because global rotations are initialised at 0., allowing the
        # scale to change at the same time can cause the scale to want to pass
        # through 0., causing a degenerate solution. This is only a problem on
        # motion with large changes in speed, but it is good to fix the scale 
        # BEFORE solving with the scale free as well
        solve_instance(-1, i, 
                       fixed_global_rotation=False,
                       fixed_scale=True,
                       no_silhouette=True)

        solve_instance(-1, i, 
                       fixed_global_rotation=False,
                       fixed_scale=False,
                       no_silhouette=True)

    for l in xrange(args.outer_loops):
        # update_silhouette immediately if `quit_after_silhouette`
        if args.quit_after_silhouette:
            map(update_silhouette, xrange(num_instances))
            break

        if args.save_optimisation_progress is not None:
            output_dir = os.path.join(
                args.output, 
                args.save_optimisation_progress, 
                str(l))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(output_dir, 'core.dat')

            callback, write_states = make_solve_core_callback(output_path)

        else:
            callback = None

        # solve_core
        print '[%d] `solve_core`:' % l

        t1 = time()
        for j in xrange(args.max_restarts):
            status = lm.solve_core(T, V, instScales, 
                                   kg, Xgb, yg, Xg,
                                   ki, Xb, y, X, V1,
                                   core_lambdas,
                                   core_preconditioners,
                                   args.uniform_weights,
                                   args.fixed_Xgb,
                                   callback=callback,
                                   **core_solver_options)
            print status[1]
            if status[0] not in (0, 4):
                break

        t2 = time()

        if callback is not None:
            write_states()

        print '[%d] `solve_core`: %.3fs' % (l, t2 - t1)
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
        map(lambda i: solve_instance(l, i, 
                                     fixed_global_rotation=True,
                                     fixed_scale=False,
                                     no_silhouette=False),
            xrange(num_instances))

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

