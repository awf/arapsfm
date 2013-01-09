# solve_linear_deformation.py

# Imports
import os
from util.cmdline import * 
from itertools import count, izip

from core_recovery import lm_alt_solvers_linear
from core_recovery import lm_alt_solvers2 as lm

from visualise.visualise import VisualiseMesh
from core_recovery.arap.sectioned import parse_k

from misc.numpy_ import mparray
from scipy.linalg import norm

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str)
    parser.add_argument('core_initialisation', type=str)
    parser.add_argument('--use_linear_transform', 
                        action='store_true',
                        default=False)
    parser.add_argument('core_silhouette_info', type=str)
    parser.add_argument('global_rotation_config', type=str)
    parser.add_argument('user_constraints', type=str)
    parser.add_argument('silhouette', type=str)
    parser.add_argument('indices', type=str)
    parser.add_argument('lambdas', type=str)
    parser.add_argument('preconditioners', type=str)
    parser.add_argument('solver', type=str)
    parser.add_argument('output', type=str)

    parser.add_argument('--frames', type=str, default=None)
    parser.add_argument('--solver_options', type=str, default='{}')

    args = parser.parse_args()

    for key in ['lambdas',
                'preconditioners',
                'solver_options',
                'indices',
                'global_rotation_config',
                ]:
        setattr(args, key, eval(getattr(args, key)))

    # setup `solver_options`
    def_solver_options = dict(maxIterations=100, 
                              gradientThreshold=1e-6,
                              updateThreshold=1e-6,
                              improvementThreshold=1e-6,
                              verbosenessLevel=1)
    def_solver_options.update(args.solver_options)
    args.solver_options = def_solver_options

    if args.frames is not None:
        args.frames = [args.frames % d for d in args.indices]
    else:
        args.frames = [None for d in args.indices]

    # output arguments
    pprint(args.__dict__)

    # parse argments and load data
    load_instance_variables = lambda *a: load_formatted(args.indices, *a, verbose=True)

    V = load_input_geometry(args.core_initialisation, 
                            args.use_linear_transform)

    C, P = load_instance_variables(args.user_constraints, 'C', 'P')
    num_instances = len(C)

    S, SN = load_instance_variables(args.silhouette, 'S', 'SN')

    silhouette_info = np.load(args.core_silhouette_info)
    print '<- %s' % args.core_silhouette_info

    T = load_input_mesh(args.mesh)

    # parse `kg` to get the configuration of all of the global rotations
    kg = np.asarray(args.global_rotation_config, dtype=np.int32)
    kg_inst, kg_basis, kg_coeff, kg_lookup = parse_k(kg)

    def make_shared(s, n, dtype=np.float64, value=0.):
        b = []
        for i in xrange(n):
            a = mparray.empty(s, dtype=dtype)
            a.fill(value)
            b.append(a)
        return b

    # allocate global basis rotations
    Xg = make_shared((1, 3), len(kg_inst))
    Xgb = make_shared((1, 3), len(kg_basis))
    yg = make_shared((1, 1), len(kg_coeff), value=1.)

    instScales = make_shared((1, 1), num_instances, value=1.)

    U = [mparray.empty((s.shape[0], 2), np.float64) for s in S]
    L = [mparray.empty(s.shape[0], np.int32) for s in S]

    V1 = make_shared(V.shape, num_instances)
    dg = make_shared((1, 3), num_instances)

    instance_preconditioners = np.r_[args.preconditioners[0], # V
                                     args.preconditioners[1], # X/Xg
                                     args.preconditioners[2], # s
                                     args.preconditioners[3], # U
                                     args.preconditioners[4]] # y
                        
    empty3 = np.array(tuple(), dtype=np.float64).reshape(0, 3)

    # XXX Require making custom `kg` for each solve
    i = 0
    V1[i].flat = V.flat
    # XXX

    if not os.path.exists(args.output):
        print 'Creating directory:', args.output
        os.makedirs(args.output)

    n = count()
    def save_V1(iteration, computeDerivatives):
        if computeDerivatives:
            output_path = os.path.join(args.output, '%d.npz' % next(n))
            d = dict(T=T, V=V1[i],
                     C=C[i], P=P[i])

            if args.frames[i] is not None:
                d['image'] = args.frames[i]

            print '-> %s' % output_path
            np.savez_compressed(output_path, **d)

    # Solve using rigid registration with LSE
    if args.solver == 'rigid_lse':
        instance_lambdas = np.r_[args.lambdas[0],  # deformation model
                                 args.lambdas[1],  # silhouette projection energy
                                 args.lambdas[2],  # silhouette normal energy
                                 args.lambdas[3]]  # user constraints

        status = lm_alt_solvers_linear.solve_instance(
            T, V, instScales[i],
            kg, Xgb, yg, Xg,
            dg[i], V1[i],
            U[i], L[i],
            S[i], SN[i],
            C[i], P[i],
            instance_lambdas,
            instance_preconditioners,
            2, # narrowBand
            True, # uniformWeights
            False, # fixedScale
            True, # fixedGlobalRotation
            True, # fixedTranslation
            True, # noSilhouetteUpdate
            callback=save_V1,
            **args.solver_options)

    elif args.solver == 'rigid_arap':
        instance_lambdas = np.r_[args.lambdas[0],  # deformation model
                                 args.lambdas[1],  # silhouette projection energy
                                 args.lambdas[2],  # silhouette normal energy
                                 0., # spillage
                                 args.lambdas[3],  # user constraints
                                 0.,   # temporal ARAP penalty
                                 0.,   # global scale acceleration penalty
                                 0.,   # global rotations acceleration penalty
                                 0.,   # frame-to-frame acceleration penalty
                                 0.,   # global scale velocity penalty
                                 0.,   # global rotations velocity penalty
                                 0.]   # frame-to-frame velocity penalty

        # fixed rotations at each vertex
        k = np.zeros(V.shape[0], dtype=np.int32)
        Xb = np.zeros((0, 3), dtype=np.float64)
        y = np.zeros((0, 1), dtype=np.float64)
        X = np.zeros((0, 3), dtype=np.float64)
        ym1 = []
        Xm1 = []
        Vm1 = sp = Xgp = Xp = empty3
        sm1 = []

        status = lm.solve_instance(
            T, V, instScales[i],
            kg, Xgb, yg, Xg,
            k, Xb, y, X, ym1, Xm1,
            Vm1, sp, Xgp, Xp, sm1,
            V1[i],
            U[i], L[i],
            S[i], SN[i],
            np.zeros((1, 1), dtype=np.float64), # Rx
            np.zeros((1, 1), dtype=np.float64), # Ry
            C[i], P[i],
            instance_lambdas,
            instance_preconditioners,
            np.r_[0., 2.], # piecewise_polynomial
            2, # narrowBand
            True, # uniformWeights
            False, # fixedScale
            True, # fixedGlobalRotation
            True, # noSilhouetteUpdate
            callback=save_V1,
            **args.solver_options)

    elif args.solver == 'free_arap':
        instance_lambdas = np.r_[args.lambdas[0],  # deformation model
                                 args.lambdas[1],  # silhouette projection energy
                                 args.lambdas[2],  # silhouette normal energy
                                 0., # spillage
                                 args.lambdas[3],  # user constraints
                                 0.,   # temporal ARAP penalty
                                 0.,   # global scale acceleration penalty
                                 0.,   # global rotations acceleration penalty
                                 0.,   # frame-to-frame acceleration penalty
                                 0.,   # global scale velocity penalty
                                 0.,   # global rotations velocity penalty
                                 0.]   # frame-to-frame velocity penalty

        # fixed rotations at each vertex
        k = np.empty(2*V.shape[0], dtype=np.int32)
        k[::2] = -1
        k[1::2] = xrange(V.shape[0])

        Xb = np.zeros((0, 3), dtype=np.float64)
        y = np.zeros((0, 1), dtype=np.float64)
        X = np.zeros_like(V, dtype=np.float64)
        ym1 = []
        Xm1 = []
        Vm1 = sp = Xgp = Xp = empty3
        sm1 = []

        status = lm.solve_instance(
            T, V, instScales[i],
            kg, Xgb, yg, Xg,
            k, Xb, y, X, ym1, Xm1,
            Vm1, sp, Xgp, Xp, sm1,
            V1[i],
            U[i], L[i],
            S[i], SN[i],
            np.zeros((1, 1), dtype=np.float64), # Rx
            np.zeros((1, 1), dtype=np.float64), # Ry
            C[i], P[i],
            instance_lambdas,
            instance_preconditioners,
            np.r_[0., 2.], # piecewise_polynomial
            2, # narrowBand
            True, # uniformWeights
            False, # fixedScale
            True, # fixedGlobalRotation
            True, # noSilhouetteUpdate
            callback=save_V1,
            **args.solver_options)

    else:
        raise ValueError('unknown solver: "%s"' % args.solver)

    print status

    d = np.amin(V, axis=0)
    V -= (d[0], d[1], 0.)

    vis = VisualiseMesh()
    # vis.add_mesh(V, T, is_base=False, color=(178, 223, 138),
    #              actor_name='V')
    vis.add_mesh(V1[i], T, is_base=True, color=(31, 120, 180),
                 actor_name='V1')
    for actor_name in ['V1']: #['V', 'V1']:
        vis.actor_properties(actor_name, ('SetRepresentation', (3,)))
        
    vis.camera_actions(('SetParallelProjection', (True,)))
    vis.add_projection(C[i], P[i])

    if args.frames[i] is not None:
        vis.add_image(args.frames[i])
    vis.execute()

if __name__ == '__main__':
    main()
