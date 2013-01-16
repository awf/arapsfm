# start_core_recovery.py

# Imports
from util.cmdline import * 
from core_recovery2.problem import CoreRecoverySolver

# parse_args
def parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument('mesh', type=str)
    parser.add_argument('core_initialisation', type=str)
    parser.add_argument('core_silhouette_info', type=str)
    parser.add_argument('arap_sections', type=str)

    parser.add_argument('global_rotation_config', type=str)
    parser.add_argument('indices', type=str)
    parser.add_argument('user_constraints', type=str)
    parser.add_argument('silhouette', type=str)
    parser.add_argument('frames', type=str)

    parser.add_argument('lambdas', type=str)
    parser.add_argument('preconditioners', type=str)

    parser.add_argument('output', type=str)

    parser.add_argument('--solver_options', type=str, default='{}')
    parser.add_argument('--narrowband', type=int, default=2)
    parser.add_argument('--no_uniform_weights', dest='uniform_weights',
                        default=True,
                        action='store_false')
    parser.add_argument('--max_restarts', type=int, default=10)
    parser.add_argument('--outer_loops', type=int, default=20)
    parser.add_argument('--candidate_radius', type=float, default=None)

    parser.add_argument('--quit_after_silhouette',
                        action='store_true',
                        default=False)

    parser.add_argument('--initial_Xgb', type=str, default='None')
    parser.add_argument('--initial_Xb', type=str, default='None')
    parser.add_argument('--use_creasing_silhouette',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    for key in ['indices',
                'lambdas',
                'preconditioners',
                'solver_options',
                'global_rotation_config',
                'initial_Xgb',
                'initial_Xb',
                ]:
        setattr(args, key, eval(getattr(args, key)))
    
    for key, dtype in [
        ('lambdas', np.float64),
        ('preconditioners', np.float64),
        ('initial_Xgb', np.float64),
        ('global_rotation_config', np.int32),
        ('initial_Xb', np.float64)]:

        a = getattr(args, key)
        if a is not None:
            setattr(args, key, np.asarray(a, dtype=dtype))

    if '{default}' in args.output:
        make_str = lambda a: ",".join('%.4g' % a_ for a_ in a)
        mesh_file = os.path.split(args.mesh)[1]
        arap_file = os.path.split(args.arap_sections)[1]

        if len(args.indices) > 10:
            indices_str = (make_str(args.indices[:5]) + '---' +
                           make_str(args.indices[-5:]))
        else:
            indices_str = make_str(args.indices)

        default_directory = '%s_%s_%s_%s_%s' % (
            os.path.splitext(mesh_file)[0],
            os.path.splitext(arap_file)[0],
            indices_str,
            make_str(args.lambdas), 
            make_str(args.preconditioners))

        args.output = args.output.format(default=default_directory)

    args.frames = map(lambda i: args.frames % i, args.indices)

    return args

# setup_solver
def setup_solver(args):    
    solver = CoreRecoverySolver(args.lambdas,
                                args.preconditioners,
                                args.solver_options,
                                args.narrowband,
                                args.uniform_weights,
                                args.max_restarts,
                                args.outer_loops,
                                args.candidate_radius,
                                args.use_creasing_silhouette)

    print '<- %s' % args.mesh
    print '<- %s' % args.core_silhouette_info
    print '<- %s' % args.core_initialisation

    solver.set_mesh(load_input_mesh(args.mesh),
                    load_input_geometry(args.core_initialisation, True),
                    np.load(args.core_silhouette_info))

    load_instance_variables = lambda *a: load_formatted(args.indices, *a, verbose=True)

    C, P = load_instance_variables(args.user_constraints, 'C', 'P')
    S, SN = load_instance_variables(args.silhouette, 'S', 'SN')

    kg = args.global_rotation_config
    ki = np.require(np.load(args.arap_sections)['k2'], 
                    dtype=np.int32)

    solver.set_data(args.frames, C, P, S, SN)

    solver.setup(kg=kg,
                 initial_Xgb=args.initial_Xgb,
                 ki=ki,
                 initial_Xb=args.initial_Xb)

    return solver

# main
def main():
    args = parse_args()
    solver = setup_solver(args)

    print '-> %s' % args.output
    pickle_.dump(args.output, solver)
 
if __name__ == '__main__':
    main()

