# solve_single.py

# Imports
from util.cmdline import *

from core_recovery.lm_solvers import \
    solve_single_arap_proj, \
    solve_single_lap_proj_silhouette, \
    solve_single_lap_proj_sil_spil

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from visualise import *
    
# main
def main():
    parser = argparse.ArgumentParser(
        description='Solve single frame core covery problem')
    parser.add_argument('solver', type=str)
    parser.add_argument('mesh', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('lambdas', type=str)

    # input geometry option(s)
    parser.add_argument('--use_linear_transform', 
                        action='store_true',
                        default=False)

    # projection energy
    parser.add_argument('--user_constraints', type=str, default=None)

    # silhouette energy
    parser.add_argument('--silhouette_info', type=str, default=None)
    parser.add_argument('--silhouette_input', type=str, default=None)
    parser.add_argument('--preconditioners', type=str, default=None)
    parser.add_argument('--narrowband', type=int, default=3)
    parser.add_argument('--max_restarts', type=int, default=5)
    parser.add_argument('--find_circular_path', action='store_true',
                        default=False)

    # spillage energy
    parser.add_argument('--spillage_input', type=str, default=None)

    # optional
    parser.add_argument('--input_frame', type=str, default=None)
    parser.add_argument('--solver_options', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    # parse the arguments
    args = parser.parse_args()

    # load the geometry and input mesh 
    V = load_input_geometry(args.input, args.use_linear_transform)
    T = load_input_mesh(args.mesh)

    # parse the lambdas
    lambdas = parse_float_string(args.lambdas)

    # solver arguments
    solver_options = parse_solver_options(args.solver_options,
        maxIterations=20, 
        gradientThreshold=1e-6,
        updateThreshold=1e-6,
        improvementThreshold=1e-6,
        verbosenessLevel=1)

    print 'V.shape:', V.shape
    print 'T.shape:', T.shape
    print 'solver:', args.solver
    print 'lambdas:', lambdas
    print 'solver_options:', solver_options

    # additional solver input files
    user_constraints = args.user_constraints
    if user_constraints is None:
        user_constraints = args.input

    # initialis visualisation (may not be used)
    vis = VisualiseMesh()
    if args.input_frame is not None:
        vis.add_image(args.input_frame)

    # initialise output dictionary
    output_d = args.__dict__.copy()
    output_d.update(V0=V, T=T, lambdas=lambdas)
    pprint(output_d) 

    if args.solver == 'single_arap_proj':
        # as-rigid-as-possible with projection constraints
        C, P = load_args(user_constraints, 'C', 'P')
        print 'C.shape:', C.shape
        print 'P.shape:', P.shape

        X = np.zeros_like(V)
        V1 = V.copy()

        status = solve_single_arap_proj(
            V, T, X, V1, C, P, lambdas,
            **solver_options)

        # augment visualisation
        vis.add_mesh(V1, T)
        vis.add_projection(C, P)

        # augment output dictionary
        output_d['C'] = C
        output_d['P'] = P
        output_d['V'] = V1

    elif args.solver == 'single_lap_proj_silhouette':
        # required arguments
        requires(args, 'silhouette_info', 
                 'silhouette_input', 
                 'preconditioners')

        # laplacian smoothing with projection and silhouette constraints
        C, P = load_args(user_constraints, 'C', 'P')
        silhouette_info = load(args.silhouette_info)

        global_silhoutte_lambdas = lambdas[:3]
        print 'global_silhoutte_lambdas:', global_silhoutte_lambdas

        S, SN = load_args(args.silhouette_input, 'S', 'SN')
        print 'S.shape:', S.shape
        print 'SN.shape:', SN.shape

        # solver for the shortest path to initialise U and L
        print 'shortest_path_solve'
        U, L = shortest_path_solve(V, T, S, SN,
                                   silhouette_info['SilCandDistances'],
                                   silhouette_info['SilEdgeCands'],
                                   silhouette_info['SilEdgeCandParam'],
                                   silhouette_info['SilCandAssignedFaces'],
                                   silhouette_info['SilCandU'],
                                   global_silhoutte_lambdas,
                                   isCircular=args.find_circular_path)

        # minimise to the silhouette
        lm_lambdas = np.r_[lambdas[3],    # laplacian
                           lambdas[4],    # projection
                           lambdas[1:3]]  # silhouette
        print 'lm_lambdas:', lm_lambdas

        preconditioners = parse_float_string(args.preconditioners)
        print 'preconditioners:', preconditioners
        print 'narrowband:', args.narrowband

        def solve_iteration():
            status = solve_single_lap_proj_silhouette(V, T, U, L, C, P, S, SN,
                                                      lm_lambdas,
                                                      preconditioners,
                                                      args.narrowband,
                                                      **solver_options)

            print 'LM Status (%d): ' % status[0], status[1]

            return status

        print 'max_restarts:', args.max_restarts
        for i in xrange(args.max_restarts):
            status = solve_iteration()
            if status[0] not in (0, 4):
                break

        # augment visualisation
        Q = geometry.path2pos(V, T, L, U)
        N = Q.shape[0]

        vis.add_mesh(V, T, L)
        vis.add_projection(C, P)
        vis.add_silhouette(Q, np.arange(N), [0, N-1], S)

    elif args.solver == 'single_lap_proj_sil_spil':
        # required arguments
        requires(args, 'silhouette_info', 
                 'silhouette_input', 
                 'spillage_input',
                 'preconditioners')

        # laplacian smoothing with projection and silhouette constraints
        C, P = load_args(user_constraints, 'C', 'P')
        silhouette_info = load(args.silhouette_info)

        global_silhoutte_lambdas = lambdas[:3]
        print 'global_silhoutte_lambdas:', global_silhoutte_lambdas

        S, SN = load_args(args.silhouette_input, 'S', 'SN')
        print 'S.shape:', S.shape
        print 'SN.shape:', SN.shape

        # load spillage information
        (R,) = load_args(args.spillage_input, 'R') 
        Rx, Ry = R

        # solver for the shortest path to initialise U and L
        print 'shortest_path_solve'
        U, L = shortest_path_solve(V, T, S, SN,
                                   silhouette_info['SilCandDistances'],
                                   silhouette_info['SilEdgeCands'],
                                   silhouette_info['SilEdgeCandParam'],
                                   silhouette_info['SilCandAssignedFaces'],
                                   silhouette_info['SilCandU'],
                                   global_silhoutte_lambdas,
                                   isCircular=args.find_circular_path)

        # minimise to the silhouette and spillage
        lm_lambdas = np.r_[lambdas[3],    # laplacian
                           lambdas[4],    # projection
                           lambdas[1:3],  # silhouette
                           lambdas[5]]    # spillage

        print 'lm_lambdas:', lm_lambdas

        preconditioners = parse_float_string(args.preconditioners)
        print 'preconditioners:', preconditioners
        print 'narrowband:', args.narrowband

        def solve_iteration():
            status = solve_single_lap_proj_sil_spil(V, T, U, L, C, P, S, SN,
                                                    Rx, Ry,
                                                    lm_lambdas,
                                                    preconditioners,
                                                    args.narrowband,
                                                    **solver_options)

            print 'LM Status (%d): ' % status[0], status[1]

            return status

        print 'max_restarts:', args.max_restarts
        for i in xrange(args.max_restarts):
            status = solve_iteration()
            if status[0] not in (0, 4):
                break

        # augment visualisation
        Q = geometry.path2pos(V, T, L, U)
        N = Q.shape[0]

        vis.add_mesh(V, T, L)
        vis.add_projection(C, P)
        vis.add_silhouette(Q, np.arange(N), [0, N-1], S)

    else:
        raise ValueError('unknown solver: %s' % args.solver)

    print 'status:', status
    
    if args.output is None:
        print 'output: Interactive'
        vis.camera_actions(('SetParallelProjection', (True,)))
        vis.execute()
    else:
        print 'output: %s' % args.output
        pickle_.dump(args.output, output_d)

if __name__ == '__main__':
    main()
