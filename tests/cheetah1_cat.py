# tests/cheetah1.py

# Imports
import os
import numpy as np
from mesh import faces, geometry

from core_recovery.lm_solvers import \
    solve_single_arap_proj, \
    solve_single_lap_proj_sil_spil

from core_recovery.silhouette_candidates import \
    generate_silhouette_candidate_info

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from visualise import *

# Directories
DIRECTORIES = {'working' : 'tests/cheetah1'}
for d in DIRECTORIES.itervalues():
    if not os.path.exists(d):
        os.makedirs(d)

# Loaders

# load_triangles
def load_triangles(filename):
    z = np.load(filename)
    return faces.faces_from_cell_array(z['cells'])
     
# load_user_constraints
def load_user_constraints(filename):
    z = np.load(filename)

    # load
    C = np.asarray(z['point_ids'], dtype=np.int32)
    P = np.asarray(z['positions'], dtype=np.float64)
    T = np.asarray(z['T'], dtype=np.float64)
    V = np.asarray(z['V'], dtype=np.float64)

    S = T[:3, :3]
    t = T[:3, -1]

    V = np.dot(V, np.transpose(S)) + t

    return V, C, P

# load_silhouette_information
def load_silhouette_information(filename):
    z = np.load(filename)
    return {k:z[k] for k in z.keys()}

# load_silhouette
def load_silhouette(filename):
    z = np.load(filename)
    return z['S'], z['SN']

# load_spillage
def load_spillage(filename):
    z = np.load(filename)
    return z['R']

# frame_path
def frame_path():
    return 'data/frames/cheetah1/4.png'

# Main

# main_silhouette_candidate_info
def main_silhouette_candidate_info():
    T = load_triangles('data/models/quad_prototype_tail.npz')
    V, C, P = load_user_constraints('data/user_constraints/'
                                    'cheetah1/quad_prototype_tail_4.npz')

    info = generate_silhouette_candidate_info(V, T, step=40., verbose=True)

    np.savez_compressed(os.path.join(
                        DIRECTORIES['working'],
                        'quad_prototype_tail_silhouette_info.npz'), **info)

# main_silhouette_candidate_info
def main_silhouette_candidate_info():
    T = load_triangles('data/models/cat/cat_simplified.npz')
    V, C, P = load_user_constraints('data/user_constraints/cheetah1/cat_simplified_4.npz')

    info = generate_silhouette_candidate_info(V, T, step=40., verbose=True)

    np.savez_compressed(os.path.join(
                        DIRECTORIES['working'],
                        'cat_simplified_4_silhouette_info.npz'), **info)

# main_user_constraints
def main_user_constraints():
    #T = load_triangles('data/models/quad_prototype_tail.npz')
    T = load_triangles('data/models/cat/cat_simplified.npz')
    # V, C, P = load_user_constraints('data/user_constraints/'
    #                                 'cheetah1/quad_prototype_tail_4.npz')
    V, C, P = load_user_constraints('data/user_constraints/cheetah1/cat_simplified_4.npz')

    V1 = V.copy()
    X = np.zeros_like(V)

    lambdas = np.array([1.0,  # as-rigid-as-possible
                        1.0], # projection
                        dtype=np.float64)

    print 'Starting `solve_single_arap_proj`'
    status, status_string = solve_single_arap_proj(
        V, T, X, V1, C, P, lambdas,
        maxIterations=100,
        verbosenessLevel=1)
    print 'Done.'

    output_path = os.path.join(DIRECTORIES['working'], 'user_constraints.npz')
    print 'Saving to: ', output_path

    np.savez_compressed(output_path,
            C=C,
            P=P,
            lambdas=lambdas,
            V=V,
            T=T,
            image=frame_path(),
            V1=V1)

# main_with_silhouette
def main_with_silhouette():
    # solving parameters
    silhouette_lambdas = np.array([1e-3,  # geodesic between preimage
                                   1e0,   # silhouette projection
                                   1e3],  # silhouette normal
                                   dtype=np.float64)

    lm_lambdas = np.r_[1.0,  # laplacian regularisation
                       1.0,  # user constraints
                       silhouette_lambdas[1:], # silhouette
                       1.0]  # spillage

    lm_preconditioners = np.array([1.0, 5.0], dtype=np.float64)
    narrowBand = 3
    solver_options = dict(
          maxIterations=20,
          gradientThreshold=1e-5,
          updateThreshold=1e-5,
          improvementThreshold=1e-5,
          verbosenessLevel=1)

    # model + user constraints
    T = load_triangles('data/models/cat/cat_simplified.npz')
    V, C, P = load_user_constraints('data/user_constraints/cheetah1/cat_simplified_4.npz')

    z = np.load(os.path.join(DIRECTORIES['working'], 'user_constraints.npz'))
    V1 = z['V1'] + (0, 0, 30)

    vis = VisualiseMesh()
    vis.add_mesh(V1, T)
    vis.add_image(frame_path())
    vis.camera_actions(('SetParallelProjection',(True,)))
    vis.execute()

    # silhouette information
    silhouette_info = load_silhouette_information(
        'tests/cheetah1/cat_simplified_4_silhouette_info.npz')

    # silhouette
    S, SN = load_silhouette('data/silhouettes/cheetah1/4_S.npz')

    # solve for the initial silhouette positions
    U, L = shortest_path_solve(V1, T, S, SN, 
                               lambdas=silhouette_lambdas,
                               isCircular=False, 
                               **silhouette_info)

    # fit to silhouette (including spillage)
    Rx, Ry = load_spillage('data/distance_maps/cheetah1/4_D.npz')    

    # solve_iteration
    def solve_iteration():
        status = solve_single_lap_proj_sil_spil(
            V1, T, U, L, C, P, S, SN, Rx, Ry,
            lm_lambdas, lm_preconditioners, narrowBand,
            **solver_options)

        print 'LM Status (%d): ' % status[0], status[1]

        if True:
            Q = geometry.path2pos(V1, T, L, U)
            N = Q.shape[0]

            vis = VisualiseMesh()
            vis.add_mesh(V1, T, L)
            vis.add_silhouette(Q, np.arange(N), [0, N-1], S)
            vis.add_image(frame_path())
            vis.camera_actions(('SetParallelProjection',(True,)))

            vis.execute()

        return status
    
    status = solve_iteration()
    while status[0] in (0, 4):
        status = solve_iteration()

    print 'Final Status (%d): ' % status[0], status[1]

if __name__ == '__main__':
    # main()
    # main_silhouette_candidate_info()
    # main_user_constraints()
    main_with_silhouette()


