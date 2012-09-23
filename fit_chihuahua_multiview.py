# generate_chihuahua_silhouette_candidates.py

# Imports
import os
import numpy as np
from mesh import faces, geometry

from core_recovery.silhouette_candidates import \
    generate_silhouette_candidate_info

from core_recovery.lm_solvers import \
    solve_single_arap_proj, \
    solve_single_lap_silhouette, \
    solve_multiview_arap_silhouette, \
    solve_multiview_lap_silhouette

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from visualise import visualise

from time import clock

# Constants
DATA_ROOT = 'data'
INPUT_MODEL = os.path.join(DATA_ROOT, 'models', 'chihuahua.npz')
PROJECTIONS_ROOT = os.path.join(DATA_ROOT, 'projection_constraints', 'chihuahua')
SILHOUETTES_ROOT = os.path.join(DATA_ROOT, 'silhouettes', 'cheetah0')
SPILLAGE_ROOT = os.path.join(DATA_ROOT, 'distance_maps', 'cheetah0')
FRAMES_ROOT = os.path.join(DATA_ROOT, 'frames', 'cheetah0')

OUTPUT_ROOT = 'working'

INPUT_SELECTION = [(0, '0b'), (5, '5c'), (9, '9b')]

# get_frame_path
def get_frame_path(index):
    return os.path.join(FRAMES_ROOT, '%d.png' % index)

# load_projection
def load_projection(stem):
    path = os.path.join(PROJECTIONS_ROOT, '%s.npz' % stem)

    z = np.load(path)

    # load
    C = np.asarray(z['point_ids'], dtype=np.int32)
    P = np.asarray(z['positions'], dtype=np.float64)
    T = np.asarray(z['T'], dtype=np.float64)
    V = np.asarray(z['V'], dtype=np.float64)

    S = T[:3, :3]
    t = T[:3, -1]

    V = np.dot(V, np.transpose(S)) + t

    return V, C, P

# load_triangles
def load_triangles():
    z = np.load(INPUT_MODEL)
    return faces.faces_from_cell_array(z['cells'])

# main_silhouette_candidate_info
def main_silhouette_candidate_info():
    V, C, P = load_projection('0b')
    T = load_triangles()

    info = generate_silhouette_candidate_info(V, T, step=40., verbose=True)

    np.savez_compressed(os.path.join(OUTPUT_ROOT,
                        'chihuahua_silhouette_info.npz'), **info)

# load_silhouette_info
def load_silhouette_info():
    z = np.load(os.path.join(OUTPUT_ROOT, 'chihuahua_silhouette_info.npz'))
    return {k:z[k] for k in z.keys()}

# main_fit_single_projections
def main_fit_single_projections():
    # use '0b' for initial geometry
    V = load_projection('0b')[0]
    T = load_triangles()

    lambdas = np.array([1.0,  # as-rigid-as-possible
                        1.0], # projection
                        dtype=np.float64)

    for index, user_constraints in INPUT_SELECTION:
        print 'index:', index

        _, C, P = load_projection(user_constraints)
        V1 = V.copy()
        X = np.zeros_like(V)

        status, status_string = solve_single_arap_proj(
            V, T, X, V1, C, P, lambdas)

        # visualise ?
        if False:
            vis = visualise.VisualiseMesh(V1, T)
            vis.execute()

        np.savez_compressed(os.path.join(OUTPUT_ROOT,
                            'chihuahua_single_projection_%s.npz' %
                            user_constraints), V1=V1, X=X, lambdas=lambdas)

# load_silhouette
def load_silhouette(index):
    z = np.load(os.path.join(SILHOUETTES_ROOT, '%d_S.npz' % index))
    return z['S'], z['SN']

# load_spillage
def load_spillage(index):
    z = np.load(os.path.join(SPILLAGE_ROOT, '%d_D.npz' % index))
    return z['R']

# main_fit_single_silhouette
def main_fit_single_silhouette():
    # use '0b' for initial core geometry
    V = load_projection('0b')[0]
    T = load_triangles()

    # information for silhouette specific only to the model (approx)
    silhouette_info = load_silhouette_info()

    global_solve_lambdas = np.array([1e-3,  # geodesic between preimage
                                     1e0,   # silhouette projection
                                     1e3],  # silhouette normal
                                     dtype=np.float64)

    lm_lambdas = np.r_[2.0,  # laplacian regularisation
                       global_solve_lambdas[1:], # silhouette
                       1.0]  # spillage

    lm_lambdas = np.asarray(lm_lambdas, dtype=np.float64)
    lm_preconditioners = np.array([1.0, 5.0], dtype=np.float64)

    for i, (index, user_constraints) in enumerate(INPUT_SELECTION):
        if i > 0:
            break
        print 'index:', index
    
        # load geometry from initial projection
        z = np.load(
            os.path.join(OUTPUT_ROOT, 'chihuahua_single_projection_%s.npz' % 
            user_constraints))
        V = z['V1']

        # get the silhouette information for the frame
        S, SN = load_silhouette(index)

        # get the spillage information for the frame
        Rx, Ry = load_spillage(index)

        # solve for the initial silhouette positions
        U, L = shortest_path_solve(V, T, S, SN, 
                                   lambdas=global_solve_lambdas,
                                   isCircular=False, 
                                   **silhouette_info)

        # fit under laplacian
        def solve_iteration():
            status = solve_single_lap_silhouette(V, T, U, L, S, SN, Rx, Ry,
                lm_lambdas, 
                lm_preconditioners, 
                narrowBand=3, 
                maxIterations=20,
                gradientThreshold=1e-5,
                updateThreshold=1e-5,
                improvementThreshold=1e-5,
                )
            print 'LM Status (%d): ' % status[0], status[1]

            return status

        status = solve_iteration()
        while status[0] in (0, 4):
            status = solve_iteration()

        print 'Final Status (%d): ' % status[0], status[1]

        # visualise ?
        if True:
            Q = geometry.path2pos(V, T, L, U)
            N = Q.shape[0]

            vis = visualise.VisualiseMesh(V, T, L)
            vis.add_silhouette(Q, np.arange(N), [0, N-1], S)
            vis.add_image(get_frame_path(index))

            vis.execute()

# main_fit_joint_arap_silhouette
def main_fit_joint_arap_silhouette():
    # use '0b' for initial core geometry
    V = load_projection('0b')[0]
    T = load_triangles()

    # information for silhouette specific only to the model (approx)
    silhouette_info = load_silhouette_info()

    global_solve_lambdas = np.array([1e-3,  # geodesic between preimage
                                     1e0,   # silhouette projection
                                     1e3],  # silhouette normal
                                     dtype=np.float64)

    lm_lambdas = np.asarray(np.r_[1.0,  # as-rigid-as-possible
                                 global_solve_lambdas[1:]], 
                           dtype=np.float64)
    
    
    lm_preconditioners = np.array([1.0, 1.0, 100.0], dtype=np.float64)

    # construct lists for minimisation
    multiX, multiV , multiU, multiL, multiS, multiSN = [list() for i in range(6)]

    for index, user_constraints in INPUT_SELECTION:
        print 'index:', index

        # load geometry from initial projection
        z = np.load(
            os.path.join(OUTPUT_ROOT, 'chihuahua_single_projection_%s.npz' % 
            user_constraints))
        X = z['X']
        V1 = z['V1']

        # get the silhouette information for the frame
        S, SN = load_silhouette(index)

        # solve for the initial silhouette positions
        U, L = shortest_path_solve(V1, T, S, SN, 
                                   lambdas=global_solve_lambdas,
                                   isCircular=False, 
                                   **silhouette_info)

        multiX.append(X)
        multiV.append(V1)
        multiU.append(U)
        multiL.append(L)
        multiS.append(S)
        multiSN.append(SN)

    # solve_iteration
    def solve_iteration():
        status = solve_multiview_arap_silhouette(
            T, V, multiX, multiV, multiU, multiL, multiS, multiSN, lm_lambdas,
            lm_preconditioners,
            narrowBand=2, 
            maxIterations=50,
            gradientThreshold=1e-6,
            updateThreshold=1e-6,
            improvementThreshold=1e-6)

        print 'LM Status (%d): ' % status[0], status[1]

        return status

    solve_iteration()

    # visualise ?
    if True:
        vis = visualise.VisualiseMesh(V, T)
        vis.execute()

        for i in xrange(len(multiV)):
            V1 = multiV[i]
            U = multiU[i]
            L = multiL[i]

            Q = geometry.path2pos(V1, T, L, U)
            N = Q.shape[0]

            vis = visualise.VisualiseMesh(V1, T, L)
            vis.add_silhouette(Q, np.arange(N), [0, N-1], multiS[i])
            vis.add_image(get_frame_path(INPUT_SELECTION[i][0]))

            vis.execute()

# main_fit_joint_lap_silhouette
def main_fit_joint_lap_silhouette():
    # use '0b' for initial core geometry
    V = load_projection('0b')[0]
    T = load_triangles()

    # information for silhouette specific only to the model (approx)
    silhouette_info = load_silhouette_info()

    # weighting lambdas
    global_solve_lambdas = np.array([1e-3,  # geodesic between preimage
                                     1e0,   # silhouette projection
                                     1e3],  # silhouette normal
                                     dtype=np.float64)

    lm_lambdas = np.asarray(np.r_[1.0,  # as-rigid-as-possible
                                 global_solve_lambdas[1:], 
                                 1e1], # laplacian
                           dtype=np.float64)
    
    # preconditioning for the joint minimisation
    lm_preconditioners = np.array([1.0, 1.0, 100.0], dtype=np.float64)

    # other solver options
    solver_options = dict(narrowBand=2, 
                          uniformWeights=True,
                          maxIterations=20,
                          gradientThreshold=1e-5,
                          updateThreshold=1e-5,
                          improvementThreshold=1e-5,
                          verbosenessLevel=1)

    # construct lists for minimisation
    multiX, multiV , multiU, multiL, multiS, multiSN = [list() for i in range(6)]

    for index, user_constraints in INPUT_SELECTION:
        print 'index:', index

        # load geometry from initial projection
        z = np.load(
            os.path.join(OUTPUT_ROOT, 'chihuahua_single_projection_%s.npz' % 
            user_constraints))
        X = z['X']
        V1 = z['V1']

        # get the silhouette information for the frame
        S, SN = load_silhouette(index)

        # solve for the initial silhouette positions
        U, L = shortest_path_solve(V1, T, S, SN, 
                                   lambdas=global_solve_lambdas,
                                   isCircular=False, 
                                   **silhouette_info)

        multiX.append(X)
        multiV.append(V1)
        multiU.append(U)
        multiL.append(L)
        multiS.append(S)
        multiSN.append(SN)

    # solve_iteration
    def solve_iteration():
        status = solve_multiview_lap_silhouette(
            T, V, multiX, multiV, multiU, multiL, multiS, multiSN, lm_lambdas,
            lm_preconditioners, **solver_options)

        print 'LM Status (%d): ' % status[0], status[1]

        return status

    # solve
    t1 = clock()

    count = 0
    status = solve_iteration()
    while status[0] in (0, 4):
        status = solve_iteration()
        count += 1

    t2 = clock()
    print 'Time taken: %.3fs' % (t2 - t1)

    np.savez_compressed(os.path.join(OUTPUT_ROOT,
        'chihuahua_lap_silhouette.npz'),
        T=T,
        V=V,
        multiV=multiV,
        multiS=multiS,
        multiSN=multiSN,
        lm_lambdas=lm_lambdas,
        lm_preconditioners=lm_preconditioners,
        solver_options=solver_options)

    # visualise ?
    if True:
        vis = visualise.VisualiseMesh(V, T)
        vis.execute()

        for i in xrange(len(multiV)):
            V1 = multiV[i]
            U = multiU[i]
            L = multiL[i]

            Q = geometry.path2pos(V1, T, L, U)
            N = Q.shape[0]

            vis = visualise.VisualiseMesh(V1, T, L)
            vis.add_silhouette(Q, np.arange(N), [0, N-1], multiS[i])
            vis.add_image(get_frame_path(INPUT_SELECTION[i][0]))

            vis.execute()

if __name__ == '__main__':
    #main_silhouette_candidate_info()
    #main_fit_single_projections()
    main_fit_single_silhouette()
    #main_fit_joint_arap_silhouette()
    #main_fit_joint_lap_silhouette()

