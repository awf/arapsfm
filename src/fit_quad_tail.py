# fit_quad_tail.py

# Imports
import os
import numpy as np
from mesh import faces, geometry

from core_recovery.silhouette_candidates import \
    generate_silhouette_candidate_info

from core_recovery.lm_solvers import \
    solve_single_arap_proj, \
    solve_single_lap_proj, \
    solve_single_lap_silhouette, \
    solve_single_lap_proj_silhouette, \
    solve_multiview_arap_silhouette, \
    solve_multiview_lap_silhouette

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from visualise import visualise

from time import clock

from project import Workspace

# Global workspace
#wsp = Workspace('quad_prototype_tail', 'cheetah0')
wsp = Workspace('chihuahua', 'cheetah0')

# main
def main():
    T = wsp.load_triangles()
    #V, C, P = wsp.load_projection('0f')
    V, C, P = wsp.load_projection('0b')

    single_arap_lambdas = np.array([1.0,  # as-rigid-as-possible
                                    1e1], # projection
                                    dtype=np.float64)

    # solve for projection
    X = np.zeros_like(V)
    V1 = V.copy()
        
    print 'solve_single_arap_proj'
    status = solve_single_arap_proj(
        V, T, X, V1, C, P, single_arap_lambdas, 
        maxIterations=20,
        gradientThreshold=1e-6,
        updateThreshold=1e-6,
        improvementThreshold=1e-6)
    print 'Status (%d): ' % status[0], status[1]

    # show
    vis = visualise.VisualiseMesh(V1, T)
    vis.add_image(wsp.get_frame_path(0))
    vis.add_projection(C, P)
    vis.execute()

    # solve for Laplacian
    global_sil_lambdas = np.array([1e0,  # geodesic between preimage points
                                   1e0,   # silhouette projection
                                   1e3],  # silhouette normal
                                   dtype=np.float64)

    #lm_lambdas = np.r_[1e1, 0.0, global_sil_lambdas[1:]]
    lm_lambdas = np.r_[1e1, global_sil_lambdas[1:]]
    lm_lambdas = np.asarray(lm_lambdas, dtype=np.float64)
    print 'lm_lambdas:', lm_lambdas

    lm_preconditioners = np.array([1.0, 100.0], dtype=np.float64)
    print 'lm_preconditioners :', lm_preconditioners 

    # use as initialisation for next stage
    V = V1.copy()

    # solve for silhouette
    silhouette_info = wsp.load_silhouette_info()
    S, SN = wsp.get_silhouette(0)

    print 'shortest_path_solve'
    U, L = shortest_path_solve(V, T, S, SN,
                               lambdas=global_sil_lambdas,
                               isCircular=True,
                               **silhouette_info)
    # show
    vis = visualise.VisualiseMesh(V, T, L)
    vis.add_image(wsp.get_frame_path(0))

    Q = geometry.path2pos(V, T, L, U)
    N = Q.shape[0]

    vis.add_silhouette(Q, np.arange(N), [0, N-1], S)
    vis.add_projection(C, P)
    vis.execute()

    # fit under laplacian
    def solve_iteration():
        status = solve_single_lap_silhouette(V, T, U, L, S, SN, 
            lm_lambdas, 
            lm_preconditioners, 
            narrowBand=3, 
            maxIterations=50,
            gradientThreshold=1e-6,
            updateThreshold=1e-6,
            improvementThreshold=1e-6,
            verbosenessLevel=1,
            useAsymmetricLambda=True,
            )
        print 'LM Status (%d): ' % status[0], status[1]
        #status = solve_single_lap_proj_silhouette(V, T, U, L, C, P, S, SN, 
        #    lm_lambdas, 
        #    lm_preconditioners, 
        #    narrowBand=2, 
        #    maxIterations=50,
        #    gradientThreshold=1e-6,
        #    updateThreshold=1e-6,
        #    improvementThreshold=1e-6,
        #    verbosenessLevel=1,
        #    useAsymmetricLambda=True,
        #    )
        #print 'LM Status (%d): ' % status[0], status[1]

        return status

    #for i in xrange(10):
    #    status = solve_iteration()
    count = 0
    while status[0] in (0, 4) and count < 5:
        status = solve_iteration()
        count += 1

    # show
    vis = visualise.VisualiseMesh(V, T, L)
    vis.add_image(wsp.get_frame_path(0))

    Q = geometry.path2pos(V, T, L, U)
    N = Q.shape[0]

    vis.add_silhouette(Q, np.arange(N), [0, N-1], S)
    vis.add_projection(C, P)
    vis.execute()

# generate_silhouette_info
def generate_silhouette_info():
    T = wsp.load_triangles()
    V, C, P = wsp.load_projection('0d')

    info = generate_silhouette_candidate_info(V, T, step=40., verbose=True)

    wsp.save('silhouette_info', **info)

if __name__ == '__main__':
    main()
    #generate_silhouette_info()

