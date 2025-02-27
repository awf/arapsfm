# test_sphere.py

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

from visualise import *
from mesh import faces, geometry

from core_recovery.silhouette_candidates import \
    generate_silhouette_candidate_info

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from core_recovery.lm_solvers import \
    solve_single_lap_silhouette, \
    solve_single_lap_silhouette_with_Jte, \
    solve_single_lap_sil_len_adj_with_Jte, \
    solve_single_spillage

# Constants
OUTPUT_ROOT = 'working'

# generate_sphere
def generate_sphere(**options):
    sphere_source = vtk.vtkSphereSource()
    for option, value in options.iteritems():
        try:
            method = getattr(sphere_source, 'Set' + option)
        except AttributeError:
            continue

        method(value)
            
    sphere_source.Update()
    
    poly_data = sphere_source.GetOutput()
    V, T_ = vtkPolyData_to_numpy(poly_data)

    return V.astype(np.float64), faces.faces_from_cell_array(T_)

# test_sphere
def test_sphere():
    return generate_sphere(PhiResolution=16, 
                           ThetaResolution=30,
                           Center=(250., 250., 150.),
                           Radius=100.)

# generate_silhouette_info
def generate_silhouette_info():
    V, T = test_sphere()
    info = generate_silhouette_candidate_info(V, T, step=20., verbose=True)
    np.savez_compressed('sphere_silhouette_info.npz', **info)

# test_sphere_silhouette_info
def test_sphere_silhouette_info():
    z = np.load('sphere_silhouette_info.npz')
    return {k:z[k] for k in z.keys()}

# test_silhouette
def test_silhouette():
    z = np.load('data/silhouettes/circle/0_S.npz')
    return z['S'], z['SN']

# test_visualise
def test_visualise(V, T, U=None, L=None, S=None):
    vis = VisualiseMesh()
    vis.add_mesh(V, T, L)
    vis.add_image('data/segmentations/circle/0-INV_S.png')
    
    if U is not None:
        Q = geometry.path2pos(V, T, L, U)
        N = Q.shape[0]
        vis.add_silhouette(Q, np.arange(N), [0, N-1], S)

    vis.camera_actions(('SetParallelProjection', True))
    vis.execute()

# main_single_lap_silhouette
def main_single_lap_silhouette():
    V, T = test_sphere()
    print '# Vertices:', V.shape[0]
    print '# Triangles:', T.shape[0]

    # test_visualise(V, T)

    sil_lambdas = np.array([1e-3, 1.0, 0.], dtype=np.float64)
    print 'Silhouette lambdas:', sil_lambdas

    S, SN = test_silhouette()
    print '# Silhouette points:', S.shape[0]

    U, L = shortest_path_solve(V, T, S, SN,
                               lambdas=sil_lambdas,
                               isCircular=True,
                               **test_sphere_silhouette_info())

    test_visualise(V, T, U, L, S)

    lm_lambdas = np.r_[1.0, sil_lambdas[1:]]
    lm_lambdas = np.array(lm_lambdas, dtype=np.float64)
    lm_precond = np.array([1.0, 50.0], dtype=np.float64)

    print 'LM lambdas:', lm_lambdas
    print 'LM preconditioners:', lm_precond

    V1 = V.copy()

    status, saved_Jte = solve_single_lap_silhouette_with_Jte(V1, T, U, L, S, SN,
                                                             lm_lambdas, lm_precond, 
                                                             narrowBand=3,
                                                             maxJteStore=100,
                                                             maxIterations=50,
                                                             gradientThreshold=1e-4,
                                                             updateThreshold=1e-4,
                                                             improvementThreshold=1e-4,
                                                             verbosenessLevel=1)

    count = 0
    while status[0] in (4,) and count < 3:
        status, saved_Jte = solve_single_lap_silhouette_with_Jte(V1, T, U, L, S, SN,
                                                          lm_lambdas, lm_precond, 
                                                          narrowBand=3,
                                                          maxJteStore=100,
                                                          maxIterations=100,
                                                          verbosenessLevel=1)
        count += 1

    # np.save('saved_Jte.npy', saved_Jte)

    indices = np.linspace(0, saved_Jte.shape[0] - 1, 4, 
                          endpoint=True).astype(int)

    max_y = np.amax(saved_Jte[0])
    min_y = np.amin(saved_Jte[0]) 
    max_y += 0.1 * np.abs(max_y)
    min_y -= 0.1 * np.abs(min_y)

    f, axs = plt.subplots(indices.shape[0], 1)
    for i, index in enumerate(indices):
        axs[i].plot(saved_Jte[index], 'r.')
        axs[i].set_title('Index: %d' % index)
        # axs[i].set_ylim(min_y, max_y)

    plt.show(block=False)

    print 'Status (%d):' % status[0], status[1]

    test_visualise(V1, T, U, L, S)

# test_spillage
def test_spillage():
    z = np.load('data/distance_maps/circle/0_D.npz')
    return z['R']

# test_spillage_sphere
def test_spillage_sphere():
    return generate_sphere(PhiResolution=16, 
                           ThetaResolution=30,
                           Center=(250., 250., 150.),
                           Radius=200.)

# main_single_spillage
def main_single_spillage():
    V, T = test_spillage_sphere()
    test_visualise(V, T)

    Rx, Ry = test_spillage()
    status = solve_single_spillage(V, Rx, Ry,
                              maxIterations=5,          
                              gradientThreshold=1e-4,
                              updateThreshold=1e-4,
                              improvementThreshold=1e-4,
                              verbosenessLevel=1)        

    print 'status:', status

    output_filename = os.path.join(OUTPUT_ROOT, 'test_sphere.npz')
    print '-> %s' % output_filename

    np.savez_compressed(output_filename, V=V, T=T, 
                        image='data/segmentations/circle/0-INV_S.png')

    test_visualise(V, T)

if __name__ == '__main__':
    # generate_silhouette_info()
    # main_single_lap_silhouette()
    main_single_spillage()

