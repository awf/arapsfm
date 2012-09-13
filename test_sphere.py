# test_sphere.py

# Imports
from visualise.visualise import *
from visualise.vtk_ import *
from mesh import faces, geometry

from core_recovery.silhouette_candidates import \
    generate_silhouette_candidate_info

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

from core_recovery.lm_solvers import \
    solve_single_lap_silhouette

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
    vis = VisualiseMesh(V, T, L)
    vis.add_image('data/segmentations/circle/0-INV_S.png')
    
    if U is not None:
        Q = geometry.path2pos(V, T, L, U)
        N = Q.shape[0]
        vis.add_silhouette(Q, np.arange(N), [0, N-1], S)

    vis.execute()

# main
def main():
    V, T = test_sphere()
    print '# Vertices:', V.shape[0]
    print '# Triangles:', T.shape[0]

    test_visualise(V, T)

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
    lm_precond = np.array([1, 1e4], dtype=np.float64)

    print 'LM lambdas:', lm_lambdas
    print 'LM preconditioners:', lm_precond

    V1 = V.copy()
    status = solve_single_lap_silhouette(V1, T, U, L, S, SN,
                                         lm_lambdas, lm_precond, 
                                         narrowBand=3,
                                         maxIterations=100)

    count = 0
    while status[0] in (4,0) and count < 3:
        status = solve_single_lap_silhouette(V1, T, U, L, S, SN,
                                             lm_lambdas, lm_precond, 
                                             narrowBand=3,
                                             maxIterations=100)
        count += 1

    print 'Status (%d):' % status[0], status[1]

    test_visualise(V1, T, U, L, S)

if __name__ == '__main__':
    main()
    # generate_silhouette_info()
