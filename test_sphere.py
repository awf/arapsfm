# test_sphere.py

# Imports
from visualise.visualise import *
from visualise.vtk_ import *
from mesh import faces, geometry

from core_recovery.silhouette_candidates import \
    generate_silhouette_candidate_info

from core_recovery.silhouette_global_solver import \
    shortest_path_solve

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

# main
def main():
    V, T = test_sphere()
    print '# Vertices:', V.shape[0]
    print '# Triangles:', T.shape[0]

    sil_lambdas = np.array([1e-3, 1.0, 1e3], dtype=np.float64)
    print 'Silhouette lambdas:', sil_lambdas

    S, SN = test_silhouette()

    U, L = shortest_path_solve(V, T, S, SN,
                               lambdas=sil_lambdas,
                               isCircular=True,
                               **test_sphere_silhouette_info())

    Q = geometry.path2pos(V, T, L, U)
    N = Q.shape[0]

    vis = VisualiseMesh(V, T)
    vis.add_image('data/segmentations/circle/0-INV_S.png')
    vis.add_silhouette(Q, np.arange(N), [0, N-1], S)
    vis.execute()

if __name__ == '__main__':
    main()
    # generate_silhouette_info()
