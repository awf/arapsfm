# run_test_shortest_path.py

# Imports
import numpy as np
from vtk_ import *
from visualise import visualise
from test_shortest_path import *

# load_silhouette_information
def load_silhouette_information():
    z = np.load('Silhouette/CHIHUAHUA_SILHOUETTE_0.npz')
    return {k:z[k] for k in z.keys()}

# load_model_3
def load_model_3():
    z = np.load('Models/CHIHUAHUA_PROJECTION_0B.npz')
    return z['V'], z['T'], z['C'], z['P']

# Barycentric conversion

# make_bary
def make_bary(u):
    u = u[:2]
    return np.r_[u, 1.0 - np.sum(u)]

# bary2pos
def bary2pos(V, u):
    return np.dot(u, V)

# main_test_shortest_path
def main_test_shortest_path():
    _, T, C, P = load_model_3()
    # requires output from `main_test_problem` under run_test_problem.py
    z = np.load('MAIN_TEST_PROBLEM.npz')
    V = z['V1'].copy()

    lambdas = np.array([1e-3, 1e1, 1e3], dtype=np.float64)

    silhouette = load_silhouette_information()

    path = test_shortest_path(V, T, isCircular=True,
                              lambdas=lambdas, **silhouette)

    # convert ALL candidates to 3D positions
    SilCandAssignedFaces = silhouette['SilCandAssignedFaces']
    SilCandU = silhouette['SilCandU']
    S = silhouette['S']
    SN = silhouette['SN']

    Q = np.empty((SilCandAssignedFaces.shape[0], 3), dtype=np.float64)
    for i, face_index in enumerate(SilCandAssignedFaces):
        Q[i] = bary2pos(V[T[face_index]], make_bary(SilCandU[i]))

    # get assigned faces
    L = SilCandAssignedFaces[path]
    U = SilCandU[path]
    np.savez_compressed('MAIN_TEST_SHORTEST_PATH.npz', L=L, U=U, S=S, SN=SN)

    vis = visualise.VisualiseMesh(V, T, L)
    vis.add_image('Frames/0.png')
    vis.add_silhouette(Q, path, [0, S.shape[0] - 1], S)
    vis.execute()

if __name__ == '__main__':
    main_test_shortest_path()

