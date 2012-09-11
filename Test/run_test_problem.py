# run_test_problem.py

# Imports
from test_problem import *
from visualise import visualise

# Barycentric conversion

# make_bary
def make_bary(u):
    u = u[:2]
    return np.r_[u, 1.0 - np.sum(u)]

# bary2pos
def bary2pos(V, u):
    return np.dot(u, V)

# load_model_1
def load_model_1():
    z = np.load('Models/BAR_PROJECTION.npz')
    return z['V'], z['T'], z['C'], z['P']

# load_model_3
def load_model_3():
    z = np.load('Models/CHIHUAHUA_PROJECTION_0B.npz')
    return z['V'], z['T'], z['C'], z['P']
    
# main_test_problem
def main_test_problem():
    V, T, C, P = load_model_3()

    V1 = V.copy()

    X = np.zeros_like(V)

    lambdas = np.array([1e+0, 1e+2], dtype=np.float64)
    status = test_problem(V, T, X, V1, C, P, lambdas, 
                          gradientThreshold=1e-6,
                          maxIterations=200)

    print 'Status:', status

    vis = visualise.VisualiseMesh(V1, T)
    vis.add_projection(C, P)
    vis.add_image('Frames/0.png')
    vis.execute()

    np.savez_compressed('MAIN_TEST_PROBLEM.npz', V1=V1)

# main_test_problem2
def main_test_problem2():
    _, T, C, P = load_model_3()
    z = np.load('MAIN_TEST_PROBLEM.npz')
    V = z['V1'].copy()

    lambdas = np.array([1e+0, 1e+2], dtype=np.float64)

    status = test_problem2(V, T, C, P, lambdas, 
                           gradientThreshold=1e-6,
                           maxIterations=5)

    vis = visualise.VisualiseMesh(V, T)
    vis.add_projection(C, P)
    vis.add_image('Frames/0.png')
    vis.execute()

# main_test_problem3
def main_test_problem3():
    _, T, C, P = load_model_3()
    z = np.load('MAIN_TEST_PROBLEM.npz')
    V = z['V1'].copy()

    z = np.load('MAIN_TEST_SHORTEST_PATH.npz')
    S = z['S']
    SN = z['SN']
    U = z['U']
    L = z['L']
    
    lambdas = np.array([1e1, 1e1], dtype=np.float64)
    preconditioners = np.array([1.0, 100.0], dtype=np.float64)

    status = test_problem3(V, T, U, L, S, SN, lambdas, preconditioners, 3,
        gradientThreshold=1e-6,
        maxIterations=15,
        verbosenessLevel=1)

    Q = np.empty((U.shape[0], 3), dtype=np.float64)
    for i, face_index in enumerate(L):
        Q[i] = bary2pos(V[T[face_index]], make_bary(U[i]))

    print 'Status:', status
    vis = visualise.VisualiseMesh(V, T, L)
    vis.add_image('Frames/0.png')
    vis.add_silhouette(Q, np.arange(Q.shape[0]), [0, S.shape[0] - 1], S)
    vis.execute()

if __name__ == '__main__':
    # main_test_problem()
    # main_test_problem2()
    main_test_problem3()

