# run_test_problem.py

# Imports
from test_problem import *
from visualise import visualise

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

if __name__ == '__main__':
    # main_test_problem()
    main_test_problem2()

