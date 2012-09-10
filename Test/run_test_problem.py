# run_test_problem.py

# Imports
from test_problem import *
from vtk_ import *

# load_model_1
def load_model_1():
    z = np.load('Models/BAR_PROJECTION.npz')
    return z['V'], z['T'], z['C'], z['P']
    
def view(V, T, C):
    T_ = faces_to_vtkCellArray(T)
    poly_data = numpy_to_vtkPolyData(V, T_)
    view_vtkPolyData(poly_data, camera_opt={'ParallelProjection' : False},
                     highlight=C)

# main
def main():
    V, T, C, P = load_model_1()

    V1 = V.copy()

    X = np.zeros_like(V)

    lambdas = np.array([1e+0, 1e+0], dtype=np.float64)
    status = test_problem(V, T, X, V1, C, P, lambdas, 
                          gradientThreshold=1e-12,
                          maxIterations=100)

    print 'Status:', status
    view(V1, T, C)

if __name__ == '__main__':
    main()

