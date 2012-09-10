# run_test_problem.py

# Imports
from test_problem import *
from vtk_ import *

# load_model_1
def load_model_1():
    z = np.load('Models/BAR_PROJECTION.npz')
    return z['V'], z['T'], z['C'], z['P']
    
def view(V, T):
    T_ = faces_to_vtkCellArray(T)
    poly_data = numpy_to_vtkPolyData(V, T_)
    view_vtkPolyData(poly_data, camera_opt={'ParallelProjection' : True})

# main
def main():
    V, T, C, P = load_model_1()

    V1 = V.copy()
    V1[40, 1] -= 5.

    X = np.zeros_like(V)
    test_problem(V, T, X, V1, maxIterations=10)

    view(V1, T)

if __name__ == '__main__':
    main()

