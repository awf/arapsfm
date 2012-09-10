# run_test_mesh.py

# Imports
import numpy as np
from test_mesh import *

# test_geometry_1
def test_geometry_1():
    T = np.array([[5, 1, 0],
                  [0, 1, 2],
                  [0, 2, 3],
                  [0, 3, 4],
                  [0, 4, 5]], dtype=np.int32)

    V = np.empty((6, 3), dtype=np.float64)
    V[0] = (0., 0., 0.)

    t = np.linspace(0, 2*np.pi, 5, endpoint=False)
    V[1:,0] = np.cos(t)
    V[1:,1] = np.sin(t)
    V[1:,2] = 0.

    return V, T

# test_geometry_2
def test_geometry_2():
    T = np.array([[0, 1, 2]], dtype=np.int32)

    V = np.empty((3, 3), dtype=np.float64)
    V[0] = (0., 0., 0.)
    V[1] = (0., 1., 0.)
    V[2] = (1., 0., 0.)

    return V, T

# test_geometry_3
def test_geometry_3():
    T = np.array([[0, 1, 3],
                  [1, 4, 3],
                  [1, 2, 4],
                  [2, 5, 4],
                  [3, 4, 6],
                  [6, 4, 7],
                  [4, 5, 7],
                  [5, 8, 7]], dtype=np.int32)

    V = np.empty((9, 3), dtype=np.float64)
    i = np.arange(3)
    V[:, 0] = np.tile(i, 3)
    V[:, 1] = 2. - np.repeat(i, 3)
    V[:, 2] = 0.

    return V, T

# main
def main():
    V, T = test_geometry_3()

    test_mesh(V, T)

if __name__ == '__main__':
    main()
