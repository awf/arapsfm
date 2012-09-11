# test_mesh.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "test_mesh.h":
    int test_mesh_c "test_mesh" (np.ndarray vertices, np.ndarray triangles)

    np.ndarray get_nring_c "get_nring" (np.ndarray npy_vertices,
                         np.ndarray npy_triangles,
                         int vertex,
                         int N,
                         bint includeSource)



# test_mesh
def test_mesh(np.ndarray[np.float64_t, ndim=2, mode='c'] vertices, 
              np.ndarray[np.int32_t, ndim=2, mode='c'] triangles):

    test_mesh_c(vertices, triangles)

# get_nring
def get_nring(np.ndarray[np.float64_t, ndim=2, mode='c'] vertices, 
              np.ndarray[np.int32_t, ndim=2, mode='c'] triangles,
              int vertex, int N, bint includeSource):

    return get_nring_c(vertices, triangles, vertex, N, includeSource)
