# test_mesh.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

# Types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "test_mesh.h":
    int test_mesh_c "test_mesh" (int numVertices, np.ndarray npy_triangles)

# test_mesh
def test_mesh(int num_vertices, np.ndarray[np.int32_t, ndim=2, mode='c'] triangles):
    test_mesh_c(num_vertices, triangles)

