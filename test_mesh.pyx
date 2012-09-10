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

# test_mesh
def test_mesh(np.ndarray[np.float64_t, ndim=2, mode='c'] vertices, 
              np.ndarray[np.int32_t, ndim=2, mode='c'] triangles):

    test_mesh_c(vertices, triangles)

