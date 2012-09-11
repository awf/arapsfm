# test_mesh_walker.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "test_mesh_walker.h":
    void apply_displacement_c "apply_displacement" (np.ndarray V,
        np.ndarray T,
        np.ndarray U,
        np.ndarray L,
        np.ndarray delta)

# apply_displacement
def apply_displacement(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                       np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                       np.ndarray[np.float64_t, ndim=2, mode='c'] U,
                       np.ndarray[np.int32_t, ndim=1] L,
                       np.ndarray[np.float64_t, ndim=2, mode='c'] delta):

    apply_displacement_c(V, T, U, L, delta)

