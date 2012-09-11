# test_shortest_path.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "test_shortest_path.h":
    np.ndarray test_shortest_path_c "test_shortest_path" (
        np.ndarray V,
        np.ndarray T,
        np.ndarray S,
        np.ndarray SN,
        np.ndarray SilCandDistances,
        np.ndarray SilEdgeCands,
        np.ndarray SilEdgeCandParam,
        np.ndarray SilCandAssignedFaces,
        np.ndarray SilCandU,
        np.ndarray lambdas,
        bint isCircular)

# test_shortest_path
def test_shortest_path(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
                       np.ndarray[np.int32_t, ndim=2, mode='c'] T,
                       np.ndarray[np.float64_t, ndim=2, mode='c'] S, 
                       np.ndarray[np.float64_t, ndim=2, mode='c'] SN, 
                       np.ndarray[np.float64_t, ndim=2, mode='c'] SilCandDistances, 
                       np.ndarray[np.int32_t, ndim=2, mode='c'] SilEdgeCands, 
                       np.ndarray[np.float64_t, ndim=1] SilEdgeCandParam, 
                       np.ndarray[np.int32_t, ndim=1] SilCandAssignedFaces, 
                       np.ndarray[np.float64_t, ndim=2, mode='c'] SilCandU,
                       np.ndarray[np.float64_t, ndim=1] lambdas,
                       bint isCircular):

    return test_shortest_path_c(V, T, S, SN, SilCandDistances, SilEdgeCands, 
                         SilEdgeCandParam, SilCandAssignedFaces, SilCandU, lambdas, isCircular)

