# silhouette_global_solver.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "silhouette_global_solver.h":
    np.ndarray shortest_path_solve_c "shortest_path_solve" (
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

# shortest_path_solve
def shortest_path_solve(np.ndarray[np.float64_t, ndim=2, mode='c'] V,
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

    cdef np.ndarray path = shortest_path_solve_c(V, T, S, SN, 
        SilCandDistances, SilEdgeCands, SilEdgeCandParam, 
        SilCandAssignedFaces, SilCandU, lambdas, isCircular)

    return SilCandU[path], SilCandAssignedFaces[path]

