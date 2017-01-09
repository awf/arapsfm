# test_silhouette.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "test_silhouette.h":
    object EvaluateSilhouetteNormal2_c 'EvaluateSilhouetteNormal2' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        int faceIndex,
        np.ndarray npy_u,
        np.ndarray npy_sn,
        double w,
        bint debug)

    object EvaluateSilhouette2Jac_V1_c 'EvaluateSilhouette2Jac_V1' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        int faceIndex,
        np.ndarray npy_u,
        np.ndarray npy_sn,
        int vertexIndex,
        double w,
        bint debug)

    object EvaluateSilhouette2Jac_u_c 'EvaluateSilhouette2Jac_u' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        int faceIndex,
        np.ndarray npy_u,
        np.ndarray npy_sn,
        double w,
        bint debug)

def EvaluateSilhouetteNormal2(np.ndarray[np.int32_t, ndim=2] T,
                              np.ndarray[DTYPE_t, ndim=2] V,
                              np.int32_t faceIndex,
                              np.ndarray[DTYPE_t, ndim=1] u,
                              np.ndarray[DTYPE_t, ndim=1] sn,
                              DTYPE_t w,
                              bint debug=False):

    return EvaluateSilhouetteNormal2_c(T, V, faceIndex, u, sn, w, debug)

def EvaluateSilhouette2Jac_V1(np.ndarray[np.int32_t, ndim=2] T,
                              np.ndarray[DTYPE_t, ndim=2] V,
                              np.int32_t faceIndex,
                              np.ndarray[DTYPE_t, ndim=1] u,
                              np.ndarray[DTYPE_t, ndim=1] sn,
                              np.int32_t vertexIndex,
                              DTYPE_t w,
                              bint debug=False):

    return EvaluateSilhouette2Jac_V1_c(T, V, faceIndex, u, sn, vertexIndex, w, debug)

def EvaluateSilhouette2Jac_u(np.ndarray[np.int32_t, ndim=2] T,
                              np.ndarray[DTYPE_t, ndim=2] V,
                              np.int32_t faceIndex,
                              np.ndarray[DTYPE_t, ndim=1] u,
                              np.ndarray[DTYPE_t, ndim=1] sn,
                              DTYPE_t w,
                              bint debug=False):

    return EvaluateSilhouette2Jac_u_c(T, V, faceIndex, u, sn, w, debug)
