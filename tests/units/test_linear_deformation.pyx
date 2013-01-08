# test_linear_deformation.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "test_linear_deformation.h":
    object EvaluateLinearDeformationEnergy_c 'EvaluateLinearDeformationEnergy' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_s,
        int n, 
        list list_Xgb,
        list list_yg,
        np.ndarray npy_Xg,
        np.ndarray npy_dg,
        np.ndarray npy_V1,
        double w,
        int k_,
        np.ndarray npy_jacDims,
        bint debug)

def EvaluateLinearDeformationEnergy(np.ndarray[np.int32_t, ndim=2] T,
                                    np.ndarray[DTYPE_t, ndim=2] V,
                                    np.ndarray[DTYPE_t, ndim=2] s,
                                    np.int32_t n,
                                    list Xgb,
                                    list yg,
                                    np.ndarray[DTYPE_t, ndim=2] xg,
                                    np.ndarray[DTYPE_t, ndim=2] dg,
                                    np.ndarray[DTYPE_t, ndim=2] V1,
                                    DTYPE_t w,
                                    np.int32_t k,
                                    np.ndarray[np.int32_t, ndim=2] jacDims,
                                    bint debug=False):
    return EvaluateLinearDeformationEnergy_c(
        T, V, s, n, Xgb, yg, xg, dg, V1, w, k, jacDims, debug)

