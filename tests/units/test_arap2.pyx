# test_arap2.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "test_arap2.h":
    object EvaluateCompleteSectionedBasisArapEnergy_c 'EvaluateCompleteSectionedBasisArapEnergy' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_s,
        int n, 
        list list_Xgb,
        list list_yg,
        np.ndarray npy_Xg,
        np.ndarray npy_k,
        np.ndarray npy_Xb,
        np.ndarray npy_y,
        np.ndarray npy_X, 
        np.ndarray npy_V1, 
        int k_,
        np.ndarray npy_jacDims,
        bint debug) 

def EvaluateCompleteSectionedBasisArapEnergy(T, V, s, n, Xgb, yg, Xg, k, Xb, y, X, V1, k_, jacDims, debug=False):
    return EvaluateCompleteSectionedBasisArapEnergy_c(T, V, s, n, Xgb, yg, Xg, k, Xb, y, X, V1, k_, jacDims, debug)
    
