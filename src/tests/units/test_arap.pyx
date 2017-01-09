# test_arap.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "test_arap.h":
    object EvaluateSingleARAP_c 'EvaluateSingleARAP' (np.ndarray npy_T,
                                                      np.ndarray npy_V,
                                                      np.ndarray npy_X,
                                                      np.ndarray npy_Xg,
                                                      np.ndarray npy_s,
                                                      np.ndarray npy_V1,
                                                      int k,
                                                      bint verbose)

    object EvaluateSingleARAP2_c 'EvaluateSingleARAP2' (np.ndarray npy_T,
                                                        np.ndarray npy_V,
                                                        np.ndarray npy_X,
                                                        np.ndarray npy_Xg,
                                                        np.ndarray npy_s,
                                                        np.ndarray npy_V1,
                                                        int k,
                                                        bint verbose)

    object EvaluateDualARAP_c 'EvaluateDualARAP' (np.ndarray npy_T,
                                                  np.ndarray npy_V,
                                                  np.ndarray npy_X,
                                                  np.ndarray npy_Xg,
                                                  np.ndarray npy_s,
                                                  np.ndarray npy_V1,
                                                  int k,
                                                  bint verbose)

    object EvaluateDualNonLinearBasisARAP_c 'EvaluateDualNonLinearBasisARAP' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_Xg,
        np.ndarray npy_s,
        list py_Xs,
        np.ndarray py_y,
        np.ndarray npy_V1,
        int k,
        bint verbose)

    object EvaluateSectionedBasisArapEnergy_c 'EvaluateSectionedBasisArapEnergy' (
        np.ndarray npy_T,
        np.ndarray npy_V,
        np.ndarray npy_Xg,
        np.ndarray npy_s,
        np.ndarray npy_Xb,
        np.ndarray npy_y,
        np.ndarray npy_X,
        np.ndarray npy_V1,
        np.ndarray npy_K,
        int k,
        np.ndarray npy_jacDims,
        bint verbose)

    object EvaluateSectionedRotationsVelocityEnergy_c 'EvaluateSectionedRotationsVelocityEnergy' (
        np.ndarray npy_Xg0,
        np.ndarray npy_y0,
        np.ndarray npy_X0,
        np.ndarray npy_Xg, 
        np.ndarray npy_y,
        np.ndarray npy_X,
        np.ndarray npy_Xb,
        np.ndarray npy_K,
        int k,
        np.ndarray npy_jacDims,        
        bint verbose)

def EvaluateSingleARAP(T, V, X, Xg, s, V1, k, verbose=False):
    return EvaluateSingleARAP_c(T, V, X, Xg, s, V1, k, verbose)

def EvaluateSingleARAP2(T, V, X, Xg, s, V1, k, verbose=False):
    return EvaluateSingleARAP2_c(T, V, X, Xg, s, V1, k, verbose)
   
def EvaluateDualARAP(T, V, X, Xg, s, V1, k, verbose=False):
    return EvaluateDualARAP_c(T, V, X, Xg, s, V1, k, verbose)

def EvaluateDualNonLinearBasisARAP(T, V, Xg, s, Xs, y, V1, k, verbose=False):
    return EvaluateDualNonLinearBasisARAP_c(T, V, Xg, s, Xs, y, V1, k, verbose)

def EvaluateSectionedBasisArapEnergy(T, V, Xg, s, Xb, y, X, V1, K, k, jacDims, verbose=False):
    return EvaluateSectionedBasisArapEnergy_c(T, V, Xg, s, Xb, y, X, V1, K, k, jacDims, verbose)

def EvaluateSectionedRotationsVelocityEnergy(Xg0, y0, X0, Xg, y, X, Xb, K, k, jacDims, verbose=False):
    return EvaluateSectionedRotationsVelocityEnergy_c(Xg0, y0, X0, Xg, y, X, Xb, K, k, jacDims, verbose)

