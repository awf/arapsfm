# test_rigid.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "test_rigid.h":
    object EvaluateRigidRegistrationEnergy_c 'EvaluateRigidRegistrationEnergy' (
        np.ndarray npy_V0,
        np.ndarray npy_V,
        np.ndarray npy_s,
        np.ndarray npy_xg,
        np.ndarray npy_d,
        double w,
        int k,
        bint useBackward,
        bint debug)

def EvaluateRigidRegistrationEnergy(np.ndarray[DTYPE_t, ndim=2] V0,
                                    np.ndarray[DTYPE_t, ndim=2] V,
                                    np.ndarray[DTYPE_t, ndim=2] s,
                                    np.ndarray[DTYPE_t, ndim=2] xg,
                                    np.ndarray[DTYPE_t, ndim=2] d,
                                    DTYPE_t w,
                                    np.int32_t k,
                                    bint useBackward=False,
                                    bint debug=False):
    
    return EvaluateRigidRegistrationEnergy_c(V0, V, s, xg, d, w, k,
                                             useBackward, debug)

