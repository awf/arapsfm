# test_linear_basis_shape.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "test_linear_basis_shape.h":
    object EvaluateLinearBasisShape_c 'EvaluateLinearBasisShape' (
        object list_Vb,
        np.ndarray npy_y,
        np.ndarray npy_s,
        np.ndarray npy_Xg,
        np.ndarray npy_Vd)

def EvaluateLinearBasisShape(Vb, y, s, Xg, Vd):
    return EvaluateLinearBasisShape_c(Vb, y, s, Xg, Vd)

