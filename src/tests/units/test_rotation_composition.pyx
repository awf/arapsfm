# test_rotation_composition.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()
from pprint import pprint

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "test_rotation_composition.h":
    object EvaluateRotationComposition_c 'EvaluateRotationComposition' (
        object list_Xb,
        object list_y,
        int k,
        bint debug)

def EvaluateRotationComposition(list Xb, list y, int k, bint debug=False):
    return EvaluateRotationComposition_c(Xb, y, k, debug)

