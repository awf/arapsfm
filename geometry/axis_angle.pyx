# axis_angle.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from "Geometry/axis_angle.h":
    void axMakeInterpolated_Unsafe(double a, double * v, 
                                   double b, double * w,
                                   double * z)

def axMakeInterpolated(DTYPE_t a,
                       np.ndarray[DTYPE_t] v,
                       DTYPE_t b,
                       np.ndarray[DTYPE_t] w):
    if v.shape[0] != 3:
        raise ValueError('v.shape[0] != 3')

    if w.shape[0] != 3:
        raise ValueError('w.shape[0] != 3')

    cdef np.ndarray[DTYPE_t] z = np.empty(3, dtype=DTYPE)

    axMakeInterpolated_Unsafe(a, <double *>v.data, 
                              b, <double *>w.data,
                              <double *>z.data)

    return z

