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
    void axScale_Unsafe(double s, double * x, double * y)
    void axAdd_Unsafe(double * a, double * b, double * c)
    void axMakeInterpolated_Unsafe(double a, double * v, 
                                   double b, double * w,
                                   double * z)

# axScale
def axScale(DTYPE_t s, np.ndarray[DTYPE_t] x):
    if x.shape[0] != 3:
        raise ValueError('x.shape[0] != 3')

    cdef np.ndarray[DTYPE_t] y = np.empty(3, dtype=DTYPE)
    axScale_Unsafe(s, <double *>x.data, <double *>y.data)

    return y

# axAdd
def axAdd(np.ndarray[DTYPE_t] a, np.ndarray[DTYPE_t] b):
    if a.shape[0] != 3:
        raise ValueError('a.shape[0] != 3')
    
    if b.shape[0] != 3:
        raise ValueError('b.shape[0] != 3')

    cdef np.ndarray[DTYPE_t] c = np.empty(3, dtype=DTYPE)
    axAdd_Unsafe(<double *>a.data, 
                 <double *>b.data, 
                 <double *>c.data)

    return c
                
# axMakeInterpolated
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

