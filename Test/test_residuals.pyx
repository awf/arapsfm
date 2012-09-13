# test_residuals.pyx
# cython:boundscheck=False
# cython:wraparound=False
# cython:cdivision=True

# Imports
import numpy as np
cimport numpy as np
np.import_array()

# Types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "test_residuals.h":
    void faceNormal_Unsafe(double *, double *, double *, double *)
    void faceNormalJac_Unsafe(double * a, double * b, double *c, double *J)
    void vertexNormal_(np.ndarray npy_Triangles,
                       np.ndarray npy_V1,
                       int vertexId,
                       double * normal)
    np.ndarray vertexNormalJac_(np.ndarray npy_Triangles,
                       np.ndarray npy_V1,
                       int vertexId)

    void silhouetteNormalResiduals_(np.ndarray  npy_Triangles,
                                    np.ndarray  npy_V1,
                                    int faceIndex,
                                    np.ndarray  npy_u,
                                    np.ndarray  npy_SN,
                                    double w,
                                    np.ndarray  npy_e)

    np.ndarray silhouetteNormalResidualsJac_V1_(np.ndarray npy_Triangles,
                                         np.ndarray npy_V1,
                                         int faceIndex,
                                         np.ndarray npy_u,
                                         int vertexIndex,
                                         double w)

    void silhouetteNormalResidualsJac_u_(np.ndarray npy_Triangles,
                                         np.ndarray npy_V1,
                                         int faceIndex,
                                         np.ndarray npy_u,
                                         double w,
                                         double * J)

def faceNormal(np.ndarray[DTYPE_t, ndim=1] V1i,
               np.ndarray[DTYPE_t, ndim=1] V1j,
               np.ndarray[DTYPE_t, ndim=1] V1k):

    cdef np.ndarray[DTYPE_t, ndim=1] n = np.empty(3, dtype=DTYPE)
    faceNormal_Unsafe(<DTYPE_t *>V1i.data, 
                <DTYPE_t *>V1j.data, 
                <DTYPE_t *>V1k.data, 
                <DTYPE_t *>n.data)
    return n

def faceNormalJac(np.ndarray[DTYPE_t, ndim=1] V1i,
                  np.ndarray[DTYPE_t, ndim=1] V1j,
                  np.ndarray[DTYPE_t, ndim=1] V1k):

    cdef np.ndarray[DTYPE_t, ndim=2] J = np.empty((3, 9), dtype=DTYPE)
    faceNormalJac_Unsafe(<DTYPE_t *>V1i.data, 
                   <DTYPE_t *>V1j.data, 
                   <DTYPE_t *>V1k.data, 
                   <DTYPE_t *>J.data)
    return J

def vertexNormal(np.ndarray T, np.ndarray V1, int i):
    cdef np.ndarray[DTYPE_t, ndim=1] normal = np.empty(3, dtype=DTYPE)

    vertexNormal_(T, V1, i, <DTYPE_t*>normal.data)
    
    return normal

def vertexNormalJac(T, V1, i):
    return vertexNormalJac_(T, V1, i)

def silhouetteNormalResiduals(T, V1, faceIndex, u, SN, w):
    cdef np.ndarray[DTYPE_t, ndim=1] e = np.empty(3, dtype=DTYPE)

    silhouetteNormalResiduals_(T, V1, faceIndex, u, SN, w, e)

    return e

def silhouetteNormalResidualsJac_V1(T, V1, faceIndex, u, vertexIndex, w):
    return silhouetteNormalResidualsJac_V1_(T, V1, faceIndex, u, vertexIndex, w)

def silhouetteNormalResidualsJac_u(T, V1, faceIndex, u, w):
    cdef np.ndarray[DTYPE_t, ndim=2] J = np.empty((3, 2), dtype=DTYPE)
    silhouetteNormalResidualsJac_u_(T, V1, faceIndex, u, w, <DTYPE_t*>J.data)

    return J



    

