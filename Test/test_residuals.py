# test_residuals.py
import numpy as np
from test_residuals import *
from scipy.optimize import approx_fprime
from scipy.linalg import norm

from mesh import faces

# General

# approx_jac
def approx_jac(f, x, epsilon=1e-4):
    x = np.asarray(x)
    e = f(x)
    J = np.empty((e.shape[0], x.shape[0]), dtype=np.float64)
    for i in xrange(e.shape[0]):
        fi = lambda x: f(x)[i]
        J[i] = approx_fprime(x, fi, epsilon)

    return J

# test_faceNormal
def test_faceNormal():
    Vi, Vj, Vk = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)

    print np.cross(Vi - Vk, Vj - Vk)
    print faceNormal(Vi, Vj, Vk)

    Vi, Vj, Vk = np.random.rand(9).reshape(3,3).astype(np.float64)

    print np.cross(Vi - Vk, Vj - Vk)
    print faceNormal(Vi, Vj, Vk)

# test_faceNormalJac
def test_faceNormalJac():
    Vi, Vj, Vk = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)

    x = np.r_[Vi, Vj, Vk]

    def f(x):
        Vi, Vj, Vk = np.split(x, 3)
        return faceNormal(Vi, Vj, Vk)

    print approx_jac(f, x)
    print faceNormalJac(Vi, Vj, Vk)

    Vi, Vj, Vk = np.random.rand(9).reshape(3,3).astype(np.float64)
    x = np.r_[Vi, Vj, Vk]

    print approx_jac(f, x)
    print faceNormalJac(Vi, Vj, Vk)

# test_vertexNormal
def test_vertexNormal():
    # configuration
    T = np.array([[0, 1, 2],
                  [0, 2, 3],
                  [0, 3, 4],
                  [0, 4, 5],
                  [0, 5, 6],
                  [0, 6, 1]], dtype=np.int32)

    V1 = np.empty((7, 3), dtype=np.float64)
    V1[0] = (0., 0., 3.)

    t = np.linspace(0, 2*np.pi, 6, endpoint=False)
    V1[1:,0] = np.cos(t)
    V1[1:,1] = np.sin(t)
    V1[1:,2] = 0.
    print V1
    V1 = np.random.rand(21).reshape(7,3)

    # vertexNormal
    print 'vertexNormal:'
    print vertexNormal(T, V1, 0)

    # "manual" vertexNormal
    trialNormal = np.zeros(3, dtype=np.float64)
    for Ti in T:
        trialNormal += faceNormal(*V1[Ti])

    print 'manual vertex normal:'
    print trialNormal

    # vertexNormalJac
    vertexIndex = 2

    J = vertexNormalJac(T, V1, vertexIndex)
    print 'J:'
    print np.around(J, decimals=2)
    print 'J.shape:', J.shape

    # approximate jacobian
    def f(x):
        V1 = x.reshape(-1, 3)
        n = vertexNormal(T, V1, vertexIndex)
        return n

    approx_J = approx_jac(f, V1.flatten(), epsilon=1e-6)
    print 'approx_J:'
    print np.around(approx_J, decimals=2)[:,:-9]
    print 'approx_J.shape:', approx_J.shape

    # NOTE can't compare directly because `vertexNormalJac` only returns Jacobian
    # over the vertices in the one ring
    # print 'allclose? ', np.allclose(approx_J, J, atol=1e-4)

# test_silhouetteNormalResiduals
def test_silhouetteNormalResiduals():
    # configuration
    T = np.array([[0, 1, 2],
                  [0, 2, 3],
                  [0, 3, 4],
                  [0, 4, 5],
                  [0, 5, 6],
                  [0, 6, 1]], dtype=np.int32)

    V1 = np.empty((7, 3), dtype=np.float64)
    V1[0] = (0., 0., 1.)

    t = np.linspace(0, 2*np.pi, 6, endpoint=False)
    V1[1:,0] = np.cos(t)
    V1[1:,1] = np.sin(t)
    V1[1:,2] = 0.

    # silhouetteNormalResiduals
    faceIndex = 0
    u = np.r_[0.4, 0.2].astype(np.float64)
    SN = np.r_[0, 0].astype(np.float64)
    w = 1.0

    print 'silhouetteNormalResiduals'
    print silhouetteNormalResiduals(T, V1, faceIndex, u, SN, w)

    # silhouetteNormalResidualsJac_V1
    print 'silhouetteNormalResidualsJac_V1'
    J = np.hstack(silhouetteNormalResidualsJac_V1(T, V1, faceIndex, u, i, w) 
                  for i in xrange(V1.shape[0]))
    print 'J:'
    print np.around(J, decimals=3)
    print 'J.shape:', J.shape

    # approx J (V1)
    def f(x):
        V1 = x.reshape(-1, 3)
        e = silhouetteNormalResiduals(T, V1, faceIndex, u, SN, w)

        return e

    approx_J = approx_jac(f, V1.flatten(), epsilon=1e-6)
    print 'approx_J:'
    print np.around(approx_J, decimals=3)
    print 'approx_J.shape:', approx_J.shape

    print 'allclose? ', np.allclose(approx_J, J, atol=1e-3)

    # silhouetteNormalResidualsJac_u
    print 'silhouetteNormalResidualsJac_u'
    J = silhouetteNormalResidualsJac_u(T, V1, faceIndex, u, w)

    print 'J:'
    print np.around(J, decimals=3)
    print 'J.shape:', J.shape

    # approx J (u)
    def f(u):   
        e = silhouetteNormalResiduals(T, V1, faceIndex, u, SN, w)
        return e

    approx_J = approx_jac(f, u, epsilon=1e-6)

    print 'approx_J:'
    print np.around(approx_J, decimals=3)
    print 'approx_J.shape:', approx_J.shape

    print 'allclose? ', np.allclose(approx_J, J, atol=1e-3)

if __name__ == '__main__':
    #test_faceNormal()
    #test_faceNormalJac()
    #test_vertexNormal()
    test_silhouetteNormalResiduals()

