# run_test_arap.py

# Imports
import numpy as np
from test_arap import EvaluateSingleARAP, EvaluateDualARAP
from pprint import pprint
from misc.scipy_ import approx_jac

# main_EvaluateSingleARAP
def main_EvaluateSingleARAP():
    T = np.r_[1, 2, 4,
              2, 3, 4,
              3, 0, 4,
              0, 1, 4].reshape(-1, 3).astype(np.int32)

    V = np.r_[0., 0., 0.,
              0., 2., 0.,
              2., 2., 0.,
              2., 0., 0.,
              1., 1., 1.].reshape(-1, 3).astype(np.float64)

    X = np.zeros((5, 3), dtype=np.float64)
    Xg = np.zeros((1, 3), dtype=np.float64)

    s = np.abs(np.random.randn(1).reshape(1, 1).astype(np.float64))

    V = np.random.randn(V.size).reshape(V.shape)
    V1 = np.random.randn(V.size).reshape(V.shape)

    k = 5

    print 'V:'
    print V
    print 'V1:'
    print V1

    r = EvaluateSingleARAP(T, V, X, Xg, s, V1, k, verbose=True)
    e, Jx, Jxg, Js, JV1i, JV1j = r
    print 'e:', e

    print '\nJx:'
    print Jx
    print 'approx_Jx:'
    approx_Jx = approx_jac(
        lambda x: EvaluateSingleARAP(T, V, x.reshape(V.shape), Xg, s, V1, k)[0], 
        np.ravel(X))
    print approx_Jx[:, 12:15]
    print 'allclose? ', np.allclose(Jx, approx_Jx[:, 12:15], atol=1e-3)

    print '\nJxg:' 
    print Jxg
    print 'approx_Jxg:'
    approx_Jxg = approx_jac(
        lambda x: EvaluateSingleARAP(T, V, X, x.reshape(1, 3), s, V1, k)[0], 
        np.ravel(Xg))
    print approx_Jxg
    print 'allclose? ', np.allclose(Jxg, approx_Jxg, atol=1e-3)

    print '\nJs:'
    print Js
    print 'approx_Js:'
    approx_Js = approx_jac(
        lambda x: EvaluateSingleARAP(T, V, X, Xg, x.reshape(1, 1), V1, k)[0], 
        np.ravel(s))
    print approx_Js
    print 'allclose? ', np.allclose(Js, approx_Js, atol=1e-3)

    print '\nJV1i:'
    print JV1i
    print 'approx_JV1i:'
    approx_JV1 = approx_jac(
        lambda x: EvaluateSingleARAP(T, V, X, Xg, s, x.reshape(V1.shape), k)[0], 
        np.ravel(V1))
    print approx_JV1[:, 12:15]

    print '\nJV1j:'
    print JV1j
    print 'approx_JV1j:'
    print approx_JV1[:, 6:9]

# main_EvaluateDualARAP
def main_EvaluateDualARAP():
    T = np.r_[1, 2, 4,
              2, 3, 4,
              3, 0, 4,
              0, 1, 4].reshape(-1, 3).astype(np.int32)

    V = np.r_[0., 0., 0.,
              0., 2., 0.,
              2., 2., 0.,
              2., 0., 0.,
              1., 1., 1.].reshape(-1, 3).astype(np.float64)

    X = np.zeros((5, 3), dtype=np.float64)
    Xg = np.zeros((1, 3), dtype=np.float64)

    s = np.r_[1.0].reshape(1, 1).astype(np.float64)
    V1 = V.copy()

    s = np.abs(np.random.randn(1).reshape(1, 1).astype(np.float64))
    V = np.random.randn(V.size).reshape(V.shape)
    V1 = np.random.randn(V1.size).reshape(V1.shape)
    X = np.random.randn(X.size).reshape(X.shape)
    Xg = np.random.randn(Xg.size).reshape(Xg.shape)

    print 's:'
    print s

    print 'X:'
    print X

    k = 5


    print 'V:'
    print V
    print 'V1:'
    print V1

    r = EvaluateDualARAP(T, V, X, Xg, s, V1, k, verbose=True)
    e, Jx, Jxg, Js, JV1i, JV1j, JVi, JVj = r
    print 'e:', e

    print '\nJx:'
    print Jx
    print 'approx_Jx:'
    approx_Jx = approx_jac(
        lambda x: EvaluateDualARAP(T, V, x.reshape(V.shape), Xg, s, V1, k)[0], 
        np.ravel(X))
    print approx_Jx[:, 12:15]
    print 'allclose? ', np.allclose(Jx, approx_Jx[:, 12:15], atol=1e-3)

    print '\nJxg:' 
    print Jxg
    print 'approx_Jxg:'
    approx_Jxg = approx_jac(
        lambda x: EvaluateDualARAP(T, V, X, x.reshape(1, 3), s, V1, k)[0], 
        np.ravel(Xg))
    print approx_Jxg
    print 'allclose? ', np.allclose(Jxg, approx_Jxg, atol=1e-3)

    print '\nJs:'
    print Js
    print 'approx_Js:'
    approx_Js = approx_jac(
        lambda x: EvaluateDualARAP(T, V, X, Xg, x.reshape(1, 1), V1, k)[0], 
        np.ravel(s))
    print approx_Js
    print 'allclose? ', np.allclose(Js, approx_Js, atol=1e-3)

    print '\nJV1i:'
    print JV1i
    print 'approx_JV1i:'
    approx_JV1 = approx_jac(
        lambda x: EvaluateDualARAP(T, V, X, Xg, s, x.reshape(V1.shape), k)[0], 
        np.ravel(V1))
    print approx_JV1[:, 12:15]
    print 'allclose? ', np.allclose(JV1i, approx_JV1[:, 12:15], atol=1e-3)

    print '\nJV1j:'
    print JV1j
    print 'approx_JV1j:'
    print approx_JV1[:, 6:9]
    print 'allclose? ', np.allclose(JV1j, approx_JV1[:, 6:9], atol=1e-3)

    print '\nJVi:'
    print JVi
    print 'approx_JVi:'
    approx_JV = approx_jac(
        lambda x: EvaluateDualARAP(T, x.reshape(V.shape), X, Xg, s, V1, k)[0], 
        np.ravel(V))
    print approx_JV[:, 12:15]
    print 'allclose? ', np.allclose(JVi, approx_JV[:, 12:15], atol=1e-3)

    print '\nJVj:'
    print JVj
    print 'approx_JVj:'
    print approx_JV[:, 6:9]
    print 'allclose? ', np.allclose(JVj, approx_JV[:, 6:9], atol=1e-3)

if __name__ == '__main__':
    # main_EvaluateSingleARAP()
    main_EvaluateDualARAP()

