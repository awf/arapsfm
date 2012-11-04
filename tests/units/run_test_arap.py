# run_test_arap.py

# Imports
import numpy as np
from test_arap import Evaluate
from pprint import pprint
from misc.scipy_ import approx_jac

# main
def main():
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

    # s = np.r_[1.0].reshape(1, 1).astype(np.float64)
    s = np.random.randn(1).reshape(1, 1).astype(np.float64)
    # print 's:', s

    V = np.random.randn(V.size).reshape(V.shape)
    V1 = np.random.randn(V.size).reshape(V.shape)
    # V1 = V.copy()
    # V1[4] += (0.0, 0.0, 1.0)

    k = 5

    print 'V:'
    print V
    print 'V1:'
    print V1

    r = Evaluate(T, V, X, Xg, s, V1, k, verbose=True)
    e, Jx, Jxg, Js, JV1i, JV1j = r
    print 'e:', e

    print '\nJx:'
    print Jx
    print 'approx_Jx:'
    approx_Jx = approx_jac(
        lambda x: Evaluate(T, V, x.reshape(V.shape), Xg, s, V1, k)[0], 
        np.ravel(X))
    print approx_Jx[:, 12:15]

    print '\nJxg:' 
    print Jxg
    print 'approx_Jxg:'
    approx_Jxg = approx_jac(
        lambda x: Evaluate(T, V, X, x.reshape(1, 3), s, V1, k)[0], 
        np.ravel(Xg))
    print approx_Jxg

    print '\nJs:'
    print Js
    print 'approx_Js:'
    approx_Js = approx_jac(
        lambda x: Evaluate(T, V, X, Xg, x.reshape(1, 1), V1, k)[0], 
        np.ravel(s))
    print approx_Js

    print '\nJV1i:'
    print JV1i
    print 'approx_JV1i:'
    approx_JV1i = approx_jac(
        lambda x: Evaluate(T, V, X, Xg, X, x.reshape(V1.shape), k)[0], 
        np.ravel(V1))
    print approx_JV1i[:, 12:15]

    print '\nJV1j:'
    print JV1j
    print 'approx_JV1j:'
    approx_JV1j = approx_jac(
        lambda x: Evaluate(T, V, X, Xg, X, x.reshape(V1.shape), k)[0], 
        np.ravel(V1))
    print approx_JV1j[:, 6:9]

if __name__ == '__main__':
    main()
