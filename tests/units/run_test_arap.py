# run_test_arap.py

# Imports
import numpy as np
from test_arap import *
from pprint import pprint
from misc.scipy_ import approx_jac
from operator import itemgetter

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

# main_EvaluateSingleARAP2
def main_EvaluateSingleARAP2():
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

    r = EvaluateSingleARAP2(T, V, X, Xg, s, V1, k, verbose=True)
    e, Jx, Jxg, Js, JV1i, JV1j = r
    print 'e:', e

    print '\nJx:'
    print Jx
    print 'approx_Jx:'
    approx_Jx = approx_jac(
        lambda x: EvaluateSingleARAP2(T, V, x.reshape(V.shape), Xg, s, V1, k)[0], 
        np.ravel(X))
    print approx_Jx[:, 12:15]
    print 'allclose? ', np.allclose(Jx, approx_Jx[:, 12:15], atol=1e-3)

    print '\nJxg:' 
    print Jxg
    print 'approx_Jxg:'
    approx_Jxg = approx_jac(
        lambda x: EvaluateSingleARAP2(T, V, X, x.reshape(1, 3), s, V1, k)[0], 
        np.ravel(Xg))
    print approx_Jxg
    print 'allclose? ', np.allclose(Jxg, approx_Jxg, atol=1e-3)

    print '\nJs:'
    print Js
    print 'approx_Js:'
    approx_Js = approx_jac(
        lambda x: EvaluateSingleARAP2(T, V, X, Xg, x.reshape(1, 1), V1, k)[0], 
        np.ravel(s))
    print approx_Js
    print 'allclose? ', np.allclose(Js, approx_Js, atol=1e-3)

    print '\nJV1i:'
    print JV1i
    print 'approx_JV1i:'
    approx_JV1 = approx_jac(
        lambda x: EvaluateSingleARAP2(T, V, X, Xg, s, x.reshape(V1.shape), k)[0], 
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

# main_EvaluateDualNonLinearBasisARAP
def main_EvaluateDualNonLinearBasisARAP():
    T = np.r_[1, 2, 4,
              2, 3, 4,
              3, 0, 4,
              0, 1, 4].reshape(-1, 3).astype(np.int32)

    V = np.r_[0., 0., 0.,
              0., 2., 0.,
              2., 2., 0.,
              2., 0., 0.,
              1., 1., 1.].reshape(-1, 3).astype(np.float64)

    Xg = np.zeros((1, 3), dtype=np.float64)
    # s = np.r_[1.0].reshape(1, 1).astype(np.float64)
    s = np.r_[np.random.rand()].reshape(1, 1).astype(np.float64)

    # n = 1
    # Xs = [np.zeros((5, 3), dtype=np.float64) for i in xrange(n)]
    # Xs[0][4, -1] = np.pi
    # ys = [np.r_[1.0 / n].reshape(1, 1).astype(np.float64) for i in xrange(n)]

    n = 5
    Xs = [np.random.randn(V.size).reshape(V.shape) for i in xrange(n)]
    y = np.random.randn(n).reshape(n, 1)
    V1 = V.copy()

    k = 5

    print 's:'
    print s
    print 'V:'
    print V
    print 'V1:'
    print V1

    r = EvaluateDualNonLinearBasisARAP(T, V, Xg, s, Xs, y, V1, k, verbose=True)
    e, Jxg, Js, JVi, JVj, JV1i, JV1j = r[:7]
    print 'e:', e

    JXs = r[7:7+n]
    Jy = r[7+n:]

    print '\nJxg:' 
    print np.around(Jxg, decimals=4)
    print 'approx_Jxg:'
    approx_Jxg = approx_jac(
        lambda x: EvaluateDualNonLinearBasisARAP(
            T, V, x.reshape(Xg.shape), s, Xs, y, V1, k)[0], 
            np.ravel(Xg))

    print np.around(approx_Jxg, decimals=4)
    print 'allclose? ', np.allclose(Jxg, approx_Jxg, atol=1e-3)

    print '\nJs:' 
    print np.around(Js, decimals=4)
    print 'approx_Js:'
    approx_Js = approx_jac(
        lambda x: EvaluateDualNonLinearBasisARAP(
            T, V, Xg, x.reshape(1, 1), Xs, y, V1, k)[0], np.ravel(s),
            epsilon=1e-6)
    print np.around(approx_Js, decimals=4)
    print 'allclose? ', np.allclose(Js, approx_Js, atol=1e-3)

    print '\nJVi:' 
    print np.around(JVi, decimals=4)
    print 'approx_JVi:'
    approx_JV = approx_jac(
        lambda x: EvaluateDualNonLinearBasisARAP(
            T, x.reshape(V.shape), Xg, s, Xs, y, V1, k)[0], 
            np.ravel(V))
    print np.around(approx_JV[:, 12:15], decimals=4)
    print 'allclose? ', np.allclose(JVi, 
        approx_JV[:, 12:15], atol=1e-3)

    print '\nJVj:' 
    print np.around(JVj, decimals=4)
    print 'approx_JVj:'
    print np.around(approx_JV[:, 6:9], decimals=4)
    print 'allclose? ', np.allclose(JVj, 
        approx_JV[:, 6:9], atol=1e-3)

    print '\nJV1i:' 
    print np.around(JV1i, decimals=4)
    print 'approx_JV1i:'
    approx_JV1 = approx_jac(
        lambda x: EvaluateDualNonLinearBasisARAP(
            T, V, Xg, s, Xs, y, x.reshape(V1.shape), k)[0], 
            np.ravel(V1))
    print np.around(approx_JV1[:, 12:15], decimals=4)
    print 'allclose? ', np.allclose(JV1i, 
        approx_JV1[:, 12:15], atol=1e-3)

    print '\nJV1j:' 
    print np.around(JV1j, decimals=4)
    print 'approx_JV1j:'
    print np.around(approx_JV1[:, 6:9], decimals=4)
    print 'allclose? ', np.allclose(JV1j, 
        approx_JV1[:, 6:9], atol=1e-3)

    for i in xrange(n):
        print '\nJX[%d]' % i
        print np.around(JXs[i], decimals=4)
    
        def f(x):
            _Xs = Xs[:]
            _Xs[i] = x.reshape(_Xs[i].shape)
            return EvaluateDualNonLinearBasisARAP(
                T, V, Xg, s, _Xs, y, V1, k)[0]

        approx_JX = approx_jac(f, np.ravel(Xs[i]))
        print 'approx_JX[%d]' % i
        print np.around(approx_JX[:, 12:15], decimals=4)
        print 'allclose? ', np.allclose(
            JXs[i], approx_JX[:, 12:15], atol=1e-3)

    print '\nJy:' 
    Jy = np.hstack(Jy)

    print np.around(Jy, decimals=4)
    print 'approx_Jy:'
    approx_Jy = approx_jac(
        lambda x: EvaluateDualNonLinearBasisARAP(
            T, V, Xg, s, Xs, x[:, np.newaxis], V1, k)[0], np.ravel(y))
    print np.around(approx_Jy, decimals=4)
    print 'allclose? ', np.allclose(Jy, approx_Jy, atol=1e-3)

    return 

# approx_jacs
def approx_jacs(f, indices, epsilon, *args, **kwargs):
    args = list(args)
    jacs = []

    for i in indices:
        orig_x = args[i]

        def wrap_f(x):
            args[i] = x.reshape(orig_x.shape)
            return f(*args, **kwargs)

        jacs.append(approx_jac(wrap_f, np.ravel(orig_x), epsilon))

        args[i] = orig_x

    return jacs

# print_comparison
def print_comparison(**kwargs):
    atol = kwargs.pop('atol', 1e-4)
    decimals = kwargs.pop('decimals', 3)

    items = sorted(kwargs.items(), key=itemgetter(0))

    for key, arr in items:
        print '%s:\n%s' % (key, np.around(arr, decimals=decimals))

    n = len(items)
    r = np.empty(n-1, dtype=bool)
    for i in xrange(n-1):
        r[i] = np.allclose(items[i][1], items[i+1][1], atol=atol)

    print 'allclose? ', np.all(r)

# main_EvaluateSectionedBasisArapEnergy
def main_EvaluateSectionedBasisArapEnergy():
    T = np.r_[1, 2, 4,
              2, 3, 4,
              3, 0, 4,
              0, 1, 4].reshape(-1, 3).astype(np.int32)

    V = np.r_[0., 0., 0.,
              0., 2., 0.,
              2., 2., 0.,
              2., 0., 0.,
              1., 1., 1.].reshape(-1, 3).astype(np.float64)

    Xg = np.zeros((1, 3), dtype=np.float64)
    s = np.ones((1, 1), dtype=np.float64)

    s[0,0] = 1.0

    # Test K[4,0] == 0 (fixed)

    k = 8 # (i, j) = (4, 3)

    K = np.r_[0, 0,
              0, 0,
              0, 0,
              0, 0,
              0, 0].reshape(-1, 2).astype(np.int32)

    Xb = np.array(tuple(), dtype=np.float64).reshape(0, 3)
    y = np.array(tuple(), dtype=np.float64).reshape(0, 1)
    X = np.array(tuple(), dtype=np.float64).reshape(0, 3)

    V1 = V.copy()
    V1[3,0] += 1.0

    randomise = lambda arr: np.random.randn(arr.size).reshape(arr.shape)

    V1 = randomise(V1)
    V = randomise(V)
    s = randomise(s)
    Xg = randomise(Xg)

    empty_jacDims = np.array(tuple(), dtype=np.int32).reshape(0, 0)
    jacDims = np.r_[3, 3, 
                    3, 3,
                    3, 3,
                    3, 1].reshape(-1, 2).astype(np.int32)

    print 'Xb:', repr(Xb)
    print 'y:', repr(y)
    print 'X:', repr(X)
    print 'jacDims:', repr(jacDims)

    r, JXg, JV1i, JV1j, Js = EvaluateSectionedBasisArapEnergy(T, V, Xg, s, Xb, y, X, V1, K, k, jacDims, verbose=True)
    print 'r:', r

    approx_JXg, approx_Js, approx_JV1 = approx_jacs(
        lambda *args, **kwargs: EvaluateSectionedBasisArapEnergy(*args, **kwargs)[0],
        [2, 3, 7],
        1e-6,
        T, V, Xg, s, Xb, y, X, V1, K, k, empty_jacDims, verbose=False)

    print_comparison(approx_JXg=approx_JXg, JXg=JXg)
    print_comparison(approx_Js=approx_Js, Js=Js)
    print_comparison(approx_V1i=approx_JV1[:, 12:15], JV1i=JV1i)
    print_comparison(approx_V1j=approx_JV1[:, 9:12], JV1j=JV1j)

    # Test K[4,0] == -1 (free)
    K[4,0] = -1
    K[4,1] = 0

    X = np.zeros((1, 3), dtype=np.float64)
    X = randomise(X)

    jacDims = np.r_[3, 3, 
                    3, 3,
                    3, 3,
                    3, 3,
                    3, 1].reshape(-1, 2).astype(np.int32)

    r, JXg, JV1i, JV1j, JX, Js = EvaluateSectionedBasisArapEnergy(T, V, Xg, s, Xb, y, X, V1, K, k, jacDims, verbose=True)

    print 'r:', r

    approx_JXg, approx_Js, approx_JX, approx_JV1 = approx_jacs(
        lambda *args, **kwargs: EvaluateSectionedBasisArapEnergy(*args, **kwargs)[0],
        [2, 3, 6, 7],
        1e-6,
        T, V, Xg, s, Xb, y, X, V1, K, k, empty_jacDims, verbose=False)

    print_comparison(approx_JXg=approx_JXg, JXg=JXg)
    print_comparison(approx_Js=approx_Js, Js=Js)
    print_comparison(approx_JX=approx_JX, JX=JX)
    print_comparison(approx_V1i=approx_JV1[:, 12:15], JV1i=JV1i)
    print_comparison(approx_V1j=approx_JV1[:, 9:12], JV1j=JV1j)

    # Test K[4,0] == 1 (basis)
    K[4,0] = 1
    K[4,1] = 0

    Xb = np.zeros((1, 3), dtype=np.float64)
    y = np.ones((1, 1), dtype=np.float64)

    Xb = randomise(Xb)
    y = randomise(y)

    jacDims = np.r_[3, 3, 
                    3, 3,
                    3, 3,
                    3, 3,
                    3, 1,
                    3, 1].reshape(-1, 2).astype(np.int32)

    r, JXg, JV1i, VJ1j, JXb, Jy, Js = EvaluateSectionedBasisArapEnergy(T, V, Xg, s, Xb, y, X, V1, K, k, jacDims, verbose=True)

    print 'r:', r

    approx_JXg, approx_Js, approx_JXb, approx_Jy, approx_JV1 = approx_jacs(
        lambda *args, **kwargs: EvaluateSectionedBasisArapEnergy(*args, **kwargs)[0],
        [2, 3, 4, 5, 7],
        1e-6,
        T, V, Xg, s, Xb, y, X, V1, K, k, empty_jacDims, verbose=False)

    print_comparison(approx_JXg=approx_JXg, JXg=JXg)
    print_comparison(approx_Js=approx_Js, Js=Js)
    print_comparison(approx_JXb=approx_JXb, JXb=JXb)
    print_comparison(approx_Jy=approx_Jy, Jy=Jy)
    print_comparison(approx_V1i=approx_JV1[:, 12:15], JV1i=JV1i)
    print_comparison(approx_V1j=approx_JV1[:, 9:12], JV1j=JV1j)

if __name__ == '__main__':
    # main_EvaluateSingleARAP()
    # main_EvaluateSingleARAP2()
    # main_EvaluateDualARAP()
    # main_EvaluateDualNonLinearBasisARAP()
    main_EvaluateSectionedBasisArapEnergy()

