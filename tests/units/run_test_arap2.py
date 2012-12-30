# run_test_arap2.py

# Imports
import numpy as np
from test_arap2 import *
from pprint import pprint
from misc.scipy_ import approx_jac
from operator import itemgetter

# unlist_fn
def unlist_fn(f, *args):
    arg_info = []
    for i, arg in enumerate(args):
        if isinstance(arg, list):
            arg_info.append(len(arg))
        else:
            arg_info.append(-1)
        
    def unlisted_function(*all_args):
        iter_args = iter(all_args)
        listed_args = []
        for i, n in enumerate(arg_info):
            if n < 0:
                listed_args.append(next(iter_args))
            else:
                listed_args.append([next(iter_args) for j in xrange(n)])

        f(*listed_args)

    return unlisted_function

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

# main_EvaluateCompleteSectionedBasisArapEnergy
def main_EvaluateCompleteSectionedBasisArapEnergy():
    randomise = lambda arr: np.random.randn(arr.size).reshape(arr.shape)

    T = np.r_[1, 2, 4,
              2, 3, 4,
              3, 0, 4,
              0, 1, 4].reshape(-1, 3).astype(np.int32)

    V = np.r_[0., 0., 0.,
              0., 2., 0.,
              2., 2., 0.,
              2., 0., 0.,
              1., 1., 1.].reshape(-1, 3).astype(np.float64)
    
    s = np.ones((1, 1), dtype=np.float64)

    k_ = 8 # (i, j) = (4, 3)

    # using single rotation
    n = -1
    Xgb = []
    yg = []
    Xg = np.zeros((1, 3), dtype=np.float64)

    # fixed vertices (only rigid motion possible locally)
    k = np.r_[0, 0, 0, 0, 0].astype(np.int32)
    Xb = np.zeros((0, 3), dtype=np.float64)
    y = np.zeros((0, 3), dtype=np.float64)
    X = np.zeros((0, 3), dtype=np.float64)

    V1 = V.copy()
    V1[3, 0] += 1.0

    # V1 = randomise(V1)
    # V = randomise(V)
    # s = randomise(s)
    # Xg = randomise(Xg)

    empty_jacDims = np.empty((0, 0), dtype=np.int32)
    jacDims = np.r_[3, 3,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 3,
                    3, 3].reshape(-1, 2).astype(np.int32)

    r, JVi, JVj, Js, JXg, JV1i, JV1j = EvaluateCompleteSectionedBasisArapEnergy(T, V, s, n, Xgb, yg, Xg, k, Xb, y, X, V1, k_, jacDims)

    print 'r:', r

    approx_JV, approx_Js, approx_JXg, approx_JV1 = approx_jacs(
        lambda *a, **k: EvaluateCompleteSectionedBasisArapEnergy(*a, **k)[0],
        [1, 2, 6, 11],
        1e-6,
        T, V, s, n, Xgb, yg, Xg, k, Xb, y, X, V1, k_, empty_jacDims)

    print_comparison(approx_Vi=approx_JV[:, 12:15], JVi=JVi)
    print_comparison(approx_Vj=approx_JV[:, 9:12], JVj=JVj)
    print_comparison(approx_Js=approx_Js, Js=Js)
    print_comparison(approx_JXg=approx_JXg, JXg=JXg)
    print_comparison(approx_V1i=approx_JV1[:, 12:15], JV1i=JV1i)
    print_comparison(approx_V1j=approx_JV1[:, 9:12], JV1j=JV1j)

    # using 1-basis for global rotation
    n = 1 
    Xgb = [0.5 * np.ones((1, 3), dtype=np.float64)]
    yg = [np.ones((1, 1), dtype=np.float64)]

    jacDims = np.r_[3, 3,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 3].reshape(-1, 2).astype(np.int32)

    r, JVi, JVj, Js, JXgb, Jyg, JV1i, JV1j = EvaluateCompleteSectionedBasisArapEnergy(T, V, s, n, Xgb, yg, Xg, k, Xb, y, X, V1, k_, jacDims)

    approx_JV, approx_Js, approx_JXgb, approx_Jyg, approx_JV1 = approx_jacs(
        lambda *a: EvaluateCompleteSectionedBasisArapEnergy(*(a[:4] + ([a[4]], [a[5]]) + a[6:]))[0],
        [1, 2, 4, 5, 11],
        1e-6,
        T, V, s, n, Xgb[0], yg[0], Xg, k, Xb, y, X, V1, k_, empty_jacDims)

    print_comparison(approx_JXgb=approx_JXgb, JXgb=JXgb)
    print_comparison(approx_Jyg=approx_Jyg, Jyg=Jyg)

    # use independent rotation at i == 4
    k = np.r_[0, 0, 0, 0, -1, 0].astype(np.int32)
    X = np.ones((1, 3), dtype=np.float64)

    jacDims = np.r_[3, 3,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 3,
                    3, 3].reshape(-1, 2).astype(np.int32)

    r, JVi, JVj, Js, JXgb, Jyg, JXi, JV1i, JV1j = EvaluateCompleteSectionedBasisArapEnergy(T, V, s, n, Xgb, yg, Xg, k, Xb, y, X, V1, k_, jacDims, False)

    approx_JV, approx_Js, approx_JXgb, approx_Jyg, approx_JXi, approx_JV1 = approx_jacs(
        lambda *a: EvaluateCompleteSectionedBasisArapEnergy(*(a[:4] + ([a[4]], [a[5]]) + a[6:]))[0],
        [1, 2, 4, 5, 10, 11],
        1e-6,
        T, V, s, n, Xgb[0], yg[0], Xg, k, Xb, y, X, V1, k_, empty_jacDims)

    print_comparison(approx_JXi=approx_JXi, JXi=JXi)
    print_comparison(approx_V1i=approx_JV1[:, 12:15], JV1i=JV1i)
    print_comparison(approx_V1j=approx_JV1[:, 9:12], JV1j=JV1j)

    # use basis rotation at i == 4
    k = np.r_[0, 0, 0, 0, 1, 0, 0].astype(np.int32)
    Xb = np.ones((1, 3), dtype=np.float64)
    y = np.ones((1, 1), dtype=np.float64)

    jacDims = np.r_[3, 3,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 3].reshape(-1, 2).astype(np.int32)

    r, JVi, JVj, Js, JXgb, Jyg, JXb, Jy, JV1i, JV1j = EvaluateCompleteSectionedBasisArapEnergy(T, V, s, n, Xgb, yg, Xg, k, Xb, y, X, V1, k_, jacDims, False)
    print 'r:', r

    approx_JV, approx_Js, approx_JXgb, approx_Jyg, approx_JXb, approx_Jy, approx_JV1 = approx_jacs(
        lambda *a: EvaluateCompleteSectionedBasisArapEnergy(*(a[:4] + ([a[4]], [a[5]]) + a[6:]))[0],
        [1, 2, 4, 5, 8, 9, 11],
        1e-6,
        T, V, s, n, Xgb[0], yg[0], Xg, k, Xb, y, X, V1, k_, empty_jacDims)

    print_comparison(approx_JXgb=approx_JXgb, JXgb=JXgb)
    print_comparison(approx_Jyg=approx_Jyg, Jyg=Jyg)
    print_comparison(approx_JXb=approx_JXb, JXb=JXb)
    print_comparison(approx_Jy=approx_Jy, Jy=Jy)
    print_comparison(approx_V1i=approx_JV1[:, 12:15], JV1i=JV1i)
    print_comparison(approx_V1j=approx_JV1[:, 9:12], JV1j=JV1j)

if __name__ == '__main__':
    main_EvaluateCompleteSectionedBasisArapEnergy()

