# run_test_linear_deformation.py

# Imports
import numpy as np
from test_linear_deformation import *
from pprint import pprint
from misc.scipy_ import approx_jac
from operator import itemgetter

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

# expaneded_EvaluateLinearDeformationEnergy
def expaneded_EvaluateLinearDeformationEnergy(*args, **kwargs):
    a = args[:3]
    args = args[3:]

    n = args[0]
    args = args[1:]
    Xgb = [args[i] for i in xrange(n)]
    args = args[n:]
    yg = [args[i] for i in xrange(n)]
    args = args[n:]

    return EvaluateLinearDeformationEnergy(*(a + (n, Xgb, yg) + args), **kwargs)

# main_EvaluateLinearDeformationEnergy
def main_EvaluateLinearDeformationEnergy():
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
    V -= np.mean(V, axis=0)
    print 'V:'
    print V

    s = np.r_[2.0].reshape(1, 1)
    w = 1.0
    k_ = 4 # i == 4

    # using single global rotation
    n = -1
    Xgb = []
    yg = []
    Xg = np.zeros((1, 3), dtype=np.float64)

    dg = np.r_[0., 0., 0.].reshape((1, 3))

    V1 = V.copy()
    V1[4, 0] += 1.0

    # XXX
    # V1 = randomise(V1)
    # V = randomise(V)
    # s = randomise(s)
    # Xg = randomise(Xg)
    # dg = randomise(dg)
    # XXX

    empty_jacDims = np.empty((0, 0), dtype=np.int32)

    jacDims = np.r_[3, 3,
                    3, 1,
                    3, 3,
                    3, 3,
                    3, 3].reshape(-1, 2).astype(np.int32)

    r, JVi, Js, Jxg, Jdg, JV1i = EvaluateLinearDeformationEnergy(
        T, V, s, n, Xgb, yg, Xg, dg, V1, w, k_, jacDims)

    print 'r:', r

    approx_JV, approx_Js, approx_Jxg, approx_Jdg, approx_JV1 = approx_jacs(
        lambda *a: EvaluateLinearDeformationEnergy(*a)[0],
        [1, 2, 6, 7, 8],
        1e-6,
        T, V, s, n, Xgb, yg, Xg, dg, V1, w, k_, empty_jacDims)

    # XXX
    # print_comparison(approx_JVi=approx_JV[:, 12:15], JVi=JVi)
    # print_comparison(approx_Js=approx_Js, Js=Js)
    # print_comparison(approx_Jxg=approx_Jxg, Jxg=Jxg)
    # print_comparison(approx_Jdg=approx_Jdg, Jdg=Jdg)
    # print_comparison(approx_JV1i=approx_JV1[:, 12:15], JV1i=JV1i)
    # XXX

    # using 2-basis for global rotation
    n = 2
    Xgb = [np.r_[0., 0., 1.].reshape(1, 3),
           np.r_[0., 1., 0.].reshape(1, 3)]
    yg = [np.r_[1.0].reshape(1,1),
          np.r_[0.5].reshape(1,1)]

    jacDims = np.r_[3, 3,
                    3, 1,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 1,
                    3, 3,
                    3, 3].reshape(-1, 2).astype(np.int32)

    r, JVi, Js, Jxgb0, Jyg0, Jxgb1, Jyg1, Jdg, JV1i = EvaluateLinearDeformationEnergy(
        T, V, s, n, Xgb, yg, Xg, dg, V1, w, k_, jacDims)

    (approx_JV, approx_Js, approx_Jxgb0, approx_Jyg0, 
     approx_Jxgb1, approx_Jyg1, approx_Jdg, approx_JV1) = approx_jacs(
        lambda *a: expaneded_EvaluateLinearDeformationEnergy(*a)[0],
        [1, 2, 4, 6, 5, 7, 9, 10],
        1e-6,
        T, V, s, n, Xgb[0], Xgb[1], yg[0], yg[1], Xg, dg, V1, w, k_, empty_jacDims)

    print_comparison(approx_JVi=approx_JV[:, 12:15], JVi=JVi)
    print_comparison(approx_Js=approx_Js, Js=Js)
    print_comparison(approx_Jxgb0=approx_Jxgb0, Jxgb0=Jxgb0)
    print_comparison(approx_Jxgb1=approx_Jxgb1, Jxgb1=Jxgb1)
    print_comparison(approx_Jyg0=approx_Jyg0, Jyg0=Jyg0)
    print_comparison(approx_Jyg1=approx_Jyg1, Jyg1=Jyg1)
    print_comparison(approx_Jdg=approx_Jdg, Jdg=Jdg)
    print_comparison(approx_JV1i=approx_JV1[:, 12:15], JV1i=JV1i)


    return







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
    main_EvaluateLinearDeformationEnergy()

