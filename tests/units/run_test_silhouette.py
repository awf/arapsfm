# run_test_silhouette.py

# Imports
import numpy as np
from test_silhouette import *
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


# main_EvaluateSilhouetteNormal2
def main_EvaluateSilhouetteNormal2():
    randomise = lambda arr: np.random.randn(arr.size).reshape(arr.shape)

    T = np.r_[0, 1, 2,
              0, 2, 3,
              0, 3, 4,
              0, 4, 1].reshape(-1,3).astype(np.int32)

    V = np.r_[0., 0., 0.,
              -1, 1., 0.,
              1., 1., 0.,
              1., -1., 0.,
              -1., -1., 0.].reshape(-1,3).astype(np.float64)

    l = 0
    u = np.r_[0., 0.]
    sn = np.r_[0., 0.5, 0.]
    w = 1.0

    # u = randomise(u)
    V = randomise(V)

    print EvaluateSilhouetteNormal2(T, V, 
                                    l, u,
                                    sn,
                                    w,
                                    debug=False)

    vertex_id = 0

    JV0 =  EvaluateSilhouette2Jac_V1(T, V, 
                                     l, u,
                                     sn,
                                     vertex_id,
                                     w,
                                     debug=False)

    def fn(v0):
        V1 = V.copy()
        V1[vertex_id] = v0
        return EvaluateSilhouetteNormal2(T, V1, l, u, sn, w, debug=False)

    approx_JV0, = approx_jacs(fn, [0], 1e-6, V[vertex_id])
    print_comparison(approx_JV0=approx_JV0, JV0=JV0)

    Ju = EvaluateSilhouette2Jac_u(T, V, l, u, sn, w, debug=False)
    print Ju

    def fn(u0):
        return EvaluateSilhouetteNormal2(T, V, l, u0, sn, w, debug=False)

    approx_Ju, = approx_jacs(fn, [0], 1e-6, u)
    print_comparison(approx_Ju=approx_Ju, Ju=Ju)

if __name__ == '__main__':
    main_EvaluateSilhouetteNormal2()

