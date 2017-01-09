# run_test_linear_basis_shape.py

# Imports
import numpy as np
from test_linear_basis_shape import *
from misc.scipy_ import approx_jac
from pprint import pprint
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

# main_EvaluateLinearBasisShape
def main_EvaluateLinearBasisShape():
    xg = np.r_[np.rad2deg(10), np.rad2deg(-15), np.deg2rad(70)].reshape(-1, 3)
    s = np.r_[2.].reshape(-1, 1)
    vd = np.r_[0., 0., 0.].reshape(-1, 3)
    Vb = [np.r_[1., 0., 0.].reshape(-1, 3),
          np.r_[0., 0.5, 0.].reshape(-1, 3)]
    y = np.r_[3.]

    k = 0

    r, JV0, JV1, Js, JXg, JVd, Jy = EvaluateLinearBasisShape(Vb, y, s, xg, vd,
                                                             k, debug=False)
    print 'r:', r

    def fn(V0, V1, s, Xg, Vd, y):
        r = EvaluateLinearBasisShape([V0, V1], y, s, Xg, Vd, k)[0]
        return r.ravel()

    aJV0, aJV1, aJs, aJXg, aJVd, aJy = approx_jacs(
        fn, 
        [0, 1, 2, 3, 4, 5], 
        1e-6, 
        Vb[0], Vb[1], s, xg, vd, y)

    print_comparison(aJV0=aJV0, JV0=JV0)
    print_comparison(aJV1=aJV1, JV1=JV1)
    print_comparison(aJs=aJs, Js=Js)
    print_comparison(aJXg=aJXg, JXg=JXg)
    print_comparison(aJVd=aJVd, JVd=JVd)
    print_comparison(aJy=aJy, Jy=Jy)

if __name__ == '__main__':
    main_EvaluateLinearBasisShape()

