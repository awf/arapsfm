# run_test_rotation_composition.py

# Imports
import numpy as np
from pprint import pprint
from misc.scipy_ import approx_jac
from operator import itemgetter
from test_rotation_composition import EvaluateRotationComposition

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

# randomise_inplace
def randomise_inplace(a, low=0., high=1.):
    a.flat = np.random.uniform(low, high, a.size)

# expanded_EvaluateRotationComposition
def expanded_EvaluateRotationComposition(*args, **kwargs):
    k = args[-1]
    n = len(args) - 1
    Xb = list(args[:n/2])
    y = list(args[n/2:-1])

    return EvaluateRotationComposition(Xb, y, k, **kwargs)

# main_EvaluateRotationComposition
def main_EvaluateRotationComposition():
    Xb = [np.array([[1., 0., 0.]], dtype=np.float64),
          np.array([[1., 0., 0.]], dtype=np.float64)]
    y = [np.array([[2.]], dtype=np.float64),
         np.array([[4.]], dtype=np.float64)]
    k = 0

    map(randomise_inplace, Xb)
    map(randomise_inplace, y)

    t = EvaluateRotationComposition(Xb, y, k, debug=False)
    r, JXb, Jy = t[0], t[1::2], t[2::2]

    approx_J = approx_jacs(
        lambda *a: expanded_EvaluateRotationComposition(*a, debug=False)[0],
        range(len(Xb) + len(y)),
        1e-6,
        *(tuple(Xb) + tuple(y) + (k,)))

    approx_JXb = approx_J[:len(Xb)]
    approx_Jy = approx_J[len(Xb):]

    for i in xrange(len(approx_JXb)):
        kw = {'atol' : 1e-3}
        kw['approx_JXb[%d]' % i] = approx_JXb[i]
        kw['JXb[%d]' % i] = JXb[i]
        print_comparison(**kw)

        kw = {'atol' : 1e-3}
        kw['approx_Jy[%d]' % i] = approx_Jy[i]
        kw['Jy[%d]' % i] = Jy[i]
        print_comparison(**kw)

if __name__ == '__main__':
    main_EvaluateRotationComposition()
