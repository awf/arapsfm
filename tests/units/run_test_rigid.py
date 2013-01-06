# run_test_rigid.py

# Imports
import numpy as np
from pprint import pprint
from misc.scipy_ import approx_jac
from operator import itemgetter
from test_rigid import EvaluateRigidRegistrationEnergy

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

# main_EvaluateRigidRegistrationEnergy
def main_EvaluateRigidRegistrationEnergy():
    V0 = np.array([[0., 0., 0.],
                   [1., 0., 0.],
                   [2., 0., 0.]], dtype=np.float64)

    V = np.array([[0., 0., 0.],
                  [0., 1., 0.],
                  [0., 2., 0.]], dtype=np.float64)

    s = np.array([[1.]], dtype=np.float64)
    xg = np.array([[0., 0., 0.]], dtype=np.float64)
    d = np.array([[5., 0., 0.]], dtype=np.float64)
    k = 1
    w = 1.0

    # evaluate residual and analytical Jacobians
    r, Js, Jxg, Jd, JV = EvaluateRigidRegistrationEnergy(V0, V, s, xg, d, w, k)
    
    # evaluate the approximate Jacobians
    approx_Js, approx_Jxg, approx_Jd, approx_JV = approx_jacs(
        lambda *a: EvaluateRigidRegistrationEnergy(*a)[0],
        [2, 3, 4, 1],
        1e-6,
        V0, V, s, xg, d, w, k)

    print 'r:', r
    print_comparison(approx_Js=approx_Js, Js=Js)
    print_comparison(approx_Jxg=approx_Jxg, Jxg=Jxg)
    print_comparison(approx_Jd=approx_Jd, Jd=Jd)
    print_comparison(approx_JV=approx_JV[:,3*k:3*(k+1)], JV=JV)

if __name__ == '__main__':
    main_EvaluateRigidRegistrationEnergy()
