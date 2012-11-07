# register.py

# Imports
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import fmin_bfgs, leastsq
from quaternion import *

# right_multiply_affine_transform
def right_multiply_affine_transform(V0, V):
    t0 = np.mean(V0, axis=0)
    t = np.mean(V, axis=0)

    V0 = V0 - t0
    A = block_diag(V0, V0, V0)
    A_T = np.transpose(A)
    A_pinv = np.dot(np.linalg.inv(np.dot(A_T, A)), A_T)
    x = np.dot(A_pinv, np.ravel(np.transpose(V - t)))

    T = np.transpose(x.reshape(3, 3))
    d = t - np.dot(t0, T)

    return T, d

# right_multiply_rigid_transform
def right_multiply_rigid_transform(V0, V, verbose=False):
    x = np.r_[0., 0., 0., 1., 0., 0., 0.]

    def f(x, verbose=False):
        ax, s, d = x[:3], x[3], x[4:]
        A = s * rotationMatrix(quat(ax))
        r = V - (np.dot(V0, A) + d)
        e = np.sum(r*r)
        if verbose:
            print '%.3f: ' % e, np.around(x, decimals=4)
        return e

    def callback(x):
        f(x, verbose=True)

    if verbose:
        callback = lambda x: f(x, True)
    else:
        callback = None

    x = fmin_bfgs(f, x, callback=callback, disp=verbose)
    x, s, d = x[:3], x[3], x[4:]

    return (s * rotationMatrix(quat(x))), d

# right_multiply_rigid_uniform_scale_transform
def right_multiply_rigid_uniform_scale_transform(V0, V, verbose=False):
    x = np.r_[0., 0., 0., 0., 0., 0.]

    def f(x, verbose=False):
        ax, d = x[:3], x[3:]
        A = rotationMatrix(quat(ax))
        r = V - (np.dot(V0, A) + d)
        e = np.sum(r*r)
        if verbose:
            print '%.3f: ' % e, np.around(x, decimals=4)
        return e

    def callback(x):
        f(x, verbose=True)

    if verbose:
        callback = lambda x: f(x, True)
    else:
        callback = None

    x = fmin_bfgs(f, x, callback=callback, disp=verbose)
    x, d = x[:3], x[3:]

    return rotationMatrix(quat(x)), d

