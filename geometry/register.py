# register.py

# Imports
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import leastsq
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
def right_multiply_rigid_transform(V0, V):
    x = np.r_[0., 0., 0., 1., 0., 0., 0.]

    def func(x):
        ax, s, d = x[:3], x[3], x[4:]
        A = s * rotationMatrix(quat(ax))
        r = V - (np.dot(V0, A) + d)
        return np.ravel(r)

    x = leastsq(func, x, xtol=1e-6, ftol=1e-6, gtol=1e-6)[0]
    x, s, d = x[:3], x[3], x[4:]

    return (s * rotationMatrix(quat(x))), d

# right_multiply_rigid_uniform_scale_transform
def right_multiply_rigid_uniform_scale_transform(V0, V, verbose=False):
    x = np.r_[0., 0., 0., 0., 0., 0.]

    def func(x):
        ax, d = x[:3], x[3:]
        A = rotationMatrix(quat(ax))
        r = V - (np.dot(V0, A) + d)
        return np.ravel(r)

    x = leastsq(func, x, xtol=1e-6, ftol=1e-6, gtol=1e-6)[0]
    x, d = x[:3], x[3:]

    return rotationMatrix(quat(x)), d

