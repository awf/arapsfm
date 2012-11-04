# run_test_quaternion.py

# Imports
import numpy as np
# from test_quaternion import *
from geometry.quaternion import *
from misc.scipy_ import approx_jac
from scipy.linalg import norm

# main_quatMultiply
def main_quatMultiply():
    # p = quat(np.r_[0.0, 0.0, np.deg2rad(15)])
    # q = quat(np.r_[0.0, 0.0, np.deg2rad(30)])
    p = quat(np.random.randn(3))
    q = quat(np.random.randn(3))
    print 'p:', np.around(p, decimals=3)
    print 'q:', np.around(q, decimals=3)

    # Test quatMultiply
    Rp = rotationMatrix(p)
    Rq = rotationMatrix(q)
    r = quatMultiply(p, q)
    R = rotationMatrix(r)
    S = np.dot(Rp, Rq) 
    print 'R:'
    print np.around(R, decimals=3)
    print 'S:'
    print np.around(S, decimals=3)
    print 'allclose? ', np.allclose(R, S, atol=1e-4)

    Dp, Dq = quatMultiply_dp_dq(p, q)

    print 'Dp:'
    print Dp

    approx_Dp = approx_jac(lambda p: quatMultiply(p, q), p)
    print 'approx_Dp:'
    print approx_Dp
    print 'allclose? ', np.allclose(Dp, approx_Dp, atol=1e-4)

    print 'Dq:'
    print Dq

    approx_Dq = approx_jac(lambda q: quatMultiply(p, q), q)
    print 'approx_Dq:'
    print approx_Dq
    print 'allclose? ', np.allclose(Dq, approx_Dq, atol=1e-4)

# py_quatInv_Jac
def py_quatInv_Jac(q):
    m = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
    dmdq = q[:3] / m
    # f = lambda q: np.r_[np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])]
    # approx_dmdq = approx_jac(f, q[:3])
    # print dmdq
    # print approx_dmdq

    t = 2.0 * np.arctan2(m, q[3])
    dtdm = q[3] / (m*m + q[3]*q[3])
    dtdq3 = -m / (m*m + q[3]*q[3])
    dtdq = 2.0 * np.r_[dtdm * dmdq, dtdq3]
    # def f(q):
    #     m = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
    #     return np.r_[np.arctan2(m, q[3])]
    # approx_dtdq = approx_jac(f, q)
    # print dtdq
    # print approx_dtdq

    a = np.sin(0.5 * t) / t
    dadt = (0.5 * t * np.cos(0.5 * t) - np.sin(0.5 * t)) / (t * t)
    dadq = dadt * dtdq

    # def f(q):
    #     m = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
    #     t = np.r_[np.arctan2(m, q[3])]
    #     return np.sin(0.5 * t) / t

    # approx_dadq = approx_jac(f, q)
    # print dadq
    # print approx_dadq
    
    dxdq = - q[:3, np.newaxis] * dadq / (a * a)
    dxdq[np.diag_indices(3)] += 1.0 / a

    return dxdq

# main_quatInv
def main_quatInv():
    # x = np.r_[0, 0, 0].astype(np.float64)
    # x = np.random.randn(3).astype(np.float64)
    # x /= np.sqrt(np.sum(x*x))
    x = np.r_[1., 0.1, 0.]
    # x *= np.deg2rad(np.random.rand()*180)
    x *= (1.0*np.pi) / norm(x)
    x = np.random.randn(3)
    q = quat(x)

    print 'x:'
    print np.around(x, decimals=3)
    print '|x| (deg): ', np.sign(x[0]) * np.rad2deg(norm(x))

    print 'q:'
    print np.around(q, decimals=3)

    y = quatInv(q)
    print 'y:'
    print np.around(y, decimals=3)
    print '|y| (deg): ', np.sign(y[0]) * np.rad2deg(norm(y))

    J = py_quatInv_Jac(q)
    print np.around(J, decimals=3)

    approx_J = approx_jac(quatInv, q)
    print np.around(approx_J, decimals=3)

    J = quatInvDq(q)
    print np.around(J, decimals=3)

if __name__ == '__main__':
    main_quatInv()

