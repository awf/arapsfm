# run_test_quaternion.py

# Imports
import numpy as np
from test_quaternion import *
from misc.scipy_ import approx_jac

def main():
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

if __name__ == '__main__':
    main()
