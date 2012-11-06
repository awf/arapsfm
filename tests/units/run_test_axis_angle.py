# run_test_axis_angle.py

# Imports
import numpy as np
from geometry.quaternion import *
from geometry.axis_angle import *
from misc.scipy_ import approx_jac
from scipy.linalg import norm

# main
def main():
    v = np.r_[np.deg2rad(10.), 0., 0.]
    w = np.r_[np.deg2rad(30.), 0., 0.]

    z = axMakeInterpolated(0.3, v, 0.6, w)
    print 'v:', v
    print 'w:', w
    print 'z:', z
    print 'z[0] (deg):', np.rad2deg(z[0])

    print 'axScale(0.3, v): ', axScale(0.3, v)
    print 'axScale(0.6, w): ', axScale(0.6, w)
    print 'axAdd(axScale(0.3, v), axScale(0.6, w)): ', 
    print  axAdd(axScale(0.3, v), axScale(0.6, w))

    approx_Dv = approx_jac(lambda x: axAdd(x, w), v, epsilon=1e-3)
    print 'approx_Dv:'
    print np.around(approx_Dv, decimals=3)

    Dv = axAdd_da(v, w)
    print 'Dv:'
    print np.around(Dv, decimals=3)
    print 'allclose? ', np.allclose(Dv, approx_Dv, atol=1e-4)

    approx_Dw = approx_jac(lambda x: axAdd(v, x), w, epsilon=1e-6)
    print 'approx_Dw:'
    print np.around(approx_Dw, decimals=3)

    Dw = axAdd_db(v, w)
    print 'Dw:'
    print np.around(Dw, decimals=3)
    print 'allclose? ', np.allclose(Dw, approx_Dw, atol=1e-4)

if __name__ == '__main__':
    main()

