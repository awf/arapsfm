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

if __name__ == '__main__':
    main()

