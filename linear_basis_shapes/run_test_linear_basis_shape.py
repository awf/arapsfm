# run_test_linear_basis_shape.py

# Imports
import numpy as np
from test_linear_basis_shape import *

# main_EvaluateLinearBasisShape
def main_EvaluateLinearBasisShape():
    xg = np.r_[0., 0., np.deg2rad(45)].reshape(-1, 3)
    s = np.r_[2.].reshape(-1, 1)
    vd = np.r_[0., 0., 1.].reshape(-1, 3)
    Vb = [np.r_[1., 0., 0.].reshape(-1, 3),
          np.r_[0., 0.5, 0.].reshape(-1, 3)]
    y = np.r_[2.]

    V = EvaluateLinearBasisShape(Vb, y, s, xg, vd)
    print V

if __name__ == '__main__':
    main_EvaluateLinearBasisShape()

