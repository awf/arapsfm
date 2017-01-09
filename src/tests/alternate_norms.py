# alternate_norms.py

# Imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

# join_quad_to_poly_C1
def join_quad_to_poly_C1(tau, p):
    c = tau*tau - (2.0 * tau / p)**(p / (p - 1))
    b = tau - (tau*tau - c)**(1.0 / p)

    return 1.0, b, c

# join_quad_to_poly_C2
def join_quad_to_poly_C2(tau, p):
    b = tau - (p - 1)*tau
    a = 2.0 / ((p*(p-1)) * (tau - b)**(p - 2))
    c = tau*tau - a*(tau - b)**p

    return a, b, c

# view_quad_to_poly_C1
def view_quad_to_poly_C1():
    tau = 8.0
    p = 4.0

    b, c = join_quad_to_poly_C1(tau, p)

    def f(x):
        y = np.empty_like(x)
        x_ = np.abs(x)
        i = x_ < tau
        j = ~i
        y[i] = x[i]*x[i]
        y[j] = (x_[j] - b)**p + c

        return y

    x = np.linspace(-12, 12, 100, endpoint=False)

    plt.plot(x, f(x), 'r.-')
    plt.show()

# residual_transform
def residual_transform(tau, p):
    a, b, c = join_quad_to_poly_C1(tau, p)

    def f(x):
        x = np.atleast_1d(x)

        y = np.empty_like(x)
        sgn_x = np.sign(x) 
        abs_x = np.abs(x)

        i = abs_x < tau
        y[i] = x[i]

        j = ~i
        y[j] = sgn_x[j] * np.sqrt(a* np.power(abs_x[j] - b, p) + c)

        return np.squeeze(y)

    return f

# test_residual_transform
def test_residual_transform():
    f = residual_transform(8.0, 4)

    x = np.linspace(-12, 12, 101, endpoint=True)
    f_x = f(x)

    def fprime(f, x, epsilon=1e-3):
        fk = f(x)
        fkp1 = f(x + epsilon)
        return (fkp1 - fk) / epsilon
        
    fprime_x = fprime(f, x)

    fig, axs = plt.subplots(1,2)
    axs[0].plot(x, f_x, 'r.-')
    axs[0].plot(x, x, 'b:')
    axs[1].plot(x, fprime_x, 'r.-')
    plt.show()

if __name__ == '__main__':
    # view_quad_to_poly()
    test_residual_transform()

