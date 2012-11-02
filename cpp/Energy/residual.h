#ifndef __RESIDUAL_H__
#define __RESIDUAL_H__

#include <cmath>

// ResidualTransform
struct ResidualTransform
{
    virtual double Transform(const double r) const = 0;
    virtual double Derivative(const double r) const = 0;
};

// LinearTransform
struct LinearTransform : public ResidualTransform
{
    virtual double Evaluate(const double r) const { return r; }
    virtual double Derivative(const double r) const { return 1.0; }
};

// PiecewisePolynomialTransform_C1
struct PiecewisePolynomialTransform_C1 : public ResidualTransform
{
    PiecewisePolynomialTransform_C1(const double tau, const double p)
        : _tau(tau), _p(p)
    {
        _c = tau*tau - std::pow(2.0 * tau / p, p / (p - 1.0));
        _b = tau - std::pow(tau*tau - _c, 1.0 / p);
    }
        
    virtual double Transform(const double r) const
    {
        const double abs_r = std::abs(r);
        
        if (abs_r <= _tau)
            return r;
        else
            return std::copysign(std::sqrt(std::pow(abs_r - _b, _p) + _c), r);
    }

    virtual double Derivative(const double r) const
    {
        const double abs_r = std::abs(r);

        if (abs_r <= _tau)
            return 1.0;
        else
            return (_p * std::pow(abs_r - _b, _p - 1.0)) / 
                   (2.0 * std::sqrt(std::pow(abs_r - _b, _p) + _c));
    }

protected:
    double _tau, _p, _b, _c;
};

#endif
