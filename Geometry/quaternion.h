/* quaternion.h */
#ifndef __QUATERNION_H__
#define __QUATERNION_H__

#include <cmath>
#include <cassert>

// Functions for manipulating and evaluating quaternions from a vector of 3
// elements (axis-angle representation)

// General

// sinc = sin(t) / t 
// half_sinc_half_t = sin(0.5) / t

// half_sinc_half_t
template <typename Elem>
inline
Elem half_sinc_half_t(Elem t)
{
    assert(t >= 0);

    if (t < 1e-4)
        return 0.5;

    return sin(0.5*t) / t;
}

// half_sinc_half_t_p is the derivative w.r.t t of half_sinc_half_t

// half_sinc_half_t_p
template <typename Elem>
inline
Elem half_sinc_half_t_p(Elem t)
{
    assert(t >= 0);

    if (t < 1e-4)
        return 0;

    return 0.5*cos(0.5*t)/t - sin(0.5*t)/(t*t);
}

// half_sinc_half_t_p_div_t
template <typename Elem>
inline
Elem half_sinc_half_t_p_div_t(Elem t)
{
    assert(t >= 0);

    if (t < 1e-3)
        return -1./(24.);

    return half_sinc_half_t_p(t) / t;
}

// Evaluation functions (UNSAFE)

// quat_Unsafe
template <typename Elem>
inline
void quat_Unsafe(const Elem * x, Elem * q)
{
    Elem t = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    Elem a = half_sinc_half_t(t);

    q[0] = a*x[0];
    q[1] = a*x[1];
    q[2] = a*x[2];
    q[3] = cos(0.5*t);
}

// quatDx_Unsafe
template <typename Elem>
inline
void quatDx_Unsafe(const Elem * x, Elem * D)
{
    Elem t = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    Elem f = half_sinc_half_t(t);
    Elem g = half_sinc_half_t_p_div_t(t);

    D[0] = f + g*x[0]*x[0];
    D[1] = g*x[0]*x[1];
    D[2] = g*x[0]*x[2];

    D[3] = D[1];            // g*x[1]*x[0];
    D[4] = f + g*x[1]*x[1];
    D[5] = g*x[1]*x[2];

    D[6] = D[2];            // g*x[2]*x[0];
    D[7] = D[5];            // g*x[2]*x[1];
    D[8] = f + g*x[2]*x[2];

    D[9] = -0.5*f*x[0];
    D[10] = -0.5*f*x[1];
    D[11] = -0.5*f*x[2];
}

// rotationMatrix_Unsafe
template <typename Elem>
inline
void rotationMatrix_Unsafe(const Elem * q, Elem * R)
{
    R[0] = 1 - 2*q[1]*q[1] - 2*q[2]*q[2];
    R[1] = 2*q[0]*q[1] - 2*q[2]*q[3];
    R[2] = 2*q[0]*q[2] + 2*q[1]*q[3];
    R[3] = 2*q[0]*q[1] + 2*q[2]*q[3];
    R[4] = 1 - 2*q[0]*q[0] - 2*q[2]*q[2];
    R[5] = 2*q[1]*q[2] - 2*q[0]*q[3];
    R[6] = 2*q[0]*q[2] - 2*q[1]*q[3];
    R[7] = 2*q[1]*q[2] + 2*q[0]*q[3];
    R[8] = 1 - 2*q[0]*q[0] - 2*q[1]*q[1];
}

#endif
